# -*- coding: utf-8 -*-
"""
Unified Evaluator for Scaling Law Discovery.
"""
import argparse
import concurrent.futures
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from data_loader import load_data

# --- Core Functions ---

def get_failure_result(error_msg: str = "Evaluation failed or timed out.") -> Dict[str, Any]:
    """Returns a standardized dictionary for failure cases."""
    return {
        "nmse": 100000.0,
        "nmae": 100000.0,
        "combined_score": 0.0,
        "error": error_msg,
    }

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds: int = 600):
    """Runs a function with a specified timeout, raising an exception on timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except Exception as e:
            print(f"Function {func.__name__} timed out or failed: {e}", file=sys.stderr)
            raise

def calculate_final_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
    tasks: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculates evaluation metrics, including aggregate and per-task scores.
    
    Assumes 1D inputs for predictions and true_values.

    Args:
        predictions: The model's predictions (1D NumPy array).
        true_values: The ground truth values (1D NumPy array).
        tasks: An array of task name labels, (1D, str, same length as predictions).

    Returns:
        A dictionary containing aggregate and per-task metrics.
    """
    # 1. Initial validation and type conversion
    try:
        pred = np.asarray(predictions, dtype=float).flatten()
        true = np.asarray(true_values, dtype=float).flatten()
        tasks = np.asarray(tasks, dtype=str).flatten()
    except (ValueError, TypeError):
        return get_failure_result("Could not convert predictions, true values, or tasks to arrays.")

    # 2. Check for invalid values in predictions
    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("Predictions contain NaN or Inf values.")

    # 3. Final shape validation
    if true.shape != pred.shape or true.shape != tasks.shape:
        return get_failure_result(
            f"Shape mismatch: true {true.shape}, predictions {pred.shape}, tasks {tasks.shape}."
        )
    if true.size == 0:
        return get_failure_result("Cannot evaluate on empty data.")

    # 4. Calculate Aggregate Metrics (on all test data)
    test_mse_agg = np.mean((true - pred) ** 2)
    test_mae_agg = np.mean(np.abs(true - pred))
    variance_agg = np.var(true)
    mean_abs_dev_agg = np.mean(np.abs(true - np.mean(true)))

    # 5. Calculate normalized aggregate metrics, avoiding division by zero
    nmse = test_mse_agg / (variance_agg + 1e-9)
    nmae = test_mae_agg / (mean_abs_dev_agg + 1e-9)
    combined_score = 1.0 / (1.0 + nmse)

    # 6. Calculate Per-Task Metrics (Request 1)
    per_task_metrics = {}
    unique_tasks = np.unique(tasks)
    
    for task_name in unique_tasks:
        try:
            task_mask = (tasks == task_name)
            pred_task = pred[task_mask]
            true_task = true[task_mask]

            if true_task.size == 0:
                continue

            # Calculate metrics for this specific task
            mse_task = np.mean((true_task - pred_task) ** 2)
            mae_task = np.mean(np.abs(true_task - pred_task))
            var_task = np.var(true_task)
            mad_task = np.mean(np.abs(true_task - np.mean(true_task)))

            # Normalize using *this task's* variance and MAD
            nmse_task = mse_task / (var_task + 1e-9)
            nmae_task = mae_task / (mad_task + 1e-9)
            
            per_task_metrics[task_name] = {
                "nmse": float(nmse_task), 
                "nmae": float(nmae_task), 
            }
        except Exception as e:
            print(f"Warning: Failed to calculate metrics for task '{task_name}': {e}", file=sys.stderr)
            per_task_metrics[task_name] = {"error": str(e)}


    # 7. Compile the results dictionary
    results = {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "combined_score": float(combined_score),
        "per_task_metrics": per_task_metrics,
    }

    return results


def _import_program(program_path: str):
    """Imports a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from path: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- Evaluation Pipelines ---

def _fit_global_model(program) -> Dict[str, Any]:
    """Loads training data, concatenates it, and fits the global model."""
    fit_scaling_law = program.fit_scaling_law

    try:
        # load_data groups by task_name
        train_data_dict = load_data(train=True, mode="evolve")
    except FileNotFoundError as e:
        return get_failure_result(f"Training data not found: {e}")
    if not train_data_dict:
        return get_failure_result("Training data is empty or failed to load.")

    all_X_train, all_y_train = [], []
    # Iterate correctly over the new data structure
    for task_name, task_data in train_data_dict.items():
        all_X_train.append(np.asarray(task_data["X"]))
        all_y_train.append(np.asarray(task_data["y"]))
        
    if not all_X_train:
         return get_failure_result("Training data contains no groups.")

    X_combined = np.concatenate(all_X_train, axis=0)
    y_combined = np.concatenate(all_y_train, axis=0)

    global_params = run_with_timeout(fit_scaling_law, args=(X_combined, y_combined))
    if global_params is None:
        return get_failure_result("fit_scaling_law returned None.")

    new_fitted_params_map = {'_global': global_params}
    return {"fitted_params": new_fitted_params_map}


def _calibrate_predictions(pred_fixed, true_loss, mode, eps):
    """Helper to perform additive or multiplicative calibration."""
    if mode == 'multiplicative':
        # L_total = L_fixed * K_i
        # Estimate K_i using the Geometric Mean of ratios
        
        # Calculate ratios, ensuring inputs are positive
        ratios = (true_loss + eps) / (pred_fixed + eps)
        
        # Work in log-space to compute geometric mean
        log_ratios = np.log(ratios)
        
        # This is the log of the geometric mean
        mean_log_ratio = np.nanmean(log_ratios)
        
        if np.isnan(mean_log_ratio) or np.isinf(mean_log_ratio):
            estimated_problem_factor = 1.0
        else:
            # Convert back to linear space
            estimated_problem_factor = np.exp(mean_log_ratio)
        
        return pred_fixed * estimated_problem_factor
    
    elif mode == 'additive':
        # L_total = L_fixed + C_i
        # Estimate C_i using the Arithmetic Mean of residuals
        residuals = true_loss - pred_fixed
        estimated_problem_effect = np.nanmean(residuals)
        
        if np.isnan(estimated_problem_effect) or np.isinf(estimated_problem_effect):
            estimated_problem_effect = 0.0

        return pred_fixed + estimated_problem_effect
    
    return None


def _evaluate_calibrated_model(program, fitted_params_map, calibration_mode):
    """Loads test data, runs calibration, and returns metrics."""
    scaling_law_func = program.scaling_law_func

    if fitted_params_map is None:
        return get_failure_result("fitted_params_map is required for evaluation.")
    global_params = fitted_params_map.get('_global')
    if global_params is None:
        return get_failure_result("No '_global' parameters found in fitted_params_map.")

    try:
        # load_data groups by task_name
        test_data_dict = load_data(train=False, mode="evolve")
    except FileNotFoundError as e:
        return get_failure_result(f"Test data not found: {e}")
    if not test_data_dict:
        return get_failure_result("Warning: Test data is empty.")

    all_calibrated_predictions = []
    all_true_values = []
    all_task_labels = []

    # Use a small epsilon to prevent log(0) or division by zero
    eps = np.finfo(float).eps
    
    # Loop 1: By Task
    for task_name, task_data in test_data_dict.items():
        X_task = task_data["X"]
        y_task = task_data["y"]
        
        if X_task.size == 0 or y_task.size == 0:
            continue
            
        problem_ids_task = X_task[:, 0] # Get problem_id column
        unique_problems_in_task = np.unique(problem_ids_task)
        
        # Loop 2: By Problem (for calibration)
        for problem_id in unique_problems_in_task:
            problem_mask = (problem_ids_task == problem_id)
            X_problem = X_task[problem_mask]
            y_problem = y_task[problem_mask]

            if X_problem.size == 0 or y_problem.size == 0:
                continue

            # Get fixed-effect predictions for this problem
            pred_fixed_problem = run_with_timeout(scaling_law_func, args=(X_problem, global_params))
            if pred_fixed_problem is None: continue
            
            true_loss_problem = y_problem
            
            # Use the helper function
            pred_calibrated_problem = _calibrate_predictions(
                pred_fixed_problem, 
                true_loss_problem, 
                calibration_mode, 
                eps
            )
            
            if pred_calibrated_problem is not None:
                all_calibrated_predictions.append(pred_calibrated_problem)
                all_true_values.append(true_loss_problem)
                # Add task labels corresponding to these data points
                all_task_labels.append(np.full(pred_calibrated_problem.shape, task_name))

    if not all_calibrated_predictions:
        return get_failure_result("No predictions were generated for the test set.")

    final_predictions = np.concatenate(all_calibrated_predictions)
    final_true_values = np.concatenate(all_true_values)
    final_task_labels = np.concatenate(all_task_labels) # Concat task labels

    return calculate_final_metrics(
        final_predictions,
        final_true_values,
        final_task_labels, # Pass tasks to metrics function
    )


def evaluate(program_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    High-level, single-call evaluation function.

    This orchestrates the entire process:
    1. Imports the user's program.
    2. Validates the 'CALIBRATION_MODE'.
    3. Fits the model on training data.
    4. Evaluates the fitted model on test data.
    5. Returns a dictionary with final metrics and fitted parameters.

    Args:
        program_path: Path to the user's Python script with scaling law functions.

    Returns:
        A dictionary containing the evaluation results.
    """
    try:
        # --- 1. Import Program and Validate Mode ---
        program = _import_program(program_path)
        
        try:
            calibration_mode = program.CALIBRATION_MODE
        except AttributeError:
            return get_failure_result(
                f"Program '{program_path}' is missing the required global constant 'CALIBRATION_MODE'."
            )
        
        if calibration_mode not in ['additive', 'multiplicative']:
                return get_failure_result(
                    f"Invalid CALIBRATION_MODE: '{calibration_mode}'. Must be 'additive' or 'multiplicative'."
                )

        # --- 2. Fit on training data ---
        fit_result = _fit_global_model(program)
        
        if "fitted_params" not in fit_result:
            error = fit_result.get("error", "Unknown fitting error.")
            return get_failure_result(f"Fitting failed: {error}")

        fitted_params_map = fit_result["fitted_params"]

        # --- 3. Evaluate on test data ---
        test_result = _evaluate_calibrated_model(
            program,
            fitted_params_map,
            calibration_mode
        )

        if verbose:
            test_result["fitted_params"] = fitted_params_map
        
        return test_result

    except Exception as e:
        return get_failure_result(str(e))

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluator for Scaling Law Discovery.")
    parser.add_argument("program_path", type=str, help="Path to the Python script with scaling law functions.")
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Path '{args.program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running Evaluation for Program: {args.program_path} ---")
    final_results = evaluate(args.program_path)

    if "error" in final_results and final_results["error"]:
        print("\n--- ⛔ EVALUATION FAILED ⛔ ---")
        print(f"Error: {final_results['error']}")
        sys.exit(1)

    print("\n--- ✅ Final Test Results (Aggregate) ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  Combined Score:        {final_results.get('combined_score', 'N/A'):.6f}")
    
    if "per_task_metrics" in final_results and final_results["per_task_metrics"]:
        print("\n --- Per-Task Metrics ---")
        try:
            # Sort by task name for consistent output
            sorted_tasks = sorted(final_results["per_task_metrics"].items())
            for task_name, metrics in sorted_tasks:
                if "error" in metrics:
                    print(f"  Task '{task_name}': FAILED ({metrics['error']})")
                    continue
                
                nmse_t = metrics.get('nmse', np.nan)
                nmae_t = metrics.get('nmae', np.nan)
                print(f"  Task '{task_name}': NMSE={nmse_t: <8.4f}, NMAE={nmae_t: <8.4f}")
        except Exception as e:
            print(f"  Failed to display per-task metrics: {e}")

    params = final_results.get('fitted_params', {})
    if params:
        print(f"\nFitted parameters for {len(params)} group(s):")
        for key, val in params.items():
            param_val = np.asarray(val)
            if param_val.size > 1:
                param_str = np.array2string(param_val, precision=4, max_line_width=80, suppress_small=True)
            else:
                param_str = f"{param_val.item():.4f}" # Use .item() for single-element arrays
            print(f"  - Group '{key}': {param_str}")
    print("--------------------------")