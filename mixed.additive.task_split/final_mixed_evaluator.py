# -*- coding: utf-8 -*-
"""
Unified Evaluator for Scaling Law Discovery.

(Refactored: Caching-only)

职责:
1. 运行模型拟合 (fit_model) -> .fit_cache.pkl
2. 运行模型评估 (_run_evaluation)
3. 将所有详细结果 (指标, X_raw, y_true, y_pred, params) 保存到 .eval_cache.pkl
"""
import argparse
import concurrent.futures
import importlib.util
import os
import sys
import traceback
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from data_loader import load_data
from mixed_evaluator import (
    calculate_final_metrics, 
    _import_program, 
    get_failure_result, 
    _calibrate_predictions
)

# --- 缓存和常量 ---
FIT_CACHE_SUFFIX = '.fit_cache.pkl'
EVAL_CACHE_SUFFIX = '.eval_cache.pkl'


def _fit_global_model(program_path: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Fits the scaling law model on the combined training data.
    """
    cache_path = Path(program_path).with_suffix(FIT_CACHE_SUFFIX)
    
    if use_cache and cache_path.exists():
        print(f"Loading cached parameters from {cache_path}", file=sys.stderr)
        try:
            with open(cache_path, 'rb') as f:
                fitted_params_map = pickle.load(f)
            return {"fitted_params": fitted_params_map}
        except Exception as e:
            print(f"Warning: Failed to load cache. Re-fitting. Error: {e}", file=sys.stderr)
            
    try:
        program = _import_program(program_path)
        fit_scaling_law = program.fit_scaling_law

        # --- FIT on training data ---
        try:
            train_data_dict = load_data(train=True, mode="final")
        except FileNotFoundError as e:
            return get_failure_result(f"Training data not found: {e}")
        if not train_data_dict:
            return get_failure_result("Training data is empty or failed to load.")

        all_X_train, all_y_train = [], []
        for task_name, task_data in train_data_dict.items():
            all_X_train.append(np.asarray(task_data["X"]))
            all_y_train.append(np.asarray(task_data["y"]))
            
        if not all_X_train:
                return get_failure_result("Training data contains no groups.")

        X_combined = np.concatenate(all_X_train, axis=0)
        y_combined = np.concatenate(all_y_train, axis=0)

        global_params = fit_scaling_law(X_combined, y_combined)
        
        if global_params is None:
            return get_failure_result("fit_scaling_law returned None.")

        new_fitted_params_map = {'_global': global_params}
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(new_fitted_params_map, f)
            print(f"Saved fitted parameters to {cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to save cache. Error: {e}", file=sys.stderr)
            
        return {"fitted_params": new_fitted_params_map}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result(f"Fitting failed: {str(e)}")


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
            pred_fixed_problem = scaling_law_func(X_problem, global_params)
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

    metrics_dict = calculate_final_metrics(
        final_predictions,
        final_true_values,
        final_task_labels, # Pass tasks to metrics function
    )

    if metrics_dict.get("error") is None:
        metrics_dict['y_pred_detailed'] = final_predictions
        metrics_dict['y_true_detailed'] = final_true_values
        metrics_dict['tasks_detailed'] = final_task_labels

    return metrics_dict


def evaluate(program_path: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    High-level, single-call evaluation function with caching.

    Orchestrates:
    1. Checks for a valid .eval_cache.pkl.
    2. Imports the user's program.
    3. Validates 'CALIBRATION_MODE'.
    4. Fits the model (using .fit_cache.pkl).
    5. Evaluates the fitted model on test data.
    6. Saves all results to .eval_cache.pkl.
    7. Returns results (pruned based on 'verbose' flag).
    """
    
    eval_cache_path = Path(program_path).with_suffix(EVAL_CACHE_SUFFIX)
    
    # --- 1. 检查评估缓存 ---
    if use_cache and eval_cache_path.exists():
        print(f"Loading cached evaluation results from {eval_cache_path}", file=sys.stderr)
        try:
            with open(eval_cache_path, 'rb') as f:
                full_eval_results = pickle.load(f)
                   
        except Exception as e:
            print(f"Warning: Failed to load eval cache. Re-evaluating. Error: {e}", file=sys.stderr)

    # --- 2. 缓存未命中或被禁用，执行完整评估 ---
    try:
        # --- 2a. 导入程序和验证模式 ---
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

        # --- 2b. 拟合训练数据 (使用拟合缓存) ---
        fit_result = _fit_global_model(
            program_path, 
            use_cache=use_cache
        )
        
        if "fitted_params" not in fit_result:
            error = fit_result.get("error", "Unknown fitting error.")
            return get_failure_result(f"Fitting failed: {error}")

        fitted_params_map = fit_result["fitted_params"]

        # --- 2c. 评估测试数据 ---
        test_result = _evaluate_calibrated_model(
            program,
            fitted_params_map,
            calibration_mode
        )
        
        if test_result.get("error"):
            return test_result # 评估失败，直接返回失败结果

        # --- 2d. 组合所有结果以进行缓存 ---
        full_eval_results = test_result.copy()
        full_eval_results['fitted_params'] = fitted_params_map

        # --- 2e. 保存到评估缓存 ---
        try:
            with open(eval_cache_path, 'wb') as f:
                pickle.dump(full_eval_results, f)
            print(f"Saved full evaluation results to {eval_cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to save eval cache. Error: {e}", file=sys.stderr)
        
        return full_eval_results

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Evaluator for Scaling Law Discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("program_path", type=str, help="Path to the Python script with scaling law functions.")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fitting and re-evaluation, ignoring all caches.")
    
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Path '{args.program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running Evaluation for Program: {args.program_path} ---")
    final_results = evaluate(args.program_path, use_cache=not args.no_cache)

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