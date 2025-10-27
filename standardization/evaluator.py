# -*- coding: utf-8 -*-
"""
Unified Evaluator for Scaling Law Discovery.
"""
import argparse
import concurrent.futures
import importlib.util
import os
import sys
import traceback
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
        "r2": -1.0,
        "combined_score": 0.0,
        "error": error_msg,
    }

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds: int = 1200):
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
) -> Dict[str, Any]:
    """
    Calculates evaluation metrics, correctly handling multi-dimensional outputs.

    For multi-dimensional targets, metrics (NMSE, NMAE) are calculated for each
    dimension separately and then averaged. The normalization factors (variance
    and mean absolute deviation) are computed using only the test data.

    Args:
        predictions: The model's predictions as a NumPy array.
        true_values: The ground truth values from the test set as a NumPy array.

    Returns:
        A dictionary containing aggregate and per-dimension metrics.
    """
    # 1. Initial validation and type conversion
    try:
        pred = np.asarray(predictions, dtype=float)
        true = np.asarray(true_values, dtype=float)
    except (ValueError, TypeError):
        return get_failure_result("Could not convert predictions or true values to float arrays.")

    # 2. Check for invalid values in predictions
    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("Predictions contain NaN or Inf values.")

    # 3. Reshape 1D arrays to 2D column vectors for consistent processing
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    # 4. Final shape validation
    if true.shape != pred.shape:
        return get_failure_result(f"Shape mismatch: true values {true.shape} vs. predictions {pred.shape}.")
    if true.size == 0:
        return get_failure_result("Cannot evaluate on empty data.")

    # 5. Calculate per-dimension errors on the test set
    test_mse_per_dim = np.mean((true - pred) ** 2, axis=0)
    test_mae_per_dim = np.mean(np.abs(true - pred), axis=0)

    # 6. Calculate normalizers using the test set only
    variance_per_dim = np.var(true, axis=0)
    mean_abs_dev_per_dim = np.mean(np.abs(true - np.mean(true, axis=0)), axis=0)

    # 7. Calculate normalized metrics, avoiding division by zero
    nmse_per_dim = np.divide(test_mse_per_dim, variance_per_dim,
                             out=np.full_like(test_mse_per_dim, np.inf), # Use np.inf where variance is zero
                             where=variance_per_dim > 1e-9)
    nmae_per_dim = np.divide(test_mae_per_dim, mean_abs_dev_per_dim,
                             out=np.full_like(test_mae_per_dim, np.inf), # Use np.inf where MAD is zero
                             where=mean_abs_dev_per_dim > 1e-9)

    # 8. Calculate R^2 for each dimension
    r2_per_dim = 1.0 - nmse_per_dim
    
    # 9. Average per-dimension metrics for final aggregate scores
    nmse = np.mean(nmse_per_dim)
    nmae = np.mean(nmae_per_dim)
    # The standard definition of R^2 relates to the total variance, so it's 1 - (total MSE / total variance)
    # which is equivalent to 1 - mean(nmse_per_dim) if variances are similar, but this is more direct.
    r2 = 1.0 - nmse

    # 10. Compile the results dictionary
    results = {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "combined_score": 1.0 / (1.0 + nmse),
    }

    # 11. Add per-dimension metrics for multi-dimensional targets
    if true.shape[1] > 1:
        results["nmse_per_dim"] = nmse_per_dim.tolist()
        results["nmae_per_dim"] = nmae_per_dim.tolist()
        results["r2_per_dim"] = r2_per_dim.tolist()

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

def evaluate_core(program_path: str, 
                  use_test_data: bool = False, 
                  fitted_params_map: Dict[Any, Any] = None) -> Dict[str, Union[float, Dict]]:

    try:
        program = _import_program(program_path)
        fit_scaling_law = program.fit_scaling_law
        scaling_law_func = program.scaling_law_func
        
        if not use_test_data:
            # --- FIT on training data ---
            X_train, y_train = load_data(train=True)
            if X_train.size == 0 or y_train.size == 0:
                return get_failure_result("No training data found.")

            params = run_with_timeout(fit_scaling_law, args=(X_train, y_train))
            
            new_fitted_params_map = {"default": params}
            return {"fitted_params": new_fitted_params_map}

        else:
            # --- EVALUATE on test data (validation set) ---
            if fitted_params_map is None or "default" not in fitted_params_map:
                return get_failure_result("fitted_params_map (with 'default' key) is required for evaluation.")

            X_test, y_test = load_data(train=False) 
            if X_test.size == 0 or y_test.size == 0:
                return get_failure_result("No test data found.")

            params = fitted_params_map["default"]          
            predictions = run_with_timeout(scaling_law_func, args=(X_test, params))

            return calculate_final_metrics(
                predictions,
                y_test,
            )

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result(str(e))

def evaluate(program_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    High-level, single-call evaluation function.
    """

    # 1. Fit on training data to get parameters
    fit_result = evaluate_core(program_path, use_test_data=False)
    
    if "fitted_params" not in fit_result:
        error = fit_result.get("error", "Unknown fitting error.")
        return get_failure_result(f"Fitting failed: {error}")

    fitted_params_map = fit_result["fitted_params"]

    # 2. Evaluate on test data using the fitted parameters
    test_result = evaluate_core(
        program_path,
        use_test_data=True,
        fitted_params_map=fitted_params_map,
    )

    # 3. Combine results into a comprehensive output
    if verbose:
        test_result["fitted_params"] = fitted_params_map
    return test_result

# --- Script Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluator for Scaling Law Discovery.")
    parser.add_argument("program_path", type=str, help="Path to the Python script with scaling law functions.")
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"Error: Path '{args.program_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Running Evaluation for Program: {args.program_path} ---")
    final_results = evaluate(args.program_path, verbose=True)

    if "error" in final_results and final_results["error"]:
        print("\n--- ⛔ EVALUATION FAILED ⛔ ---")
        print(f"Error: {final_results['error']}")
        sys.exit(1)

    print("\n--- ✅ Final Test Results (Aggregate) ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  R-squared (R²):        {final_results.get('r2', 'N/A'):.6f}")
    print(f"  Combined Score:        {final_results.get('combined_score', 'N/A'):.6f}")
    
    # Print per-dimension metrics if they exist
    if "nmse_per_dim" in final_results:
        print("\n  --- Per-Dimension Metrics ---")
        nmse_vals = final_results["nmse_per_dim"]
        nmae_vals = final_results["nmae_per_dim"]
        r2_vals = final_results["r2_per_dim"]
        for i, (nmse_d, nmae_d, r2_d) in enumerate(zip(nmse_vals, nmae_vals, r2_vals)):
            print(f"    Dim {i+1}: NMSE={nmse_d:.4f}, NMAE={nmae_d:.4f}, R²={r2_d:.4f}")

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