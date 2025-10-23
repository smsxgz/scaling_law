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

# Add parent directory to path to find data_loader
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import load_data

# --- Task Configuration ---
SUPPORTED_TASKS = {
    "logprobs_mixed_scaling_law"
}

os.environ["EVAL_TASK_NAME"] = "logprobs_mixed_scaling_law"

# --- 缓存和常量 ---
FIT_CACHE_SUFFIX = '.fit_cache.pkl'
EVAL_CACHE_SUFFIX = '.eval_cache.pkl'
LOG_EPS = np.finfo(float).eps

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
) -> Dict[str, Any]:
    """
    Calculates evaluation metrics for a single-dimension (1D) output.
    (此函数内容未改变)
    """
    try:
        pred = np.asarray(predictions, dtype=float)
        true = np.asarray(true_values, dtype=float)
    except (ValueError, TypeError):
        return get_failure_result("Could not convert predictions or true values to float arrays.")

    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("Predictions contain NaN or Inf values.")

    true = true.ravel()
    pred = pred.ravel()

    if true.shape != pred.shape:
        return get_failure_result(f"Shape mismatch after flattening: true values {true.shape} vs. predictions {pred.shape}.")
    if true.size == 0:
        return get_failure_result("Cannot evaluate on empty data.")

    test_mse = np.mean((true - pred) ** 2)
    test_mae = np.mean(np.abs(true - pred))

    variance = np.var(true)
    mean_abs_dev = np.mean(np.abs(true - np.mean(true)))

    eps = 1e-9 

    if variance > eps:
        nmse = test_mse / variance
    else:
        nmse = 0.0 if test_mse < eps else np.inf

    if mean_abs_dev > eps:
        nmae = test_mae / mean_abs_dev
    else:
        nmae = 0.0 if test_mae < eps else np.inf

    r2 = 1.0 - nmse
    
    results = {
        "nmse": float(nmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "combined_score": 1.0 / (1.0 + nmse),
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

def resolve_task_name(program_path: str) -> str:
    """Infers the task name from environment variables or the file path."""
    env_task = os.getenv("EVAL_TASK_NAME") or os.getenv("SCALING_TASK_NAME")
    if env_task and env_task in SUPPORTED_TASKS:
        return env_task

    p = Path(program_path)
    parts_to_check = [p.parent.name, p.stem]
    for part in parts_to_check:
        for task in SUPPORTED_TASKS:
            if task in part:
                return task

    raise ValueError(
        "Could not resolve task_name. Set env var EVAL_TASK_NAME or "
        f"ensure a supported task name (e.g., '{next(iter(SUPPORTED_TASKS))}') "
        "is in the script's parent folder or file name."
    )

# --- Evaluation Pipelines ---

def fit_model(
    program_path: str, 
    task_name: str,
    use_cache: bool = True
) -> Dict[str, Any]:
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
            train_data_dict = load_data(task_name, train=True, mode="final")
        except FileNotFoundError as e:
            return get_failure_result(f"Training data not found: {e}")
        if not train_data_dict:
            return get_failure_result("Training data is empty or failed to load.")

        all_X_train, all_y_train = [], []
        for key, (X_group, y_group) in train_data_dict.items():
            all_X_train.append(np.asarray(X_group))
            all_y_train.append(np.asarray(y_group))
        if not all_X_train:
                return get_failure_result("Training data contains no groups.")

        X_combined = np.concatenate(all_X_train, axis=0)
        y_combined = np.concatenate(all_y_train, axis=0)

        global_params = run_with_timeout(fit_scaling_law, args=(X_combined, y_combined))
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


# 在 evaluate.py 中，替换掉 _run_evaluation 函数

def _run_evaluation(
    program_path: str,
    task_name: str,
    fitted_params_map: Dict[Any, Any],
) -> Dict[str, Any]:
    """
    Evaluates the model on test data and returns detailed results for caching.
    (V2: 添加了 .ravel() 修复和调试打印)
    """
    try:
        program = _import_program(program_path)
        scaling_law_func = program.scaling_law_func
        
        try:
            calibration_mode = program.CALIBRATION_MODE
        except AttributeError:
            fail_result = get_failure_result(
                f"Program '{program_path}' is missing the required global constant 'CALIBRATION_MODE'."
            )
            return {"aggregate_metrics": fail_result, "per_group_metrics": []}
        
        if calibration_mode not in ['additive', 'multiplicative']:
            fail_result = get_failure_result(
                f"Invalid CALIBRATION_MODE: '{calibration_mode}'. Must be 'additive' or 'multiplicative'."
            )
            return {"aggregate_metrics": fail_result, "per_group_metrics": []}

        global_params = fitted_params_map.get('_global')
        if global_params is None:
            fail_result = get_failure_result("No '_global' parameters found in fitted_params_map.")
            return {"aggregate_metrics": fail_result, "per_group_metrics": []}

        try:
            test_data_dict = load_data(task_name, train=False, mode="final")
        except FileNotFoundError as e:
            fail_result = get_failure_result(f"Test data not found: {e}")
            return {"aggregate_metrics": fail_result, "per_group_metrics": []}
        if not test_data_dict:
            print("Warning: Test data is empty.", file=sys.stderr)
            return {
                "aggregate_metrics": {"nmse": 0.0, "nmae": 0.0, "r2": 1.0, "note": "Empty test set."}, 
                "per_group_metrics": []
            }

        all_calibrated_predictions = []
        all_true_values = []
        per_group_results = []

        for group_key, (X_test_group, y_test_group) in test_data_dict.items():
            if X_test_group.size == 0 or y_test_group.size == 0:
                continue

            pred_fixed_group = run_with_timeout(scaling_law_func, args=(X_test_group, global_params))
            if pred_fixed_group is None: continue
            
            # --- (修复 V2: 带有打印) ---
            true_loss_group = np.asarray(y_test_group, dtype=float)
            pred_fixed_group = np.asarray(pred_fixed_group, dtype=float)

            # (新) 调试打印: 检查你出问题的那个组
            is_debug_group = 'mmlu_high_school_computer_science' in group_key
            if is_debug_group:
                 print(f"\n[DEBUG] Group: {group_key}", file=sys.stderr)
                 print(f"[DEBUG] Original true_loss_group shape: {true_loss_group.shape}", file=sys.stderr)
                 print(f"[DEBUG] Original pred_fixed_group shape: {pred_fixed_group.shape}", file=sys.stderr)

            # 强制将两者都转换为 1D 数组 (N,)
            true_loss_group = true_loss_group.ravel()
            pred_fixed_group = pred_fixed_group.ravel()

            if is_debug_group:
                 print(f"[DEBUG] Ravelled true_loss_group shape: {true_loss_group.shape}", file=sys.stderr)
                 print(f"[DEBUG] Ravelled pred_fixed_group shape: {pred_fixed_group.shape}", file=sys.stderr)

            if true_loss_group.shape != pred_fixed_group.shape:
                print(f"Warning: Shape mismatch for group {group_key}. Skipping. True: {true_loss_group.shape}, Pred: {pred_fixed_group.shape}", file=sys.stderr)
                continue
            # --- (结束修复 V2) ---

            random_effect_value = 0.0
            
            if calibration_mode == 'multiplicative':
                ratios = (true_loss_group + LOG_EPS) / (pred_fixed_group + LOG_EPS)
                log_ratios = np.log(ratios)
                mean_log_ratio = np.nanmean(log_ratios)
                
                estimated_problem_factor = 1.0 if np.isnan(mean_log_ratio) or np.isinf(mean_log_ratio) else np.exp(mean_log_ratio)
                
                random_effect_value = estimated_problem_factor
                pred_calibrated_group = pred_fixed_group * estimated_problem_factor
            
            elif calibration_mode == 'additive':
                residuals = true_loss_group - pred_fixed_group
                estimated_problem_effect = np.nanmean(residuals)
                
                if is_debug_group:
                     print(f"[DEBUG] Residuals shape: {residuals.shape}", file=sys.stderr)
                     print(f"[DEBUG] Calculated Effect (mean residual): {estimated_problem_effect}", file=sys.stderr) # <-- 我们要看这个新值！

                random_effect_value = estimated_problem_effect
                pred_calibrated_group = pred_fixed_group + estimated_problem_effect
            
            # ... (函数其余部分不变) ...
            
            group_metrics = calculate_final_metrics(
                pred_calibrated_group, 
                true_loss_group
            )
            group_metrics['group_key'] = group_key
            group_metrics['random_effect'] = random_effect_value
            
            try:
                X_raw_numeric = np.asarray(X_test_group[:, 1:3], dtype=float)
                group_metrics['X_raw_numeric'] = X_raw_numeric
                group_metrics['y_true_raw'] = true_loss_group
                group_metrics['y_pred_calibrated_raw'] = pred_calibrated_group
            except Exception as e:
                print(f"Warning: Could not extract raw data for group {group_key}. Analysis may fail. Error: {e}", file=sys.stderr)
                group_metrics['X_raw_numeric'] = None
                group_metrics['y_true_raw'] = None
                group_metrics['y_pred_calibrated_raw'] = None
            
            per_group_results.append(group_metrics)
            all_calibrated_predictions.append(pred_calibrated_group)
            all_true_values.append(true_loss_group)

        if not all_calibrated_predictions:
            fail_result = get_failure_result("No predictions were generated for the test set.")
            return {"aggregate_metrics": fail_result, "per_group_metrics": []}

        final_predictions = np.concatenate(all_calibrated_predictions)
        final_true_values = np.concatenate(all_true_values)

        aggregate_metrics = calculate_final_metrics(
            final_predictions,
            final_true_values,
        )
        
        return {
            "aggregate_metrics": aggregate_metrics,
            "per_group_metrics": per_group_results
        }

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        fail_result = get_failure_result(str(e))
        return {"aggregate_metrics": fail_result, "per_group_metrics": []}


def evaluate(program_path: str, verbose: bool = False, force_refit: bool = False) -> Dict[str, Any]:
    """
    High-level, single-call evaluation function with evaluation caching.
    """
    try:
        task_name = resolve_task_name(program_path)
    except ValueError as e:
        return get_failure_result(str(e))
    
    fit_cache_path = Path(program_path).with_suffix(FIT_CACHE_SUFFIX)
    eval_cache_path = Path(program_path).with_suffix(EVAL_CACHE_SUFFIX)

    # 1. 检查评估缓存 (Evaluation Cache)
    if not force_refit and eval_cache_path.exists():
        try:
            fit_mtime = fit_cache_path.stat().st_mtime if fit_cache_path.exists() else -1
            eval_mtime = eval_cache_path.stat().st_mtime
            
            if fit_mtime <= eval_mtime:
                print(f"Loading cached evaluation from {eval_cache_path}", file=sys.stderr)
                with open(eval_cache_path, 'rb') as f:
                    final_results = pickle.load(f)
                
                # `verbose` 标志现在只用于打印, 缓存中已有所有数据
                if verbose:
                    final_results["task_name"] = task_name
                return final_results
            else:
                print("Fit cache is newer than eval cache. Re-evaluating.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load eval cache. Re-evaluating. Error: {e}", file=sys.stderr)

    # 2. 缓存未命中或强制刷新：执行拟合
    print("Running model fitting...", file=sys.stderr)
    fit_result = fit_model(program_path, task_name, use_cache=(not force_refit))
    if "fitted_params" not in fit_result:
        error = fit_result.get("error", "Unknown fitting error.")
        return get_failure_result(f"Fitting failed: {error}")

    fitted_params_map = fit_result["fitted_params"]

    # 3. 运行评估
    print("Running model evaluation...", file=sys.stderr)
    test_result = _run_evaluation(
        program_path,
        task_name,
        fitted_params_map=fitted_params_map,
    )

    # 4. 组合最终输出
    final_output = test_result.get("aggregate_metrics", {})
    final_output["per_group_metrics"] = test_result.get("per_group_metrics", [])
    
    # --- 优化 (重构) ---
    # 5. 将 *所有* 需要的数据（包括参数）保存到评估缓存
    final_output["fitted_params"] = fitted_params_map
    final_output["task_name"] = task_name # 缓存 task_name
    
    try:
        with open(eval_cache_path, 'wb') as f:
            pickle.dump(final_output, f)
        print(f"Saved complete evaluation results to {eval_cache_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to save eval cache. Error: {e}", file=sys.stderr)
    # --- 结束优化 ---
        
    if "error" in final_output:
        final_output["error"] = final_output.get("error")
        
    return final_output


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
    
    # `verbose=True` 确保 task_name 和 params 被添加到结果中（即使是从缓存加载）
    final_results = evaluate(args.program_path, verbose=True, force_refit=args.no_cache)

    task_name = final_results.get('task_name', 'N/A')
    print(f"Inferred Task: {task_name}")

    if "error" in final_results and final_results["error"]:
        print("\n--- ⛔ EVALUATION FAILED ⛔ ---")
        print(f"Error: {final_results['error']}")
        sys.exit(1)

    print("\n--- ✅ Final Test Results (Aggregate) ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  R-squared (R²):       {final_results.get('r2', 'N/A'):.6f}")
    print(f"  Combined Score:       {final_results.get('combined_score', 'N/A'):.6f}")

    # 打印拟合参数 (快速反馈)
    params = final_results.get('fitted_params', {})
    if params:
        print(f"\nFitted parameters ({len(params)} group(s)):")
        for key, val in params.items():
            param_val = np.asarray(val)
            if param_val.size > 1:
                param_str = np.array2string(param_val, precision=4, max_line_width=80, suppress_small=True)
            else:
                param_str = f"{param_val.item():.4f}"
            print(f"  - Group '{key}': {param_str}")
    
    print("\n--------------------------")
    print(f"Evaluation complete. Cache saved to {Path(args.program_path).with_suffix(EVAL_CACHE_SUFFIX)}")
    print(f"Run 'python analyze.py \"{args.program_path}\" --plot-groups 5 --plot-residuals' for details.")