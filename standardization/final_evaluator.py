# -*- coding: utf-8 -*-
"""
最终测试脚本 (Final Test Script)

职责:
1. 在 'train' + 'validation' 数据集上拟合最终模型 (fit_model)。
2. 在 'test' 数据集上评估模型 (_run_evaluation)。
3. 将所有结果 (指标, X_test, y_test, y_pred, params) 保存到 .eval_final.pkl
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

# --- 缓存和常量 ---
FIT_CACHE_SUFFIX = '.fit_final.pkl'
EVAL_CACHE_SUFFIX = '.eval_final.pkl'

# --- Core Functions ---

def get_failure_result(error_msg: str = "Evaluation failed or timed out.") -> Dict[str, Any]:
    return {
        "nmse": 100000.0,
        "nmae": 100000.0,
        "r2": -1.0,
        "combined_score": 0.0,
        "error": error_msg,
    }

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds: int = 600):
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
    try:
        pred = np.asarray(predictions, dtype=float)
        true = np.asarray(true_values, dtype=float)
    except (ValueError, TypeError):
        return get_failure_result("无法将预测或真实值转换为浮点数组。")

    if np.isnan(pred).any() or np.isinf(pred).any():
        return get_failure_result("预测包含 NaN 或 Inf。")

    # 确保它们是 1D 数组
    true = true.ravel()
    pred = pred.ravel()

    if true.shape != pred.shape:
        return get_failure_result(f"展平后形状不匹配: 真实 {true.shape} vs. 预测 {pred.shape}.")
    if true.size == 0:
        return get_failure_result("无法在空数据上评估。")

    test_mse = np.mean((true - pred) ** 2)
    test_mae = np.mean(np.abs(true - pred))

    variance = np.var(true)
    mean_abs_dev = np.mean(np.abs(true - np.mean(true)))

    eps = 1e-9 # 避免除零

    if variance > eps:
        nmse = test_mse / variance
    else:
        # 如果方差为0（所有真实值都相同）
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
    """从给定路径导入 Python 模块。"""
    spec = importlib.util.spec_from_file_location("scaling_law_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从路径创建模块规范: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- 评估流程 ---

def fit_model(
    program_path: str, 
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    在合并的 (train + validation) 数据集上拟合 scaling law 模型。
    """
    cache_path = Path(program_path).with_suffix(FIT_CACHE_SUFFIX)
    
    if use_cache and cache_path.exists():
        print(f"从 {cache_path} 加载缓存的参数", file=sys.stderr)
        try:
            with open(cache_path, 'rb') as f:
                fitted_params_map = pickle.load(f)
            return {"fitted_params": fitted_params_map}
        except Exception as e:
            print(f"警告: 加载缓存失败。重新拟合。错误: {e}", file=sys.stderr)
            
    try:
        program = _import_program(program_path)
        fit_scaling_law = program.fit_scaling_law

        # --- 在 (train + validation) 数据上拟合 ---
        try:
            # data_loader 返回扁平化的 (X, y)
            X_combined, y_combined = load_data(train=True, mode="final")
        except FileNotFoundError as e:
            return get_failure_result(f"训练数据未找到: {e}")
        
        if X_combined.size == 0 or y_combined.size == 0:
            return get_failure_result("合并的训练数据为空。")

        global_params = run_with_timeout(fit_scaling_law, args=(X_combined, y_combined))
        if global_params is None:
            return get_failure_result("fit_scaling_law 返回 None。")

        new_fitted_params_map = {'default': global_params}
        
        # 保存到缓存
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(new_fitted_params_map, f)
            print(f"已保存拟合参数到 {cache_path}", file=sys.stderr)
        except Exception as e:
            print(f"警告: 保存缓存失败。错误: {e}", file=sys.stderr)
            
        return {"fitted_params": new_fitted_params_map}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return get_failure_result(f"拟合失败: {str(e)}")


def _run_evaluation(
    program_path: str,
    fitted_params_map: Dict[Any, Any],
) -> Dict[str, Any]:
    try:
        program = _import_program(program_path)
        scaling_law_func = program.scaling_law_func
        
        global_params = fitted_params_map.get('default')
        if global_params is None:
            fail_result = get_failure_result("在 fitted_params_map 中未找到 'default' 参数。")
            return {"aggregate_metrics": fail_result, "analysis_data": {}}

        try:
            X_test, y_test = load_data(train=False, mode="final")
        except FileNotFoundError as e:
            fail_result = get_failure_result(f"测试数据未找到: {e}")
            return {"aggregate_metrics": fail_result, "analysis_data": {}}
        
        if X_test.size == 0 or y_test.size == 0:
            print("警告: 测试数据为空。", file=sys.stderr)
            return {
                "aggregate_metrics": {"nmse": 0.0, "nmae": 0.0, "r2": 1.0, "note": "空测试集。"}, 
                "analysis_data": {}
            }

        y_pred = run_with_timeout(scaling_law_func, args=(X_test, global_params))
        if y_pred is None: 
            fail_result = get_failure_result("scaling_law_func 返回 None。")
            return {"aggregate_metrics": fail_result, "analysis_data": {}}
            
        aggregate_metrics = calculate_final_metrics(
            y_pred, 
            y_test
        )
        
        analysis_data = {
            "X_test": X_test,
            "y_test": y_test.ravel(), # 确保 1D
            "y_pred": y_pred.ravel()  # 确保 1D
        }
        
        return {
            "aggregate_metrics": aggregate_metrics,
            "analysis_data": analysis_data
        }

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        fail_result = get_failure_result(str(e))
        return {"aggregate_metrics": fail_result, "analysis_data": {}}


def evaluate(program_path: str, verbose: bool = False, force_refit: bool = False) -> Dict[str, Any]:
    fit_cache_path = Path(program_path).with_suffix(FIT_CACHE_SUFFIX)
    eval_cache_path = Path(program_path).with_suffix(EVAL_CACHE_SUFFIX)

    # 1. 检查评估缓存 (Evaluation Cache)
    if not force_refit and eval_cache_path.exists():
        try:
            fit_mtime = fit_cache_path.stat().st_mtime if fit_cache_path.exists() else -1
            eval_mtime = eval_cache_path.stat().st_mtime
            
            # 如果拟合缓存不比评估缓存新
            if fit_mtime <= eval_mtime:
                print(f"从 {eval_cache_path} 加载缓存的评估", file=sys.stderr)
                with open(eval_cache_path, 'rb') as f:
                    final_results = pickle.load(f)
                return final_results
            else:
                print("拟合缓存比评估缓存新。重新评估。", file=sys.stderr)
        except Exception as e:
            print(f"警告: 加载评估缓存失败。重新评估。错误: {e}", file=sys.stderr)

    # 2. 缓存未命中或强制刷新：执行拟合
    print("运行模型拟合 (train + validation)...", file=sys.stderr)
    fit_result = fit_model(program_path, use_cache=(not force_refit))
    
    if "fitted_params" not in fit_result:
        error = fit_result.get("error", "Unknown fitting error.")
        return get_failure_result(f"拟合失败: {error}")

    fitted_params_map = fit_result["fitted_params"]

    # 3. 运行评估 (on test set)
    print("运行模型评估 (test set)...", file=sys.stderr)
    test_result = _run_evaluation(
        program_path,
        fitted_params_map=fitted_params_map,
    )

    # 4. 组合最终输出
    final_output = test_result.get("aggregate_metrics", {})
    final_output["analysis_data"] = test_result.get("analysis_data", {})
    final_output["fitted_params"] = fitted_params_map
    
    # 5. 保存到评估缓存
    try:
        with open(eval_cache_path, 'wb') as f:
            pickle.dump(final_output, f)
        print(f"已将完整评估结果保存到 {eval_cache_path}", file=sys.stderr)
    except Exception as e:
        print(f"警告: 保存评估缓存失败。错误: {e}", file=sys.stderr)
        
    if "error" in final_output:
        final_output["error"] = final_output.get("error")
        
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final Test Script for Scaling Law Discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("program_path", type=str, help="Path to the Python script (e.g., 'init_program.py').")
    parser.add_argument("--no-cache", action="store_true", help="强制重新拟合和评估，忽略所有缓存。")
    
    args = parser.parse_args()

    if not os.path.exists(args.program_path):
        print(f"错误: 路径 '{args.program_path}' 不存在。", file=sys.stderr)
        sys.exit(1)

    print(f"--- 运行最终测试: {args.program_path} ---")
    
    final_results = evaluate(args.program_path, verbose=True, force_refit=args.no_cache)

    if "error" in final_results and final_results["error"]:
        print("\n--- ⛔ 评估失败 ⛔ ---")
        print(f"错误: {final_results['error']}")
        sys.exit(1)

    print("\n--- ✅ 最终测试集结果 (聚合) ---")
    print(f"  Normalized MSE (NMSE): {final_results.get('nmse', 'N/A'):.6f}")
    print(f"  Normalized MAE (NMAE): {final_results.get('nmae', 'N/A'):.6f}")
    print(f"  R-squared (R²):        {final_results.get('r2', 'N/A'):.6f}")
    print(f"  Combined Score:        {final_results.get('combined_score', 'N/A'):.6f}")

    params = final_results.get('fitted_params', {})
    if 'default' in params:
        print(f"\n最终拟合参数 ('default'):")
        val = params['default']
        param_val = np.asarray(val)
        if param_val.size > 1:
            param_str = np.array2string(param_val, precision=4, max_line_width=80, suppress_small=True)
        else:
            param_str = f"{param_val.item():.8f}"
        print(f"  - {param_str}")
    
    print("\n--------------------------")
    print(f"评估完成。缓存保存到 {Path(args.program_path).with_suffix(EVAL_CACHE_SUFFIX)}")