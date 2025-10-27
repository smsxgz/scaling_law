#!/usr/bin/env python3

import os
import json
import logging
import csv
from pathlib import Path
import torch
import numpy as np
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== JSON 序列化辅助函数 ====================
class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 和 torch 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.dtype):  # Handle numpy dtypes
            return str(obj)
        # --- THIS IS THE NEW, CRITICAL FIX ---
        elif isinstance(obj, torch.dtype): # Handle torch dtypes
            return str(obj)
        # ------------------------------------
        return super(NumpyEncoder, self).default(obj)


def safe_json_dump(data, file_path):
    """安全地保存 JSON，处理各种类型转换"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


# ==================== 模型配置 ====================
def parse_tokens(token_str):
    """解析 tokens 字符串，e.g., '300B' -> 300e9, '2T' -> 2e12"""
    token_str = str(token_str).strip().upper()
    multipliers = {
        'M': 1e6,
        'B': 1e9,
        'T': 1e12,
    }
    for suffix, multiplier in multipliers.items():
        if token_str.endswith(suffix):
            return float(token_str[:-1]) * multiplier
    return float(token_str)


def load_models_from_csv(csv_path="models.csv"):
    """从 CSV 文件加载模型列表"""
    models = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model_name'].strip():
                models.append({
                    "name": row['model_name'].strip(),
                    "tokens": parse_tokens(row['tokens'].strip()),
                })
    return models


# ==================== 评测配置 ====================
EVAL_CONFIG = {
    "tasks": ["mmlu"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./eval_results",
    "log_samples": True,
}


def setup_output_dir(num_fewshot):
    """创建输出目录：eval_results/{num_fewshot}_shot"""
    output_dir = Path(EVAL_CONFIG["output_dir"]) / f"{num_fewshot}_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def is_model_evaluated(model_name, output_dir):
    """检查模型是否已经评测过"""
    model_file = output_dir / f"{model_name.replace('/', '_')}.json"
    return model_file.exists()


def get_completed_models(output_dir):
    """获取已完成的模型列表"""
    if not output_dir.exists():
        return set()
    return {f.stem for f in output_dir.glob("*.json")}


def evaluate_model(model_info, output_dir, batch_size=8, num_fewshot=0):
    """评测单个模型"""
    model_name = model_info["name"]
    logger.info(f"开始评测模型: {model_name}")

    try:
        logger.info(f"加载模型...")
        lm = HFLM(
            pretrained=model_name,
            batch_size=batch_size,
            device=EVAL_CONFIG["device"],
            dtype="auto",
            trust_remote_code=True,
            parallelize=True,
        )

        # 获取参数量
        params = lm.model.num_parameters()
        if params is None:
            logger.error(f"无法获取模型 {model_name} 的参数量，跳过")
            return None

        logger.info(f"参数量: {params:.2e}, 训练tokens: {model_info['tokens']:.2e}")

        logger.info(f"开始评测...")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=EVAL_CONFIG["tasks"],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            log_samples=EVAL_CONFIG["log_samples"],
        )

        results["model_info"] = {
            "name": model_name,
            "params": params,
            "tokens": model_info["tokens"],
            "tokens_per_param": model_info["tokens"] / params if params > 0 else 0,
        }

        output_file = output_dir / f"{model_name.replace('/', '_')}.json"
        safe_json_dump(results, output_file)
        logger.info(f"结果已保存到: {output_file}")

        if "results" in results and "mmlu" in results["results"]:
            acc = results["results"]["mmlu"].get("acc", results["results"]["mmlu"].get("acc,none", 0))
            logger.info(f"完成评测: {model_name}")
            logger.info(f"MMLU准确率: {acc:.4f}")
        else:
            logger.warning(f"未找到 MMLU 准确率指标")

        return results

    except Exception as e:
        logger.error(f"评测失败 {model_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_evaluation_suite(batch_size=8, num_fewshot=0, csv_path="models.csv"):
    """运行完整的评测套件"""
    # 加载模型列表
    models = load_models_from_csv(csv_path)
    if not models:
        logger.error(f"未能从 {csv_path} 加载任何模型")
        return

    output_dir = setup_output_dir(num_fewshot)
    logger.info(f"结果将保存到: {output_dir}")

    # 获取已完成的模型
    completed_models = get_completed_models(output_dir)
    logger.info(f"已完成的模型: {len(completed_models)}/{len(models)}")
    if completed_models:
        logger.info(f"跳过已完成的模型: {completed_models}")

    all_results = []
    models_to_eval = [m for m in models if m["name"].replace('/', '_') not in completed_models]

    logger.info(f"待评测模型数: {len(models_to_eval)}")

    for i, model_info in enumerate(models_to_eval, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"进度: {i}/{len(models_to_eval)} (总体: {len(completed_models) + i}/{len(models)})")
        logger.info(f"{'='*60}\n")

        results = evaluate_model(model_info, output_dir, batch_size=batch_size, num_fewshot=num_fewshot)

        if results:
            all_results.append(results)

    logger.info(f"\n{'='*60}")
    logger.info(f"所有评测完成! 结果保存在: {output_dir}")
    logger.info(f"{'='*60}\n")

    # 汇总结果
    print("\n模型性能汇总:")
    print(f"{'模型名称':<50} {'参数量':<12} {'MMLU准确率':<12}")
    print("-" * 80)
    for result in all_results:
        model_name = result["model_info"]["name"]
        params = result["model_info"]["params"]
        acc = 0.0
        if "results" in result and "mmlu" in result["results"]:
            acc = result["results"]["mmlu"].get("acc", result["results"]["mmlu"].get("acc,none", 0.0))
        print(f"{model_name:<50} {params:>12.2e} {acc:>12.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量评测LLM模型")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot示例数量")
    parser.add_argument("--csv", type=str, default="models.csv", help="模型列表CSV文件路径")

    args = parser.parse_args()

    run_evaluation_suite(batch_size=args.batch_size, num_fewshot=args.num_fewshot, csv_path=args.csv)
