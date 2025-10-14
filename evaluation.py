#!/usr/bin/env python3

import os
import json
import logging
from datetime import datetime
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
MODELS = {
    # Pythia 系列 - 主要研究对象
    "pythia": [
        {"name": "EleutherAI/pythia-14m", "params": 14e6, "tokens": 300e9}, # Corrected param count
        {"name": "EleutherAI/pythia-31m", "params": 31e6, "tokens": 300e9}, # Corrected param count
        {"name": "EleutherAI/pythia-70m", "params": 70e6, "tokens": 300e9},
        {"name": "EleutherAI/pythia-160m", "params": 160e6, "tokens": 300e9},
        {"name": "EleutherAI/pythia-410m", "params": 410e6, "tokens": 300e9},
        {"name": "EleutherAI/pythia-1b", "params": 1e9, "tokens": 300e9},
        {"name": "EleutherAI/pythia-1.4b", "params": 1.4e9, "tokens": 300e9},
        {"name": "EleutherAI/pythia-2.8b", "params": 2.8e9, "tokens": 300e9},
        {"name": "EleutherAI/pythia-6.9b", "params": 6.9e9, "tokens": 300e9},
        {"name": "EleutherAI/pythia-12b", "params": 12e9, "tokens": 300e9},
    ],
    
    # OPT 系列
    "opt": [
        {"name": "facebook/opt-125m", "params": 125e6, "tokens": 180e9},
        {"name": "facebook/opt-350m", "params": 350e6, "tokens": 180e9},
        {"name": "facebook/opt-1.3b", "params": 1.3e9, "tokens": 180e9},
        {"name": "facebook/opt-2.7b", "params": 2.7e9, "tokens": 180e9},
        {"name": "facebook/opt-6.7b", "params": 6.7e9, "tokens": 180e9},
        {"name": "facebook/opt-13b", "params": 13e9, "tokens": 180e9},
        {"name": "facebook/opt-30b", "params": 30e9, "tokens": 180e9},
        {"name": "facebook/opt-66b", "params": 66e9, "tokens": 180e9},
    ],
}


# ==================== 评测配置 ====================
EVAL_CONFIG = {
    "tasks": ["mmlu"],
    "num_fewshot": 0,
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./eval_results",
    "log_samples": True,
}


def setup_output_dir():
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(EVAL_CONFIG["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def evaluate_model(model_info, output_dir):
    """评测单个模型"""
    model_name = model_info["name"]
    logger.info(f"开始评测模型: {model_name}")
    logger.info(f"参数量: {model_info['params']:.2e}, 训练tokens: {model_info['tokens']:.2e}")
    
    try:
        logger.info(f"加载模型...")
        lm = HFLM(
            pretrained=model_name,
            batch_size=EVAL_CONFIG["batch_size"],
            device=EVAL_CONFIG["device"],
            dtype="auto",
        )
        
        logger.info(f"开始评测...")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=EVAL_CONFIG["tasks"],
            num_fewshot=EVAL_CONFIG["num_fewshot"],
            batch_size=EVAL_CONFIG["batch_size"],
            log_samples=EVAL_CONFIG["log_samples"],
        )
        
        output_file = output_dir / f"{model_name.replace('/', '_')}.json"
        safe_json_dump(results, output_file)
        logger.info(f"结果已保存到: {output_file}")
        
        results["model_info"] = {
            "name": model_name,
            "params": model_info["params"],
            "tokens": model_info["tokens"],
            "tokens_per_param": model_info["tokens"] / model_info["params"] if model_info["params"] > 0 else 0,
        }
        
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


def extract_logprobs(results, output_dir):
    """从评测结果中提取每个问题的 logprob，特别是正确选项的 logprob"""
    if not results or "samples" not in results:
        logger.warning("没有样本数据可以提取 logprobs")
        return

    model_name = results["model_info"]["name"]
    logprob_data = []

    samples_list = results.get("samples", [])
    if isinstance(samples_list, dict):
        # MMLU 任务的结果按子任务分组，需要将其展平
        samples_list = [sample for task_samples in samples_list.values() for sample in task_samples]

    for sample in samples_list:
        # 1. 获取正确答案的索引 (通常是 'gold' 或 'target' 键)
        #    这个索引对应于 'choices' 和 'resps' 列表中的位置
        correct_index = sample.get("gold", sample.get("target"))

        # 2. 提取所有选项的 logprob 列表
        logprobs_list = [resp[0][0] for resp in sample.get("resps", []) if resp]
        

        # 3. 根据正确答案的索引，从列表中提取对应的 logprob
        correct_choice_logprob = None
        if correct_index is not None and logprobs_list and 0 <= correct_index < len(logprobs_list):
            correct_choice_logprob = logprobs_list[correct_index]
        else:
            logger.debug(f"无法为 doc_id={sample.get('doc_id')} 找到正确选项的 logprob。索引: {correct_index}, logprobs数量: {len(logprobs_list)}")

        choices = sample.get("doc", {}).get("choices", [])

        item = {
            "doc_id": sample.get("doc_id"),
            "question": sample.get("doc", {}).get("question", ""),
            "choices": choices,
            "gold_index": correct_index,
            "is_correct": sample.get("acc", False),
            "correct_choice_logprob": correct_choice_logprob,
            "all_logprobs": logprobs_list,
        }
        logprob_data.append(item)

    if logprob_data:
        logprob_file = output_dir / f"{model_name.replace('/', '_')}_logprobs.json"
        safe_json_dump(logprob_data, logprob_file)
        logger.info(f"Logprobs 已保存到: {logprob_file}")
    else:
        logger.warning(f"在 {model_name} 的结果中未找到可提取的样本。")

def run_evaluation_suite(model_series="pythia"):
    """运行完整的评测套件"""
    if model_series not in MODELS:
        logger.error(f"未知的模型系列: {model_series}")
        logger.info(f"可用的系列: {list(MODELS.keys())}")
        return
    
    output_dir = setup_output_dir()
    logger.info(f"结果将保存到: {output_dir}")
    
    all_results = []
    models_to_eval = MODELS[model_series]
    
    for i, model_info in enumerate(models_to_eval, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"进度: {i}/{len(models_to_eval)}")
        logger.info(f"{'='*60}\n")
        
        results = evaluate_model(model_info, output_dir)
        
        if results:
            all_results.append(results)
            extract_logprobs(results, output_dir)
    
    summary_file = output_dir / "summary.json"
    safe_json_dump(all_results, summary_file)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"所有评测完成! 结果保存在: {output_dir}")
    logger.info(f"汇总文件: {summary_file}")
    logger.info(f"{'='*60}\n")
    
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
    parser.add_argument("--series", type=str, default="pythia", choices=list(MODELS.keys()), help="要评测的模型系列")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot示例数量")
    
    args = parser.parse_args()
    
    EVAL_CONFIG["batch_size"] = args.batch_size
    EVAL_CONFIG["num_fewshot"] = args.num_fewshot
    
    run_evaluation_suite(model_series=args.series)