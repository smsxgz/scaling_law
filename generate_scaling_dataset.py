#!/usr/bin/env python3
"""
生成用于探究 logprobs 和 tokens、params 之间 scaling law 的数据集

从 eval_results 中随机选择固定问题，收集对应的 logprobs 数据，
保存成可以被 SLD 脚本使用的格式。
"""

import os
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型元数据（与 analyze_scaling.py 保持一致）
MODELS_METADATA = {
    # Pythia
    "EleutherAI/pythia-14m": {"params": 14e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-31m": {"params": 31e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-70m": {"params": 70e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-160m": {"params": 160e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-410m": {"params": 410e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-1b": {"params": 1e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-1.4b": {"params": 1.4e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-2.8b": {"params": 2.8e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-6.9b": {"params": 6.9e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-12b": {"params": 12e9, "tokens": 300e9, "family": "pythia"},
    # OPT
    "facebook/opt-125m": {"params": 125e6, "tokens": 180e9, "family": "opt"},
    "facebook/opt-350m": {"params": 350e6, "tokens": 180e9, "family": "opt"},
    "facebook/opt-1.3b": {"params": 1.3e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-2.7b": {"params": 2.7e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-6.7b": {"params": 6.7e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-13b": {"params": 13e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-30b": {"params": 30e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-66b": {"params": 66e9, "tokens": 180e9, "family": "opt"},
}


def load_all_logprob_data(eval_results_dir: Path) -> Dict[str, List[Dict]]:
    """
    加载所有模型的 logprob 数据

    Returns:
        字典，键为模型名，值为该模型的所有问题数据
    """
    all_model_data = {}

    # 查找所有 _logprobs.json 文件
    logprob_files = list(eval_results_dir.glob("*_logprobs.json"))

    logger.info(f"找到 {len(logprob_files)} 个 logprob 文件")

    for file_path in logprob_files:
        # 从文件名解析模型名
        model_name_slug = file_path.name.replace("_logprobs.json", "")
        parts = model_name_slug.split('_')
        if len(parts) > 1 and parts[0] in ['facebook', 'EleutherAI']:
            model_name = f"{parts[0]}/{'_'.join(parts[1:])}"
        else:
            model_name = model_name_slug.replace('_', '/')

        # 检查模型是否在元数据中
        if model_name not in MODELS_METADATA:
            logger.warning(f"跳过模型 {model_name}，未在元数据中找到")
            continue

        # 加载数据
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_model_data[model_name] = data
            logger.info(f"加载模型 {model_name} 的数据，包含 {len(data)} 个问题")
        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {e}")

    return all_model_data


def select_common_questions(all_model_data: Dict[str, List[Dict]],
                          min_models_threshold: int = None,
                          num_questions: int = 100) -> List[int]:
    """
    选择被足够多模型评估过的问题

    Args:
        all_model_data: 所有模型的数据
        min_models_threshold: 最少需要多少个模型评估过
        num_questions: 要选择的问题数量

    Returns:
        选中问题的 doc_id 列表
    """
    if min_models_threshold is None:
        min_models_threshold = int(len(all_model_data) * 0.8)  # 至少80%的模型

    # 统计每个问题被多少个模型评估过
    question_counts = {}

    for model_name, model_data in all_model_data.items():
        for item in model_data:
            doc_id = item.get('doc_id')
            if doc_id is not None:
                question_counts[doc_id] = question_counts.get(doc_id, 0) + 1

    # 筛选出被足够多模型评估过的问题
    eligible_questions = [
        doc_id for doc_id, count in question_counts.items()
        if count >= min_models_threshold
    ]

    logger.info(f"找到 {len(eligible_questions)} 个被至少 {min_models_threshold} 个模型评估过的问题")

    # 随机选择指定数量的问题
    if len(eligible_questions) > num_questions:
        selected_questions = random.sample(eligible_questions, num_questions)
    else:
        selected_questions = eligible_questions
        logger.warning(f"可用问题数量 {len(eligible_questions)} 少于请求的 {num_questions} 个")

    logger.info(f"最终选择了 {len(selected_questions)} 个问题进行分析")
    return sorted(selected_questions)


def generate_scaling_dataset(all_model_data: Dict[str, List[Dict]],
                           selected_questions: List[int], output_path: Path) -> pd.DataFrame:
    """
    生成 scaling law 数据集

    Args:
        all_model_data: 所有模型的数据
        selected_questions: 选中的问题 doc_id 列表

    Returns:
        包含 scaling law 数据的 DataFrame
    """
    rows = []

    for question_id in selected_questions:
        # 为每个问题创建一个 group
        group_name = f"question_{question_id}"

        for model_name, model_data in all_model_data.items():
            # 查找该模型对这个问题的数据
            question_data = None
            for item in model_data:
                if item.get('doc_id') == question_id:
                    question_data = item
                    break

            if question_data is None:
                continue  # 该模型没有评估这个问题

            # 获取模型元数据
            model_info = MODELS_METADATA[model_name]

            # 获取 logprob 数据
            correct_logprob = question_data.get('correct_choice_logprob')
            if correct_logprob is None:
                continue

            # 创建数据行
            row = {
                'group': group_name,
                'params': model_info['params'],
                'tokens': model_info['tokens'],
                'logprobs': correct_logprob,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"生成的数据集包含 {len(df)} 行数据，涵盖 {df['group'].nunique()} 个问题组")

    df.to_csv(output_path, index=False)
    logger.info(f"数据已保存到 {output_path}")

    return df

def main():
    # 设置随机种子以保证可重现性
    random.seed(42)
    np.random.seed(42)

    # 路径设置
    eval_results_dir = Path("./eval_results")
    output_path = Path("./logprobs_scaling_dataset.csv")

    if not eval_results_dir.exists():
        logger.error(f"评估结果目录 {eval_results_dir} 不存在")
        return

    # 加载所有模型的数据
    logger.info("开始加载所有模型的 logprob 数据...")
    all_model_data = load_all_logprob_data(eval_results_dir)

    if not all_model_data:
        logger.error("没有找到有效的模型数据")
        return

    logger.info(f"成功加载了 {len(all_model_data)} 个模型的数据")

    # 选择通用问题
    logger.info("选择被多个模型评估过的问题...")
    selected_questions = select_common_questions(
        all_model_data,
        min_models_threshold=max(3, int(len(all_model_data) * 0.8)),  # 至少80%的模型
        num_questions=1  # 只选择1个问题进行分析
    )

    if not selected_questions:
        logger.error("没有找到符合条件的问题")
        return

    # 生成数据集
    logger.info("生成 scaling law 数据集...")
    df = generate_scaling_dataset(all_model_data, selected_questions)

    if df.empty:
        logger.error("生成的数据集为空")
        return

    # 同时保存包含更多信息的完整数据集
    full_output_path = Path("./logprobs_scaling_dataset_full.csv")
    df.to_csv(full_output_path, index=False)
    logger.info(f"完整数据集已保存到 {full_output_path}")

    logger.info("数据集生成完成!")


if __name__ == "__main__":
    main()