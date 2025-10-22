#!/usr/bin/env python3
"""
生成用于探究 loss 和 tokens、params 之间 scaling law 的数据集
直接从 eval_results/0_shot/ 中的原始结果文件读取数据

数据包含：problem_id、model_name、params、tokens、loss、is_correct
loss 是 logprobs 的相反数（非负值）
"""

import os
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_model_data_from_result(result_file: Path) -> Tuple[str, Dict, List[Dict]]:
    """
    从单个结果文件中提取模型信息和logprobs数据

    Returns:
        (model_name, model_info, logprob_data_list)
    """
    logger.info(f"处理文件: {result_file}")

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 提取模型信息
    model_info = results.get("model_info", {})
    model_name = model_info.get("name", "")

    if not model_name:
        # 从文件名推断模型名称
        model_name = result_file.stem
        logger.warning(f"从文件名推断模型名称: {model_name}")

    # 提取参数和token信息
    params = model_info.get("params", 0)
    tokens = model_info.get("tokens", 0)

    if not params or not tokens:
        logger.warning(f"模型 {model_name} 缺少 params 或 tokens 信息")

    logprob_data = []

    # 提取samples数据
    samples_dict = results.get("samples", {})
    if not samples_dict:
        logger.warning(f"文件 {result_file} 中没有 samples 数据")
        return model_name, model_info, []

    # MMLU 任务的结果按子任务分组
    for task_name, task_samples in samples_dict.items():
        for sample in task_samples:
            # 获取正确答案的索引
            correct_index = sample.get("gold", sample.get("target"))

            # 提取所有选项的 logprob 列表
            resps = sample.get("resps", [])
            if not resps:
                continue

            logprobs_list = []
            for resp in resps:
                if resp and len(resp) > 0 and len(resp[0]) > 0:
                    logprobs_list.append(resp[0][0])

            # 根据正确答案的索引，提取对应的 logprob
            correct_choice_logprob = None
            # 使用 taskname_docid 格式避免不同子任务的 doc_id 重复
            combined_doc_id = f"{task_name}_{sample.get('doc_id')}"

            if correct_index is not None and logprobs_list and 0 <= correct_index < len(logprobs_list):
                correct_choice_logprob = logprobs_list[correct_index]
            else:
                logger.debug(f"无法为 doc_id={combined_doc_id} 找到正确选项的 logprob")
                continue

            item = {
                "problem_id": combined_doc_id,
                "model_name": model_name,
                "params": params,
                "tokens": tokens,
                "logprobs": correct_choice_logprob,
                "task_name": task_name,
                "is_correct": sample.get("acc", False),
                "all_logprobs": logprobs_list,  # 保留所有选项的logprobs以备后用
            }
            logprob_data.append(item)

    logger.info(f"从 {model_name} 提取了 {len(logprob_data)} 条数据")
    return model_name, model_info, logprob_data


def load_all_model_data(eval_results_dir: Path) -> Dict[str, List[Dict]]:
    """
    加载所有模型的数据

    Returns:
        字典，键为模型名，值为该模型的所有问题数据
    """
    all_model_data = {}
    model_infos = {}

    # 查找所有 JSON 文件
    json_files = list(eval_results_dir.glob("*.json"))

    logger.info(f"找到 {len(json_files)} 个结果文件")

    for file_path in json_files:
        try:
            model_name, model_info, logprob_data = extract_model_data_from_result(file_path)

            if logprob_data:
                all_model_data[model_name] = logprob_data
                model_infos[model_name] = model_info
                logger.info(f"成功加载模型 {model_name}: {len(logprob_data)} 个问题")
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    return all_model_data


def find_common_problems(all_model_data: Dict[str, List[Dict]],
                         min_models_threshold: int = None) -> List[str]:
    """
    选择被足够多模型评估过的问题

    Args:
        all_model_data: 所有模型的数据
        min_models_threshold: 最少需要多少个模型评估过

    Returns:
        所有符合条件的问题的 problem_id 列表
    """
    if min_models_threshold is None:
        min_models_threshold = max(3, int(len(all_model_data) * 0.8))  # 至少80%的模型或最少3个
        logger.info(f"自动设置 min_models_threshold 为 {min_models_threshold}")

    # 统计每个问题被多少个模型评估过
    problem_counts = {}

    for model_name, model_data in all_model_data.items():
        for item in model_data:
            problem_id = item.get('problem_id')
            if problem_id:
                problem_counts[problem_id] = problem_counts.get(problem_id, 0) + 1

    # 筛选出被足够多模型评估过的问题
    eligible_problems = [
        problem_id for problem_id, count in problem_counts.items()
        if count >= min_models_threshold
    ]

    logger.info(f"找到 {len(eligible_problems)} 个被至少 {min_models_threshold} 个模型评估过的问题")
    logger.info(f"问题覆盖率统计: 最多 {max(problem_counts.values()) if problem_counts else 0} 个模型，"
                f"最少 {min(problem_counts.values()) if problem_counts else 0} 个模型")

    return sorted(eligible_problems)


def generate_dataset(all_model_data: Dict[str, List[Dict]],
                     selected_problems: List[str],
                     output_path: Path) -> pd.DataFrame:
    """
    生成数据集

    Args:
        all_model_data: 所有模型的数据
        selected_problems: 选中的问题 problem_id 列表
        output_path: 保存 CSV 文件的路径

    Returns:
        包含数据的 DataFrame
    """
    rows = []

    for problem_id in selected_problems:
        for model_name, model_data in all_model_data.items():
            # 查找该模型对这个问题的数据
            problem_data = None
            for item in model_data:
                if item.get('problem_id') == problem_id:
                    problem_data = item
                    break

            if problem_data is None:
                continue  # 该模型没有评估这个问题

            # 创建数据行
            row = {
                'problem_id': problem_id,
                'model_name': model_name,
                'params': problem_data['params'],
                'tokens': problem_data['tokens'],
                'loss': -problem_data['logprobs'],  # 取相反数，变成非负值
                'is_correct': problem_data.get('is_correct', False),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning(f"生成的数据集为空 (针对 {output_path})")
    else:
        # 数据统计
        logger.info(f"数据集统计 ({output_path}):")
        logger.info(f"  - 总行数: {len(df)}")
        logger.info(f"  - 问题数: {df['problem_id'].nunique()}")
        logger.info(f"  - 模型数: {df['model_name'].nunique()}")
        logger.info(f"  - 平均每个问题的模型数: {len(df) / df['problem_id'].nunique():.2f}")
        logger.info(f"  - Loss 范围: [{df['loss'].min():.4f}, {df['loss'].max():.4f}]")
        logger.info(f"  - 正确率: {df['is_correct'].mean():.2%}")

        # 保存数据
        df.to_csv(output_path, index=False)
        logger.info(f"数据已保存到 {output_path}")

    return df


def split_problems(problems: List[str], train_ratio: float = 0.6,
                   val_ratio: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
    """
    将问题列表划分为训练集、验证集和测试集

    Args:
        problems: 问题ID列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    Returns:
        (train_problems, val_problems, test_problems)
    """
    # 随机打乱问题列表
    shuffled_problems = problems.copy()
    random.shuffle(shuffled_problems)

    # 计算分割点
    n = len(shuffled_problems)
    train_split = int(n * train_ratio)
    val_split = int(n * (train_ratio + val_ratio))

    # 分割
    train_problems = shuffled_problems[:train_split]
    val_problems = shuffled_problems[train_split:val_split]
    test_problems = shuffled_problems[val_split:]

    return train_problems, val_problems, test_problems


def main():
    # 设置随机种子以保证可重现性
    random.seed(42)
    np.random.seed(42)

    # 路径设置
    eval_results_dir = Path("./eval_results/0_shot")
    output_dir = Path("./SLD")
    output_dir.mkdir(exist_ok=True)

    train_output_path = output_dir / "scaling_train_dataset.csv"
    val_output_path = output_dir / "scaling_validation_dataset.csv"
    test_output_path = output_dir / "scaling_test_dataset.csv"

    if not eval_results_dir.exists():
        logger.error(f"评估结果目录 {eval_results_dir} 不存在")
        return

    # 加载所有模型的数据
    logger.info("=" * 60)
    logger.info("开始加载所有模型的数据...")
    all_model_data = load_all_model_data(eval_results_dir)

    if not all_model_data:
        logger.error("没有找到有效的模型数据")
        return

    logger.info(f"成功加载了 {len(all_model_data)} 个模型的数据")

    # 选择通用问题
    logger.info("=" * 60)
    logger.info("查找被多个模型评估过的问题...")

    # 设置阈值：至少被80%的模型评估过，但不少于3个模型
    min_models = max(3, int(len(all_model_data) * 0.8))
    all_common_problems = find_common_problems(all_model_data, min_models)

    if not all_common_problems:
        logger.error("没有找到符合条件的问题")
        return

    # 划分数据集
    logger.info("=" * 60)
    logger.info(f"总共找到 {len(all_common_problems)} 个通用问题")
    logger.info("进行 60/20/20 数据集划分...")

    train_problems, val_problems, test_problems = split_problems(
        all_common_problems, train_ratio=0.6, val_ratio=0.2
    )

    logger.info(f"划分结果:")
    logger.info(f"  - 训练集: {len(train_problems)} 个问题")
    logger.info(f"  - 验证集: {len(val_problems)} 个问题")
    logger.info(f"  - 测试集: {len(test_problems)} 个问题")

    # 生成数据集
    logger.info("=" * 60)
    logger.info("生成训练数据集...")
    train_df = generate_dataset(all_model_data, train_problems, train_output_path)

    logger.info("-" * 40)
    logger.info("生成验证数据集...")
    val_df = generate_dataset(all_model_data, val_problems, val_output_path)

    logger.info("-" * 40)
    logger.info("生成测试数据集...")
    test_df = generate_dataset(all_model_data, test_problems, test_output_path)

    # 最终统计
    logger.info("=" * 60)
    logger.info("数据集生成完成！最终统计:")

    if not train_df.empty:
        logger.info(f"\n训练集 ({train_output_path.name}):")
        logger.info(f"  - 数据点数: {len(train_df)}")
        logger.info(f"  - 问题数: {train_df['problem_id'].nunique()}")
        logger.info(f"  - 模型数: {train_df['model_name'].nunique()}")

    if not val_df.empty:
        logger.info(f"\n验证集 ({val_output_path.name}):")
        logger.info(f"  - 数据点数: {len(val_df)}")
        logger.info(f"  - 问题数: {val_df['problem_id'].nunique()}")
        logger.info(f"  - 模型数: {val_df['model_name'].nunique()}")

    if not test_df.empty:
        logger.info(f"\n测试集 ({test_output_path.name}):")
        logger.info(f"  - 数据点数: {len(test_df)}")
        logger.info(f"  - 问题数: {test_df['problem_id'].nunique()}")
        logger.info(f"  - 模型数: {test_df['model_name'].nunique()}")

    logger.info("\n所有数据集已保存到 SLD 目录")


if __name__ == "__main__":
    main()