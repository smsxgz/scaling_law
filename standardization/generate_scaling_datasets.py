"""
生成用于探究 logprob 和 tokens、params 之间 scaling law 的数据集
直接从 data/eval_results/0_shot/ 中的原始结果文件读取数据

数据包含：problem_id、model_name、params、tokens、logprob、is_correct
"""

import os
import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_model_data_from_result(result_file: Path) -> Tuple[str, List[Dict]]:
    """
    从单个结果文件中提取模型信息和logprobs数据

    Returns:
        (model_name, model_info, logprob_data_list)
    """
    logger.info(f"处理文件: {result_file}")

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 提取模型信息
    try:
        model_info = results["model_info"]
        model_name = model_info["name"]
        params = model_info["params"]
        tokens = model_info["tokens"]
    except KeyError as e:
        logger.warning(f"文件 {result_file} model_info 获取不正确，跳过")
        return None, []

    logprob_data = []

    # 提取samples数据
    samples_dict = results.get("samples", {})
    if not samples_dict:
        logger.warning(f"文件 {result_file} 中没有 samples 数据")
        return model_name, []

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
                "logprob": correct_choice_logprob,
                "is_correct": sample.get("acc", False),
            }
            logprob_data.append(item)

    logger.info(f"从 {model_name} 提取了 {len(logprob_data)} 条数据")
    if len(logprob_data) < 14042:
        logger.warning(f"{model_name}评测数据不全，跳过")
        logprob_data = []
    return model_name, logprob_data


def load_all_model_data(eval_results_dir: Path) -> pd.DataFrame:
    """
    加载所有模型的数据

    Returns:
        字典，键为模型名，值为该模型的所有问题数据
    """
    all_rows: List[Dict] = []

    # 查找所有 JSON 文件
    json_files = list(eval_results_dir.glob("*.json"))

    logger.info(f"找到 {len(json_files)} 个结果文件")

    for file_path in json_files:
        try:
            model_name, logprob_data = extract_model_data_from_result(file_path)

            if logprob_data:
                all_rows.extend(logprob_data)
                logger.info(f"成功加载模型 {model_name}: {len(logprob_data)} 个问题")
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    df = pd.DataFrame(all_rows)
    logger.info(f"已创建总 DataFrame, 总行数: {len(df)}")
    logger.info(f"总问题数 (problem_id): {df['problem_id'].nunique()}")
    logger.info(f"总模型数 (model_name): {df['model_name'].nunique()}")

    return df


def split_problems(problems: List[str], train_ratio: float = 0.6,
                   val_ratio: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
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


def add_zscore_by_group(df: pd.DataFrame, group_col: str = 'problem_id', target_col: str = 'logprob') -> pd.DataFrame:
    """
    对 DataFrame 按列分组, 计算 Z-score
    """
    if df.empty:
        return df
        
    logger.info(f"开始对 '{target_col}' 进行 Z-score 标准化 (按 {group_col} 分组)...")
    grouped = df.groupby(group_col)[target_col]
    mean = grouped.transform('mean')
    std = grouped.transform('std')
    
    df[f'{target_col}_zscore'] = (df[target_col] - mean) / std
    logger.info("Z-score 标准化完成。")
    return df


def log_and_save_dataset(df: pd.DataFrame, output_path: Path, dataset_name: str):
    """
    记录数据集统计信息并保存到 CSV
    """
    if df.empty:
        logger.warning(f"{dataset_name} 数据集为空, 已跳过保存。")
        return

    # 数据统计
    logger.info(f"{dataset_name} 数据集统计:")
    logger.info(f"  - 总行数: {len(df)}")
    logger.info(f"  - 问题数 (problem_id): {df['problem_id'].nunique()}")
    logger.info(f"  - 模型数 (model_name): {df['model_name'].nunique()}")
    logger.info(f"  - Logprob (原始) 范围: [{df['logprob'].min():.4f}, {df['logprob'].max():.4f}]")
    if 'logprob_zscore' in df.columns:
        logger.info(f"  - Logprob (Z-score) 范围: [{df['logprob_zscore'].min():.4f}, {df['logprob_zscore'].max():.4f}]")

    # 保存数据
    df.to_csv(output_path, index=False)
    logger.info(f"{dataset_name} 数据已保存到 {output_path}")


def main():
    # 设置随机种子以保证可重现性
    random.seed(42)

    # 路径设置
    eval_results_dir = Path("../data/eval_results/0_shot")
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)

    train_output_path = output_dir / "scaling_train_dataset.csv"
    val_output_path = output_dir / "scaling_validation_dataset.csv"
    test_output_path = output_dir / "scaling_test_dataset.csv"

    if not eval_results_dir.exists():
        logger.error(f"评估结果目录 {eval_results_dir} 不存在")
        return

    # 1. 一次性加载所有数据到 DataFrame
    logger.info("=" * 60)
    logger.info("开始加载所有模型数据到一个 DataFrame...")
    all_data_df = load_all_model_data(eval_results_dir)

    if all_data_df.empty:
        logger.error("没有找到有效的模型数据")
        return
    
    # 2. 先对总表进行 Z-score 标准化
    logger.info("=" * 60)
    all_data_df = add_zscore_by_group(all_data_df, 'problem_id', 'logprob')
    
    # 3. 提取所有不重复的问题ID
    all_problems_list = sorted(list(all_data_df['problem_id'].unique()))
    logger.info(f"总共找到 {len(all_problems_list)} 个不重复的问题")

    # 4. 划分数据集 (按问题ID)
    logger.info("=" * 60)
    logger.info("进行 60/20/20 问题集划分...")
    train_problems, val_problems, test_problems = split_problems(
        all_problems_list, train_ratio=0.6, val_ratio=0.2
    )
    logger.info(f"划分结果:")
    logger.info(f"  - 训练集问题数: {len(train_problems)}")
    logger.info(f"  - 验证集问题数: {len(val_problems)}")
    logger.info(f"  - 测试集问题数: {len(test_problems)}")

    # 5. 根据划分的问题ID列表, 筛选 DataFrame
    logger.info("=" * 60)
    logger.info("根据问题ID划分 DataFrame...")
    # 使用 .copy() 避免后续操作出现 SettingWithCopyWarning
    train_df = all_data_df[all_data_df['problem_id'].isin(train_problems)].copy()
    val_df = all_data_df[all_data_df['problem_id'].isin(val_problems)].copy()
    test_df = all_data_df[all_data_df['problem_id'].isin(test_problems)].copy()

    # 6. 保存数据集
    logger.info("=" * 60)
    log_and_save_dataset(train_df, train_output_path, "训练")
    logger.info("-" * 40)
    log_and_save_dataset(val_df, val_output_path, "验证")
    logger.info("-" * 40)
    log_and_save_dataset(test_df, test_output_path, "测试")

    logger.info("\n数据集生成完毕")

if __name__ == "__main__":
    main()