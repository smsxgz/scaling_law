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

# MMLU 总问题数
MMLU_TOTAL_PROBLEMS = 14042


def extract_model_data_from_result(result_file: Path) -> Tuple[str, List[Dict]]:
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

            logprob_list = []
            for resp in resps:
                if resp and len(resp) > 0 and len(resp[0]) > 0:
                    logprob_list.append(resp[0][0])

            # 根据正确答案的索引，提取对应的 logprob
            correct_choice_logprob = None
            # 使用 taskname_docid 格式避免不同子任务的 doc_id 重复
            combined_doc_id = f"{task_name}_{sample.get('doc_id')}"

            if correct_index is not None and logprob_list and 0 <= correct_index < len(logprob_list):
                correct_choice_logprob = logprob_list[correct_index]
            else:
                logger.debug(f"无法为 doc_id={combined_doc_id} 找到正确选项的 logprob")
                continue

            item = {
                "problem_id": combined_doc_id,
                "task_name": task_name,
                "model_name": model_name,
                "params": params,
                "tokens": tokens,
                "logprob": correct_choice_logprob,
            }
            logprob_data.append(item)

    logger.info(f"从 {model_name} 提取了 {len(logprob_data)} 条数据")
    if len(logprob_data) < MMLU_TOTAL_PROBLEMS:
        logger.warning(f"{model_name}评测数据不全，跳过")
        logprob_data = []
    return model_name, logprob_data


def load_all_data_to_dataframe(eval_results_dir: Path) -> pd.DataFrame:
    all_data_rows = []
    
    # 查找所有 JSON 文件
    json_files = list(eval_results_dir.glob("*.json"))
    logger.info(f"找到 {len(json_files)} 个结果文件")

    for file_path in json_files:
        try:
            model_name, logprob_data = extract_model_data_from_result(file_path)

            if logprob_data:
                all_data_rows.extend(logprob_data)
                logger.info(f"成功加载模型 {model_name}: {len(logprob_data)} 个问题")
            else:
                 logger.warning(f"模型 {model_name} 没有加载到任何数据")
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    df = pd.DataFrame(all_data_rows)
    logger.info(f"已创建总 DataFrame, 总行数: {len(df)}")
    logger.info(f"总任务数 (task_name): {df['task_name'].nunique()}")
    logger.info(f"总问题数 (problem_id): {df['problem_id'].nunique()}")
    logger.info(f"总模型数 (model_name): {df['model_name'].nunique()}")

    return df


def split_by_task(df: pd.DataFrame, train_ratio: float = 0.6,
                  val_ratio: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
    # 获取所有唯一的任务名
    all_tasks = sorted(list(df['task_name'].unique()))
    logger.info(f"总共找到 {len(all_tasks)} 个 MMLU 任务")

    # 随机打乱任务列表
    shuffled_tasks = all_tasks.copy()
    random.shuffle(shuffled_tasks)

    # 计算分割点
    n = len(shuffled_tasks)
    train_split = int(n * train_ratio)
    val_split = int(n * (train_ratio + val_ratio))

    # 分割
    train_tasks = shuffled_tasks[:train_split]
    val_tasks = shuffled_tasks[train_split:val_split]
    test_tasks = shuffled_tasks[val_split:]

    return train_tasks, val_tasks, test_tasks


def save_and_log(df: pd.DataFrame, columns: List[str], output_path: Path):
    """
    保存 DataFrame 到 CSV 并记录统计信息
    """
    if df.empty:
        logger.warning(f"数据集为空，跳过保存: {output_path}")
        return

    # 筛选最终列
    final_df = df[columns]

    # 数据统计
    logger.info(f"数据集统计 ({output_path.name}):")
    logger.info(f"  - 总行数: {len(final_df)}")
    logger.info(f"  - 问题数: {final_df['problem_id'].nunique()}")
    logger.info(f"  - 任务数: {final_df['task_name'].nunique()}")
    logger.info(f"  - 模型数: {final_df['model_name'].nunique()}")
    logger.info(f"  - Logprob 范围: [{final_df['logprob'].min():.4f}, {final_df['logprob'].max():.4f}]")

    # 保存数据
    final_df.to_csv(output_path, index=False)
    logger.info(f"数据已保存到 {output_path}")


def main():
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

    # 1. 加载所有模型的数据到 DataFrame
    logger.info("=" * 60)
    logger.info("开始加载所有模型的数据到 DataFrame...")
    all_data_df = load_all_data_to_dataframe(eval_results_dir)

    if all_data_df.empty:
        logger.error("没有找到有效的模型数据")
        return

    # 2. 按任务划分数据集
    logger.info("=" * 60)
    logger.info("按 60/20/20 比例划分 MMLU 任务...")

    train_tasks, val_tasks, test_tasks = split_by_task(
        all_data_df, train_ratio=0.6, val_ratio=0.2
    )

    logger.info(f"划分结果:")
    logger.info(f"  - 训练集: {len(train_tasks)} 个任务")
    logger.info(f"  - 验证集: {len(val_tasks)} 个任务")
    logger.info(f"  - 测试集: {len(test_tasks)} 个任务")

    # 4. 生成并保存数据集
    logger.info("=" * 60)
    
    # 定义最终输出的列
    output_columns = [
        'problem_id', 
        'task_name',
        'params', 
        'tokens', 
        'logprob',
    ]

    # 生成训练集
    logger.info("生成训练数据集...")
    train_df = all_data_df[all_data_df['task_name'].isin(train_tasks)]
    save_and_log(train_df, output_columns, train_output_path)

    # 生成验证集
    logger.info("-" * 40)
    logger.info("生成验证数据集...")
    val_df = all_data_df[all_data_df['task_name'].isin(val_tasks)]
    save_and_log(val_df, output_columns, val_output_path)

    # 生成测试集
    logger.info("-" * 40)
    logger.info("生成测试数据集...")
    test_df = all_data_df[all_data_df['task_name'].isin(test_tasks)]
    save_and_log(test_df, output_columns, test_output_path)

    logger.info("=" * 60)
    logger.info("数据集生成完成！")
    logger.info(f"所有数据集已保存到 {output_dir.resolve()} 目录")


if __name__ == "__main__":
    main()