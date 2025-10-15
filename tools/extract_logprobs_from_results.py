#!/usr/bin/env python3
"""
从原始评测结果文件中提取 logprobs 数据
处理 evaluation.py 生成的原始 JSON 文件
"""

import os
import json
import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 和 torch 类型的 JSON 编码器"""
    def default(self, obj):
        import numpy as np
        import torch

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif isinstance(obj, torch.dtype):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


def safe_json_dump(data, file_path):
    """安全地保存 JSON，处理各种类型转换"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def extract_logprobs_from_result(result_file):
    """从单个结果文件中提取 logprobs"""
    logger.info(f"处理文件: {result_file}")

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if "samples" not in results:
        logger.warning(f"文件 {result_file} 中没有 samples 数据")
        return None

    # 提取模型名称
    model_name = results.get("model_info", {}).get("name")
    if not model_name:
        # 从文件名推断模型名称
        model_name = Path(result_file).stem
        logger.info(f"从文件名推断模型名称: {model_name}")

    logprob_data = []

    # MMLU 任务的结果按子任务分组
    samples_dict = results.get("samples", {})
    for task_name, task_samples in samples_dict.items():
        for sample in task_samples:
            # 获取正确答案的索引
            correct_index = sample.get("gold", sample.get("target"))

            # 提取所有选项的 logprob 列表
            logprobs_list = [resp[0][0] for resp in sample.get("resps", []) if resp]

            # 根据正确答案的索引，提取对应的 logprob
            correct_choice_logprob = None
            # 使用 taskname_docid 格式避免不同子任务的 doc_id 重复
            combined_doc_id = f"{task_name}_{sample.get('doc_id')}"

            if correct_index is not None and logprobs_list and 0 <= correct_index < len(logprobs_list):
                correct_choice_logprob = logprobs_list[correct_index]
            else:
                logger.debug(f"无法为 doc_id={combined_doc_id} 找到正确选项的 logprob")

            choices = sample.get("doc", {}).get("choices", [])

            item = {
                "doc_id": combined_doc_id,
                "task_name": task_name,
                "question": sample.get("doc", {}).get("question", ""),
                "choices": choices,
                "gold_index": correct_index,
                "is_correct": sample.get("acc", False),
                "correct_choice_logprob": correct_choice_logprob,
                "all_logprobs": logprobs_list,
            }
            logprob_data.append(item)

    logger.info(f"提取了 {len(logprob_data)} 条 logprob 数据")
    return logprob_data, model_name


def process_directory(input_dir, output_dir=None):
    """处理目录下所有的原始结果文件"""
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = input_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有不是 logprobs 的 JSON 文件
    result_files = []
    for json_file in input_path.glob("*.json"):
        if "logprobs" not in json_file.name and "summary" not in json_file.name:
            result_files.append(json_file)

    if not result_files:
        logger.warning(f"在 {input_dir} 中没有找到原始结果文件")
        return

    logger.info(f"找到 {len(result_files)} 个结果文件待处理")

    for result_file in result_files:
        try:
            logprob_data, model_name = extract_logprobs_from_result(result_file)

            if logprob_data:
                output_file = output_dir / f"{model_name}_logprobs.json"
                safe_json_dump(logprob_data, output_file)
                logger.info(f"Logprobs 已保存到: {output_file}")
        except Exception as e:
            logger.error(f"处理文件 {result_file} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("所有文件处理完成")


def main():
    parser = argparse.ArgumentParser(
        description="从原始评测结果中提取 logprobs 数据"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="包含原始结果 JSON 文件的目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认与输入目录相同）"
    )

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
