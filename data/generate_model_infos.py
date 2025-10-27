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


def extract_model_data_from_result(result_file: Path) -> dict:
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
    except KeyError as e:
        logger.warning(f"文件 {result_file} model_info 获取不正确，跳过")
        return

    return model_info


def main():
    # 查找所有 JSON 文件
    json_files = list(Path('./eval_results/0_shot').glob("*.json"))
    logger.info(f"找到 {len(json_files)} 个结果文件")

    model_infos: List[Dict] = []

    for file_path in json_files:
        try:
            model_info = extract_model_data_from_result(file_path)
            model_infos.append(model_info)
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    df = pd.DataFrame(model_infos)
    df.to_csv("model_info.csv", index=False)

    logger.info(f" params 范围: [{df['params'].min():.4e}, {df['params'].max():.4e}]")
    logger.info(f" tokens 范围: [{df['tokens'].min():.4e}, {df['tokens'].max():.4e}]")


if __name__ == "__main__":
    main()