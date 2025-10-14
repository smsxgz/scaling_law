"""
本地数据加载器，用于加载 logprobs scaling law 数据集
适配 SLD 框架的数据加载接口
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# 本地数据配置
LOCAL_TASK_SCHEMA = {
    "logprobs_scaling_law": {
        "file_path": "logprobs_scaling_dataset.csv",
        "feature_names": ["params", "tokens", "log_params", "log_tokens"],
        "target_name": "loss",
        "group_column": "group"
    }
}


def load_local_data(
    app_name: str,
    train: bool = True,
    data_dir: str = "."
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """
    加载本地 CSV 数据，适配 SLD 框架的数据格式

    Args:
        app_name: 任务名称 (例如 'logprobs_scaling_law')
        train: 是否加载训练数据 (True) 或测试数据 (False)
        data_dir: 数据文件所在目录

    Returns:
        字典，键为组名，值为 (features, targets) 元组
    """
    if app_name not in LOCAL_TASK_SCHEMA:
        raise ValueError(f"任务 '{app_name}' 未在本地任务配置中找到。可用任务: {list(LOCAL_TASK_SCHEMA.keys())}")

    schema = LOCAL_TASK_SCHEMA[app_name]
    file_path = Path(data_dir) / schema["file_path"]

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件 {file_path} 不存在")

    # 读取 CSV 数据
    df = pd.read_csv(file_path)

    feature_names = schema["feature_names"]
    target_name = schema["target_name"]
    group_column = schema["group_column"]

    if group_column not in df.columns:
        raise ValueError(f"数据中缺少分组列 '{group_column}'")

    # 检查必要的列是否存在
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"数据中缺少特征列: {missing_features}")

    if target_name not in df.columns:
        raise ValueError(f"数据中缺少目标列 '{target_name}'")

    processed_data = {}

    # 获取所有唯一的组
    unique_groups = sorted(df[group_column].unique())

    for group_key in unique_groups:
        # 过滤当前组的数据
        group_data = df[df[group_column] == group_key].copy()

        # 提取特征
        X = group_data[feature_names].values.astype(float)

        # 提取目标
        y = group_data[target_name].values.astype(float)

        # 简单的 train/test 分割 (70%/30%)
        n_samples = len(group_data)
        n_train = int(0.7 * n_samples)

        if train:
            X_split = X[:n_train]
            y_split = y[:n_train]
        else:
            X_split = X[n_train:]
            y_split = y[n_train:]

        # 如果测试集为空，则跳过该组
        if len(X_split) == 0:
            continue

        processed_data[group_key] = (X_split, y_split)

    return processed_data


def get_data_info(app_name: str, data_dir: str = ".") -> Dict[str, Any]:
    """
    获取数据集的基本信息

    Returns:
        包含数据集信息的字典
    """
    if app_name not in LOCAL_TASK_SCHEMA:
        raise ValueError(f"任务 '{app_name}' 未在本地任务配置中找到")

    schema = LOCAL_TASK_SCHEMA[app_name]
    file_path = Path(data_dir) / schema["file_path"]

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件 {file_path} 不存在")

    df = pd.read_csv(file_path)

    info = {
        "feature_names": schema["feature_names"],
        "target_name": schema["target_name"],
        "num_groups": df[schema["group_column"]].nunique(),
        "total_samples": len(df),
        "groups": sorted(df[schema["group_column"]].unique()),
        "feature_stats": df[schema["feature_names"]].describe().to_dict(),
        "target_stats": df[schema["target_name"]].describe().to_dict()
    }

    return info


if __name__ == "__main__":
    # 测试数据加载器
    app_name = "logprobs_scaling_law"

    try:
        # 获取数据信息
        info = get_data_info(app_name)
        print(f"数据集信息:")
        print(f"特征: {info['feature_names']}")
        print(f"目标: {info['target_name']}")
        print(f"组数: {info['num_groups']}")
        print(f"总样本数: {info['total_samples']}")

        # 加载训练数据
        train_data = load_local_data(app_name, train=True)
        print(f"\n训练数据:")
        print(f"组数: {len(train_data)}")

        # 显示第一个组的数据
        first_group = next(iter(train_data))
        X_train, y_train = train_data[first_group]
        print(f"第一组 '{first_group}': X shape={X_train.shape}, y shape={y_train.shape}")

        # 加载测试数据
        test_data = load_local_data(app_name, train=False)
        print(f"\n测试数据:")
        print(f"组数: {len(test_data)}")

        if test_data:
            first_test_group = next(iter(test_data))
            X_test, y_test = test_data[first_test_group]
            print(f"第一组 '{first_test_group}': X shape={X_test.shape}, y shape={y_test.shape}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()