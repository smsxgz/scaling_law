"""
初始程序模板，用于 logprobs scaling law 发现
这是一个简单的基准实现，用作进化的起点
"""

import numpy as np
from typing import Tuple


def scaling_law(X: np.ndarray) -> np.ndarray:
    """
    LogProbs Scaling Law 预测函数

    预测语言模型在给定参数量和训练tokens下的损失 (negative log probability)

    Args:
        X: 特征矩阵，形状为 (n_samples, 4)
           - X[:, 0]: params (模型参数量)
           - X[:, 1]: tokens (训练tokens数)

    Returns:
        预测的 logprobs 值，形状为 (n_samples,)
    """

    # 提取特征
    params = X[:, 0]
    tokens = X[:, 1]

    # 简单的 Chinchilla-风格 scaling law
    # logprobs = L_inf + A * params^(-alpha) + B * tokens^(-beta)

    # 初始参数（这些会被进化算法优化）
    L_inf = 2.0  # 基础损失
    A = 1e9      # 参数缩放系数
    B = 1e11     # tokens缩放系数
    alpha = 0.3  # 参数指数
    beta = 0.1   # tokens指数

    # 计算预测损失
    param_term = A * (params ** (-alpha))
    token_term = B * (tokens ** (-beta))

    logprobs_pred = L_inf + param_term + token_term

    # 确保预测值为负数（损失不能为负）
    logprobs_pred = np.minimum(logprobs_pred, 0)

    return logprobs_pred


def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    训练模型并进行预测

    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征

    Returns:
        测试集上的预测结果
    """
    # 这是一个简单的基准实现
    # 在实际的进化过程中，这个函数会被替换为更复杂的实现

    # 简单起见，我们直接使用 scaling_law 函数
    # 更复杂的实现可能会基于训练数据进行参数拟合

    return scaling_law(X_test)


# 这个函数是 SLD 框架期望的主要接口
def predict(X: np.ndarray) -> np.ndarray:
    """
    主预测函数，SLD 框架会调用这个函数

    Args:
        X: 特征矩阵

    Returns:
        预测结果
    """
    return scaling_law(X)


if __name__ == "__main__":
    # 测试初始程序
    print("测试初始 scaling law 程序...")

    # 创建一些测试数据
    test_params = np.array([14e6, 160e6, 1e9, 12e9])  # 不同规模的模型
    test_tokens = np.array([300e9, 300e9, 300e9, 300e9])  # 训练tokens

    X_test = np.column_stack([test_params, test_tokens])

    predictions = scaling_law(X_test)

    print("测试预测结果:")
    for i, (params, pred) in enumerate(zip(test_params, predictions)):
        print(f"模型大小: {params:.2e} 参数, 预测损失: {pred:.4f}")

    print("\n初始程序测试完成!")