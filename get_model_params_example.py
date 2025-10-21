#!/usr/bin/env python3
"""
演示如何从 lm_eval HFLM 类和 transformers 库获取模型参数量

主要方法：
1. 使用 transformers PreTrainedModel.num_parameters() 方法
2. 使用 PyTorch model.parameters() 手动计算
3. 使用 model.named_parameters() 查看各层参数分布
"""

import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoConfig


def demo_get_params_from_hflm():
    """演示从 HFLM 对象获取模型参数量"""
    print("=" * 80)
    print("方法1: 从 HFLM 对象获取参数量")
    print("=" * 80)

    # 加载一个小模型作为示例
    model_name = "EleutherAI/pythia-14m"

    print(f"\n正在加载模型: {model_name}")
    lm = HFLM(
        pretrained=model_name,
        device="cpu",  # 使用 CPU 避免显存问题
    )

    # 方法1: 使用 transformers PreTrainedModel 的 num_parameters() 方法
    # HFLM 的 model 属性返回底层的 transformers 模型
    total_params = lm.model.num_parameters()
    trainable_params = lm.model.num_parameters(only_trainable=True)

    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"总参数量 (科学计数法): {total_params:.2e}")

    # 方法2: 使用 PyTorch 手动计算
    pytorch_total = sum(p.numel() for p in lm.model.parameters())
    pytorch_trainable = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)

    print(f"\nPyTorch 手动计算:")
    print(f"总参数量: {pytorch_total:,}")
    print(f"可训练参数量: {pytorch_trainable:,}")

    # 验证两种方法结果一致
    assert total_params == pytorch_total, "两种方法计算结果不一致！"
    print("\n✓ 验证通过: transformers.num_parameters() 与 PyTorch 手动计算结果一致")

    return lm


def demo_parameter_breakdown(lm):
    """演示查看模型各层参数分布"""
    print("\n" + "=" * 80)
    print("方法2: 查看模型各层参数分布")
    print("=" * 80)

    print("\n模型结构和参数量分布:")
    print(f"{'层名称':<60} {'参数量':>15} {'形状'}")
    print("-" * 95)

    total = 0
    for name, param in lm.model.named_parameters():
        param_count = param.numel()
        total += param_count
        print(f"{name:<60} {param_count:>15,} {str(tuple(param.shape))}")

    print("-" * 95)
    print(f"{'总计':<60} {total:>15,}")

    return total


def demo_config_based_params():
    """演示从 config 获取参数量信息"""
    print("\n" + "=" * 80)
    print("方法3: 从 AutoConfig 获取参数相关信息")
    print("=" * 80)

    model_name = "EleutherAI/pythia-14m"
    config = AutoConfig.from_pretrained(model_name)

    print(f"\n模型配置信息:")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"intermediate_size: {getattr(config, 'intermediate_size', 'N/A')}")

    # 注意: 大多数 config 不直接包含 num_parameters 属性
    # 需要加载模型后才能准确获取
    if hasattr(config, 'num_parameters'):
        print(f"num_parameters (from config): {config.num_parameters}")
    else:
        print("num_parameters: 不在 config 中，需要加载模型后计算")


def demo_direct_transformers_model():
    """演示直接使用 transformers 模型获取参数量"""
    print("\n" + "=" * 80)
    print("方法4: 直接使用 transformers AutoModel")
    print("=" * 80)

    model_name = "EleutherAI/pythia-14m"

    print(f"\n加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 最简单直接的方法
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    non_trainable_params = total_params - trainable_params

    print(f"\n参数统计:")
    print(f"  总参数量:     {total_params:>15,} ({total_params:.2e})")
    print(f"  可训练参数:   {trainable_params:>15,} ({trainable_params:.2e})")
    print(f"  不可训练参数: {non_trainable_params:>15,} ({non_trainable_params:.2e})")

    # 计算模型大小 (以 MB 为单位)
    # 假设使用 float32 (4 bytes per parameter)
    param_size_mb = (total_params * 4) / (1024 ** 2)
    print(f"\n估计模型大小 (float32): {param_size_mb:.2f} MB")

    return model


def demo_exclude_embeddings():
    """演示排除 embedding 层后的参数量"""
    print("\n" + "=" * 80)
    print("方法5: 获取不包含 embedding 的参数量")
    print("=" * 80)

    model_name = "EleutherAI/pythia-14m"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # transformers 4.x 版本支持 exclude_embeddings 参数
    try:
        non_embedding_params = model.num_parameters(exclude_embeddings=True)
        total_params = model.num_parameters()
        embedding_params = total_params - non_embedding_params

        print(f"\n参数分布:")
        print(f"  总参数量:             {total_params:>15,}")
        print(f"  非 embedding 参数:    {non_embedding_params:>15,}")
        print(f"  embedding 参数:       {embedding_params:>15,}")
        print(f"  embedding 占比:       {(embedding_params/total_params)*100:.2f}%")
    except TypeError:
        print("\n当前 transformers 版本不支持 exclude_embeddings 参数")
        print("手动计算 embedding 参数量:")

        embedding_params = 0
        for name, param in model.named_parameters():
            if 'embed' in name.lower():
                embedding_params += param.numel()

        total_params = model.num_parameters()
        print(f"  总参数量:             {total_params:>15,}")
        print(f"  embedding 参数 (估计): {embedding_params:>15,}")
        print(f"  embedding 占比:       {(embedding_params/total_params)*100:.2f}%")


def main():
    """主函数：运行所有演示"""
    print("\n" + "=" * 80)
    print("lm_eval HFLM 和 transformers 获取模型参数量完整演示")
    print("=" * 80)

    # 演示1: 从 HFLM 获取参数量
    lm = demo_get_params_from_hflm()

    # 演示2: 查看参数分布
    demo_parameter_breakdown(lm)

    # 演示3: 从 config 获取信息
    demo_config_based_params()

    # 演示4: 直接使用 transformers
    demo_direct_transformers_model()

    # 演示5: 排除 embeddings
    demo_exclude_embeddings()

    print("\n" + "=" * 80)
    print("核心要点总结")
    print("=" * 80)
    print("""
1. HFLM 模型对象访问:
   - lm.model 或 lm._model 获取底层 transformers 模型

2. transformers PreTrainedModel 方法:
   - model.num_parameters()                    # 总参数量
   - model.num_parameters(only_trainable=True) # 仅可训练参数
   - model.num_parameters(exclude_embeddings=True) # 排除 embedding (某些版本)

3. PyTorch 原生方法:
   - sum(p.numel() for p in model.parameters())  # 总参数量
   - sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数

4. 详细分析:
   - model.named_parameters() 遍历各层查看参数分布
   - param.numel() 获取单个参数张量的元素数量
   - param.shape 获取参数形状

5. 模型配置:
   - AutoConfig.from_pretrained() 获取配置信息
   - config 通常不包含 num_parameters，需加载模型后计算
""")


if __name__ == "__main__":
    main()
