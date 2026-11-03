dsv3.2 稀疏注意力选top2048
lighting indexer算子，NPU不亲和，耗时比较大。
-> PCP，切多份输入，弥补NPU上lighting indexer性能差的缺点。

## PCP
模型在 Prefill 阶段的大部分计算（如 RMSNorm, FFN, Embedding 等）都是
Token-Wise（逐 Token） 的，彼此独立，完全可以分开计算。只有 Attention 模块需要
Token 间的交互，而 PCP 通过特定的通信机制解决了这个问题。

## 不下载权重打印 Transformers 类

脚本 `deepseek_v32_class.py` 只读取 `deepseek-ai/DeepSeek-V3.2` 的配置，在 PyTorch `meta` 设备上构建 `DeepseekV32ForCausalLM`，然后打印 Transformers 模型结构。它不下载 checkpoint 权重，不需要 GPU，也不能执行推理。

```bash
# 首次运行：只下载配置
HF_ENDPOINT=https://hf-mirror.com python deepseek_v32_class.py

# 配置已缓存后：完全离线
HF_HUB_OFFLINE=1 python deepseek_v32_class.py
```

当前 Transformers 实现构建 61 个主干 decoder 层；官方配置还声明 1 个 MTP 层，但该类不把 `model.layers.61.*` 构建为模型模块。打印结果是 Transformers 类的结构，不是 checkpoint 原始张量布局。

