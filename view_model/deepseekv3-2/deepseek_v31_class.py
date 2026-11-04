from __future__ import annotations

from inspect import getfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_ID = "deepseek-ai/DeepSeek-V3.1"


def print_comment(message: str) -> None:
    print(f"\033[33m{message}\033[0m")


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=config.dtype)

    print(f"model_id: {MODEL_ID}")
    print(f"class: {type(model)}")
    print(f"source: {getfile(type(model))}")
    print("\n=== print(model) ===")
    print(model)
    print("\n=== Meta model state ===")
    print(f"decoder layers: {len(model.model.layers)}")
    print(f"parameter numel: {sum(p.numel() for p in model.parameters()):,}")
    print(f"parameter dtypes: {sorted({str(p.dtype) for p in model.parameters()})}")
    print(f"all parameters meta: {all(p.is_meta for p in model.parameters())}")
    print(f"all buffers meta: {all(b.is_meta for b in model.buffers())}")
    print_comment("仅根据官方配置构建模块；没有真实权重数值，不能直接推理。")
    print_comment(
        "当前 Transformers 实现不构建 MTP 层，并合并专家投影；"
        "这里的 BF16 参数不代表 checkpoint 的 FP8 存储布局和量化 scale。"
    )


if __name__ == "__main__":
    main()
