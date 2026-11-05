from __future__ import annotations

import torch
from torch import nn
from transformers import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Indexer,
    DeepseekV32RotaryEmbedding,
)


CONFIG_SOURCE = "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json"
SEED = 0
BATCH_SIZE = 1
SEQUENCE_LENGTH = 8
DTYPE = torch.bfloat16


def print_comment(message: str) -> None:
    print(f"\033[33m{message}\033[0m")


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel():,}")


def main() -> None:
    torch.manual_seed(SEED)
    config = DeepseekV32Config(
        hidden_size=7168,
        q_lora_rank=1536,
        index_n_heads=64,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=2048,
        initializer_range=0.02,
        max_position_embeddings=163840,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    indexer = DeepseekV32Indexer(config, layer_idx=0).to(dtype=DTYPE).eval()
    for module in indexer.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
    rotary_emb = DeepseekV32RotaryEmbedding(config).eval()

    hidden_states = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, config.hidden_size, dtype=DTYPE)
    q_resid = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, config.q_lora_rank, dtype=DTYPE)
    position_ids = torch.arange(SEQUENCE_LENGTH).unsqueeze(0).expand(BATCH_SIZE, -1)
    attention_mask = torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool).tril().unsqueeze(0)

    with torch.inference_mode():
        position_embeddings = rotary_emb(hidden_states, position_ids)
        topk_indices = indexer(
            hidden_states=hidden_states,
            q_resid=q_resid,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    print(f"config source (reference only, no download): {CONFIG_SOURCE}")
    print(f"seed: {SEED}")
    print("\n=== Indexer ===")
    print(indexer)
    for name, parameter in indexer.named_parameters():
        print_tensor(name, parameter)
    print(f"parameter numel: {sum(parameter.numel() for parameter in indexer.parameters()):,}")
    print_comment("仅构建单层 Indexer；Linear 权重随机正态初始化，LayerNorm 保留 weight=1、bias=0。")
    print_comment("使用 BF16 浮点实现，不加载官方 FP8 checkpoint。")

    print("\n=== Random inputs ===")
    print_tensor("hidden_states", hidden_states)
    print_tensor("q_resid", q_resid)
    print_comment("这里 q_resid 是独立随机输入；完整模型中它来自 q_a_layernorm(q_a_proj(hidden_states))。")
    print_tensor("position_ids", position_ids)
    print_tensor("cos", position_embeddings[0])
    print_tensor("sin", position_embeddings[1])
    print_tensor("attention_mask (True=allowed)", attention_mask)

    print("\n=== Forward output ===")
    print_tensor("topk_indices", topk_indices)
    print(topk_indices)
    print_comment(
        f"index_topk={config.index_topk}，本例只有 {SEQUENCE_LENGTH} 个 token，"
        "返回数量为 min(index_topk, sequence_length)。"
        "前面的 query 可见 token 更少，topk 会包含分数为 -inf 的未来位置，不能把所有返回项当作有效关注位置。"
    )


if __name__ == "__main__":
    main()