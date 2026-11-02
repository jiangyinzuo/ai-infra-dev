from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from print_llama_attention import print_attention_details


CONFIG_VALUES = {
    "vocab_size": 32000,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "initializer_range": 0.02,
}


def initialize_attention(attention: LlamaAttention, std: float) -> None:
    with torch.no_grad():
        for module in attention.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


def main() -> None:
    config = LlamaConfig(**CONFIG_VALUES)
    attention = LlamaAttention(config=config, layer_idx=0)
    initialize_attention(attention, config.initializer_range)

    print("=== Llama 2 70B-like GQA config ===")
    for name, value in CONFIG_VALUES.items():
        print(f"{name}: {value}")
    print(f"head_dim: {attention.head_dim}")
    print(f"num_key_value_groups: {attention.num_key_value_groups}")
    print("weights: newly random-initialized; no checkpoint loaded")
    print("initialization: normal(mean=0, std=initializer_range)")
    print()
    print_attention_details(attention, "Randomly initialized Llama GQA attention")


if __name__ == "__main__":
    main()
