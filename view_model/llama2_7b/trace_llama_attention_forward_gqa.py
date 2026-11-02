from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from trace_llama_attention_forward import make_causal_mask, run_phase


CONFIG_VALUES = {
    "vocab_size": 32000,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "attention_dropout": 0.0,
}
BATCH_SIZE = 1
PREFILL_LENGTH = 3


def main() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(**CONFIG_VALUES)
    config._attn_implementation = "eager"
    attention = LlamaAttention(config=config, layer_idx=0).eval()
    rotary_embedding = LlamaRotaryEmbedding(config)
    cache = DynamicCache(config=config)

    print("=== Llama 2 70B-dimension GQA config ===")
    for name, value in CONFIG_VALUES.items():
        print(f"{name}: {value}")
    print(f"num_key_value_groups: {attention.num_key_value_groups}")
    print("weights: randomly initialized; no checkpoint loaded")
    print("instantiated module: one LlamaAttention layer, not the full 80-layer model")
    print("query-head to KV-head mapping:")
    for kv_head in range(config.num_key_value_heads):
        first_query_head = kv_head * attention.num_key_value_groups
        last_query_head = first_query_head + attention.num_key_value_groups - 1
        print(f"  KV head {kv_head}: query heads {first_query_head}-{last_query_head}")

    prefill_position_ids = torch.arange(PREFILL_LENGTH).reshape(1, -1)
    prefill_hidden_states = torch.randn(
        (BATCH_SIZE, PREFILL_LENGTH, config.hidden_size),
        dtype=torch.float32,
    )
    prefill_mask = make_causal_mask(
        prefill_position_ids[0],
        torch.arange(PREFILL_LENGTH),
    )
    run_phase(
        "GQA prefill attention",
        attention,
        rotary_embedding,
        prefill_hidden_states,
        prefill_position_ids,
        prefill_mask,
        cache,
    )

    decode_position_ids = torch.tensor([[PREFILL_LENGTH]])
    decode_hidden_states = torch.randn(
        (BATCH_SIZE, 1, config.hidden_size),
        dtype=torch.float32,
    )
    decode_mask = make_causal_mask(
        decode_position_ids[0],
        torch.arange(PREFILL_LENGTH + 1),
    )
    run_phase(
        "GQA decode attention",
        attention,
        rotary_embedding,
        decode_hidden_states,
        decode_position_ids,
        decode_mask,
        cache,
    )


if __name__ == "__main__":
    main()
