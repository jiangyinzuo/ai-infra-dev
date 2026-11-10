from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008
NUM_ATTENTION_HEADS = 32
VOCAB_SIZE = 32000


def main() -> None:
    # 这里只创建 Llama 的模块，不加载 7B 权重；重点是观察 Linear 的真实调用。
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=32,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_ATTENTION_HEADS,
        vocab_size=VOCAB_SIZE,
        attention_bias=False,
        mlp_bias=False,
    )
    attention = LlamaAttention(config, layer_idx=0)
    mlp = LlamaMLP(config)
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    hidden_states = torch.randn(1, 7, HIDDEN_SIZE)
    print(f"hidden_states: shape={tuple(hidden_states.shape)}")

    print("\n=== LlamaAttention.forward 中的 Linear 调用 ===")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)

    # transformers/.../modeling_llama.py 的 LlamaAttention.forward 原代码：
    # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    print(f"q_proj: input={tuple(hidden_states.shape)} -> output={tuple(query_states.shape)}")
    print(f"k_proj: input={tuple(hidden_states.shape)} -> output={tuple(key_states.shape)}")
    print(f"v_proj: input={tuple(hidden_states.shape)} -> output={tuple(value_states.shape)}")

    # 源码在 attention 计算后先 reshape，再调用：
    # attn_output = self.o_proj(attn_output)
    attention_output = torch.randn_like(hidden_states)
    output = attention.o_proj(attention_output)
    print(f"o_proj: input={tuple(attention_output.shape)} -> output={tuple(output.shape)}")

    print("\n=== LlamaMLP.forward 中的 Linear 调用 ===")
    # transformers/.../modeling_llama.py 的 LlamaMLP.forward 原代码：
    # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    gate_projection = mlp.gate_proj(hidden_states)
    up_projection = mlp.up_proj(hidden_states)
    activated_gate = mlp.act_fn(gate_projection)
    mlp_output = mlp.down_proj(activated_gate * up_projection)
    print(f"gate_proj: {tuple(hidden_states.shape)} -> {tuple(gate_projection.shape)}")
    print(f"up_proj:   {tuple(hidden_states.shape)} -> {tuple(up_projection.shape)}")
    print(f"down_proj: {tuple(gate_projection.shape)} -> {tuple(mlp_output.shape)}")

    print("\n=== LlamaForCausalLM.forward 中的 lm_head 调用 ===")
    # transformers/.../modeling_llama.py 的 LlamaForCausalLM.forward 原代码：
    # logits = self.lm_head(hidden_states[:, slice_indices, :])
    slice_indices = slice(-1, None)
    logits = lm_head(hidden_states[:, slice_indices, :])
    print(f"lm_head: input={tuple(hidden_states[:, slice_indices, :].shape)} -> output={tuple(logits.shape)}")


if __name__ == "__main__":
    main()