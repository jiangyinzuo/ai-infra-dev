from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.nn.functional as functional
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding


MODEL_PATH = "hf-models/Llama-2-7b-hf"
BATCH_SIZE = 1
PREFILL_LENGTH = 4
INPUT_CYAN = "\033[36m"
OUTPUT_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def tensor_description(value: object) -> str:
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    return f"shape={tuple(value.shape)}, dtype={value.dtype}"


def format_input(description: str) -> str:
    return f"{INPUT_CYAN}input=({description}){ANSI_RESET}"


def format_output(description: str) -> str:
    return f"{OUTPUT_GREEN}output=({description}){ANSI_RESET}"


def print_operator(
    index: int,
    name: str,
    detail: str = "",
    depth: int = 0,
) -> None:
    suffix = f": {detail}" if detail else ""
    indentation = "  " * depth
    print(f"{indentation}[{index:02}] {name}{suffix}")


def make_causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return torch.where(allowed, 0.0, float("-inf")).reshape(
        BATCH_SIZE,
        1,
        query_positions.numel(),
        key_positions.numel(),
    )


def run_phase(
    phase_name: str,
    attention: LlamaAttention,
    rotary_embedding: LlamaRotaryEmbedding,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_embeddings = rotary_embedding(hidden_states, position_ids)
    operator_index = 0
    matmul_count = [0]
    handles: list[Any] = []

    print(f"=== {phase_name} ===")
    print(f"hidden_states: {tensor_description(hidden_states)}")
    print(f"position_ids: {tensor_description(position_ids)}")
    print(f"attention_mask: {tensor_description(attention_mask)}")
    print(
        f"{INPUT_CYAN}attention_mask[0, 0]: "
        f"{attention_mask[0, 0].tolist()}{ANSI_RESET}"
    )
    print(f"cache tokens before attention: {cache.get_seq_length(attention.layer_idx)}")
    print()
    print("=== Python-level operator trace ===")

    def trace(name: str, detail: str = "", depth: int = 0) -> None:
        nonlocal operator_index
        operator_index += 1
        print_operator(operator_index, name, detail, depth)

    def make_projection_hook(name: str) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            input_description = tensor_description(inputs[0]) if inputs else "<none>"
            trace(
                name,
                f"Linear, {format_input(input_description)}, "
                f"{format_output(tensor_description(output))}",
                depth=1,
            )
            if name != "o_proj" and isinstance(output, torch.Tensor):
                linear_shape = tuple(output.shape)
                reshaped_shape = (
                    *linear_shape[:-1],
                    linear_shape[-1] // attention.head_dim,
                    attention.head_dim,
                )
                transposed_shape = (
                    reshaped_shape[0],
                    reshaped_shape[2],
                    reshaped_shape[1],
                    reshaped_shape[3],
                )
                trace(
                    f"{name}: view + transpose",
                    format_output(f"shape={transposed_shape}"),
                    depth=1,
                )

        return hook

    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        handles.append(
            getattr(attention, name).register_forward_hook(make_projection_hook(name))
        )

    def trace_output_reshape(
        module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        input_description = tensor_description(inputs[0]) if inputs else "<none>"
        trace(
            "attention output reshape + contiguous",
            format_input(input_description),
            depth=1,
        )

    handles.append(
        attention.o_proj.register_forward_pre_hook(trace_output_reshape)
    )

    def traced_rotary_pos_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace(
            "apply_rotary_pos_emb (RoPE)",
            format_input(
                f"q={tensor_description(query)}, "
                f"k={tensor_description(key)}, "
                f"cos={tensor_description(cos)}"
            ),
            depth=1,
        )
        result = original_rotary_pos_emb(query, key, cos, sin, *args, **kwargs)
        trace(
            "RoPE output",
            format_output(
                f"q={tensor_description(result[0])}, "
                f"k={tensor_description(result[1])}"
            ),
            depth=2,
        )
        return result

    def traced_cache_update(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace(
            "KV cache update",
            format_input(
                f"new_k={tensor_description(key_states)}, "
                f"new_v={tensor_description(value_states)}, "
                f"layer_idx={layer_idx}"
            ),
            depth=1,
        )
        result = original_cache_update(
            key_states,
            value_states,
            layer_idx,
            *args,
            **kwargs,
        )
        trace(
            "KV cache output",
            format_output(
                f"key={tensor_description(result[0])}, "
                f"value={tensor_description(result[1])}, "
                f"cached_tokens={cache.get_seq_length(layer_idx)}"
            ),
            depth=2,
        )
        return result

    def traced_repeat_kv(
        hidden_states_to_repeat: torch.Tensor,
        n_rep: int,
    ) -> torch.Tensor:
        trace(
            "repeat_kv",
            format_input(
                f"hidden_states={tensor_description(hidden_states_to_repeat)}, "
                f"n_rep={n_rep}"
            ),
            depth=2,
        )
        result = original_repeat_kv(hidden_states_to_repeat, n_rep)
        trace(
            "repeat_kv output",
            format_output(tensor_description(result)),
            depth=3,
        )
        return result

    def traced_matmul(
        input_tensor: torch.Tensor,
        other_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        matmul_labels = ("Q @ K^T × scaling", "attention_weights @ V")
        label = (
            matmul_labels[matmul_count[0]]
            if matmul_count[0] < 2
            else "matmul"
        )
        matmul_count[0] += 1
        scaling = f", scaling={attention.scaling}" if matmul_count[0] == 1 else ""
        trace(
            label,
            format_input(
                f"left={tensor_description(input_tensor)}, "
                f"right={tensor_description(other_tensor)}{scaling}"
            ),
            depth=2,
        )
        result = original_matmul(input_tensor, other_tensor, *args, **kwargs)
        trace(
            f"{label} output",
            format_output(tensor_description(result)),
            depth=3,
        )
        return result

    def traced_softmax(*args: Any, **kwargs: Any) -> torch.Tensor:
        input_tensor = args[0] if args else kwargs["input"]
        trace(
            "attn_weights + attention_mask",
            format_input(
                f"weights={tensor_description(input_tensor)}, "
                f"mask={tensor_description(attention_mask)}"
            ),
            depth=2,
        )
        trace(
            "softmax",
            format_input(
                f"{tensor_description(input_tensor)}, "
                f"dim={kwargs.get('dim', args[1] if len(args) > 1 else None)}, "
                f"dtype={kwargs.get('dtype')}"
            ),
            depth=2,
        )
        result = original_softmax(*args, **kwargs)
        trace(
            "softmax output",
            format_output(tensor_description(result)),
            depth=3,
        )
        return result

    def traced_dropout(*args: Any, **kwargs: Any) -> torch.Tensor:
        input_tensor = args[0] if args else kwargs["input"]
        trace(
            "dropout",
            format_input(
                f"{tensor_description(input_tensor)}, "
                f"p={kwargs.get('p', args[1] if len(args) > 1 else 0.5)}, "
                f"training={kwargs.get('training', args[2] if len(args) > 2 else True)}"
            ),
            depth=2,
        )
        result = original_dropout(*args, **kwargs)
        trace(
            "dropout output",
            format_output(tensor_description(result)),
            depth=3,
        )
        return result

    def traced_eager_attention_forward(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        trace("eager_attention_forward", depth=1)
        result = original_eager_attention_forward(*args, **kwargs)
        trace(
            "attention output transpose + contiguous",
            format_output(
                f"{tensor_description(result[0])}, "
                f"weights={tensor_description(result[1])}"
            ),
            depth=2,
        )
        return result

    original_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
    original_cache_update = cache.update
    original_repeat_kv = modeling_llama.repeat_kv
    original_matmul = torch.matmul
    original_softmax = functional.softmax
    original_dropout = functional.dropout
    original_eager_attention_forward = modeling_llama.eager_attention_forward

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    modeling_llama,
                    "apply_rotary_pos_emb",
                    traced_rotary_pos_emb,
                )
            )
            stack.enter_context(
                patch.object(cache, "update", traced_cache_update)
            )
            stack.enter_context(
                patch.object(modeling_llama, "repeat_kv", traced_repeat_kv)
            )
            stack.enter_context(patch.object(torch, "matmul", traced_matmul))
            stack.enter_context(
                patch.object(functional, "softmax", traced_softmax)
            )
            stack.enter_context(
                patch.object(functional, "dropout", traced_dropout)
            )
            stack.enter_context(
                patch.object(
                    modeling_llama,
                    "eager_attention_forward",
                    traced_eager_attention_forward,
                )
            )
            trace("LlamaAttention.forward")
            with torch.inference_mode():
                output, attention_weights = attention(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                )
    finally:
        for handle in handles:
            handle.remove()

    print()
    print("=== returned values ===")
    print(f"{OUTPUT_GREEN}output: {tensor_description(output)}{ANSI_RESET}")
    print(
        f"{OUTPUT_GREEN}attention_weights: "
        f"{tensor_description(attention_weights)}{ANSI_RESET}"
    )
    print(f"cache tokens after attention: {cache.get_seq_length(attention.layer_idx)}")
    print()
    return output, attention_weights


def main() -> None:
    torch.manual_seed(0)

    config = AutoConfig.from_pretrained(MODEL_PATH)
    # 选择显式的 Python eager attention，而不是 SDPA 或 Flash Attention。
    config._attn_implementation = "eager"
    attention = LlamaAttention(config=config, layer_idx=0).eval()
    rotary_embedding = LlamaRotaryEmbedding(config)
    cache = DynamicCache(config=config)

    prefill_position_ids = torch.arange(PREFILL_LENGTH).reshape(1, -1)
    prefill_hidden_states = torch.randn(
        (BATCH_SIZE, PREFILL_LENGTH, config.hidden_size),
        dtype=torch.float32,
    )
    prefill_key_positions = torch.arange(PREFILL_LENGTH)
    prefill_mask = make_causal_mask(
        prefill_position_ids[0],
        prefill_key_positions,
    )

    run_phase(
        "prefill attention",
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
    decode_key_positions = torch.arange(PREFILL_LENGTH + 1)
    decode_mask = make_causal_mask(
        decode_position_ids[0],
        decode_key_positions,
    )

    run_phase(
        "decode attention",
        attention,
        rotary_embedding,
        decode_hidden_states,
        decode_position_ids,
        decode_mask,
        cache,
    )

    print("=== prefill vs decode ===")
    print(
        "prefill: query length=4, cache grows from 0 to 4, "
        "score shape=(1, 32, 4, 4)"
    )
    print(
        "decode:  query length=1, cache grows from 4 to 5, "
        "score shape=(1, 32, 1, 5)"
    )


if __name__ == "__main__":
    main()
