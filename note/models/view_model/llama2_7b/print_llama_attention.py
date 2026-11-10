from __future__ import annotations

from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention


MODEL_PATH = "hf-models/Llama-2-7b-hf"


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    attention = LlamaAttention(config=config, layer_idx=0)

    print("=== LlamaAttention ===")
    print(attention)
    print()
    print("=== runtime fields ===")
    for name in (
        "layer_idx",
        "head_dim",
        "num_key_value_groups",
        "scaling",
        "attention_dropout",
        "is_causal",
    ):
        print(f"{name}: {getattr(attention, name)!r}")

    print("\n=== parameters ===")
    parameter_numel = 0
    parameter_tensors = 0
    for name, parameter in attention.named_parameters():
        parameter_tensors += 1
        parameter_numel += parameter.numel()
        print(
            f"{name}: shape={tuple(parameter.shape)}, "
            f"dtype={parameter.dtype}, "
            f"requires_grad={parameter.requires_grad}, "
            f"numel={parameter.numel()}"
        )
    print(f"parameter tensors: {parameter_tensors}")
    print(f"parameter numel: {parameter_numel}")


if __name__ == "__main__":
    main()
