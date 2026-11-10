from __future__ import annotations

import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

MODEL_ID = "deepseek-ai/DeepSeek-V3"
SEED = 0
BATCH_SIZE = 1
SEQUENCE_LENGTH = 4


TRACED_FUNCTIONS = {
    "linear",
    "sigmoid",
    "topk",
}


def tensor_shapes(value: object) -> list[tuple[int, ...]]:
    if isinstance(value, torch.Tensor):
        return [tuple(value.shape)]
    if isinstance(value, (tuple, list)):
        return [shape for item in value for shape in tensor_shapes(item)]
    if isinstance(value, dict):
        return [shape for item in value.values() for shape in tensor_shapes(item)]
    return []


class RouterOpTrace(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        function_name = getattr(func, "__name__", str(func))
        traced = function_name in TRACED_FUNCTIONS
        input_shapes = tensor_shapes((args, kwargs)) if traced else []
        output = func(*args, **kwargs)
        if traced:
            print(
                f"{function_name}: input_shapes={input_shapes}, "
                f"output_shapes={tensor_shapes(output)}"
            )
        return output


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel():,}"
    )


def main() -> None:
    torch.manual_seed(SEED)
    config = AutoConfig.from_pretrained(MODEL_ID)
    router = DeepseekV3TopkRouter(config)
    nn.init.normal_(router.weight, mean=0.0, std=0.02)

    hidden_states = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, config.hidden_size)
    print("\n=== PyTorch Python API trace ===")
    with RouterOpTrace():
        router_logits, topk_weights, topk_indices = router(hidden_states)

    print(f"model config: {MODEL_ID}")
    print(
        "router config: "
        f"hidden_size={router.hidden_dim}, experts={router.num_experts}, "
        f"top_k={router.top_k}, groups={router.num_group}, topk_group={router.topk_group}, "
        f"norm_topk_prob={router.norm_topk_prob}, "
        f"routed_scaling_factor={router.routed_scaling_factor}"
    )
    print("\n=== Random router tensors ===")
    print_tensor("router.weight", router.weight)
    print(f"router.weight first 8 values: {router.weight.flatten()[:8].tolist()}")
    print_tensor("router.e_score_correction_bias", router.e_score_correction_bias)
    print(
        f"input hidden_states: shape={tuple(hidden_states.shape)} "
        f"(batch={hidden_states.shape[0]}, sequence_length={hidden_states.shape[1]}, "
        f"hidden_size={hidden_states.shape[2]}), dtype={hidden_states.dtype}, "
        f"numel={hidden_states.numel():,}"
    )

    print("\n=== Forward activations ===")
    print_tensor("router_logits", router_logits)
    print_tensor("topk_weights", topk_weights)
    print_tensor("topk_indices", topk_indices)
    print(f"topk_indices: {topk_indices.tolist()}")
    print(f"topk_weights: {topk_weights.tolist()}")


if __name__ == "__main__":
    main()
