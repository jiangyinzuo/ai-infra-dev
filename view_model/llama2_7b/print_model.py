import torch
from transformers import AutoModelForCausalLM

model_path = "hf-models/Llama-2-7b-hf"


def format_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.3f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.3f}M"
    if count >= 1_000:
        return f"{count / 1_000:.3f}K"
    return str(count)


def print_module_counts(name: str, module: torch.nn.Module) -> None:
    direct_numel = sum(
        parameter.numel() for parameter in module.parameters(recurse=False)
    )
    recursive_numel = sum(parameter.numel() for parameter in module.parameters())
    print(
        f"{name}: type={type(module).__name__}, "
        f"direct_numel={direct_numel}, "
        f"recursive_numel={recursive_numel} "
        f"({format_count(recursive_numel)})"
    )


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
model = model.cuda().eval()

print("=== print(model) ===")
print(model)

named_parameters = list(model.named_parameters())
named_buffers = list(model.named_buffers())
parameter_numel = sum(parameter.numel() for _, parameter in named_parameters)
trainable_numel = sum(
    parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
)
parameter_logical_bytes = sum(
    parameter.numel() * parameter.element_size() for _, parameter in named_parameters
)
parameter_storage_bytes = sum(
    parameter.untyped_storage().nbytes() for _, parameter in named_parameters
)
buffer_numel = sum(buffer.numel() for _, buffer in named_buffers)
buffer_logical_bytes = sum(
    buffer.numel() * buffer.element_size() for _, buffer in named_buffers
)

print("\n=== 已加载对象的实际状态 ===")
print(f"model.training: {model.training}")
print(f"parameter tensors: {len(named_parameters)}")
print(f"parameter numel: {parameter_numel} ({format_count(parameter_numel)})")
print(f"trainable numel: {trainable_numel}")
print(f"parameter logical bytes: {parameter_logical_bytes}")
print(f"parameter storage bytes: {parameter_storage_bytes}")
print(f"buffer tensors: {len(named_buffers)}")
print(f"buffer numel: {buffer_numel}")
print(f"buffer logical bytes: {buffer_logical_bytes}")

print("\n=== model.named_parameters()（保留 embedding、layer 0 和输出层）===")
for name, parameter in named_parameters:
    if name.startswith("model.layers.") and not name.startswith("model.layers.0."):
        continue
    print(
        f"{name}: "
        f"shape={tuple(parameter.shape)}, "
        f"dtype={parameter.dtype}, "
    )

print("\n=== model.named_buffers() ===")
for name, buffer in named_buffers:
    print(f"{name}: shape={tuple(buffer.shape)}, dtype={buffer.dtype}, ")

backbone = model.model
first_layer = backbone.layers[0]
attention = first_layer.self_attn

print("\n=== 模块参数归属 ===")
for name, module in (
    ("model.model", backbone),
    ("model.model.embed_tokens", backbone.embed_tokens),
    ("model.model.layers", backbone.layers),
    ("model.model.layers[0]", first_layer),
    ("model.model.layers[0].self_attn", attention),
    ("model.model.layers[0].mlp", first_layer.mlp),
    ("model.model.norm", backbone.norm),
    ("model.lm_head", model.lm_head),
):
    print_module_counts(name, module)

print("\n=== 第 0 层 attention 的运行时字段 ===")
for name in (
    "num_heads",
    "num_key_value_heads",
    "num_key_value_groups",
    "head_dim",
    "is_causal",
):
    if name in vars(attention):
        print(f"self_attn.{name}: {vars(attention)[name]!r}")
print(f"model.model.rotary_emb: {backbone.rotary_emb!r}")

embedding_weight = backbone.embed_tokens.weight
lm_head_weight = model.lm_head.weight
print("\n=== embedding 与 lm_head 的实际存储关系 ===")
print(f"embed_tokens.weight shape: {tuple(embedding_weight.shape)}")
print(f"lm_head.weight shape: {tuple(lm_head_weight.shape)}")
print(
    "embed_tokens.weight storage pointer: "
    f"{embedding_weight.untyped_storage().data_ptr()}"
)
print(f"lm_head.weight storage pointer: {lm_head_weight.untyped_storage().data_ptr()}")
print(
    "same storage: "
    f"{embedding_weight.untyped_storage().data_ptr() == lm_head_weight.untyped_storage().data_ptr()}"
)
