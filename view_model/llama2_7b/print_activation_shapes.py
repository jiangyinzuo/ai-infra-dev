from __future__ import annotations

import argparse
import inspect
import sys
import weakref
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


MODEL_PATH = "hf-models/Llama-2-7b-hf"


DIMENSION_NAMES = {
    "batch": "batch size",
    "sequence": "token count",
    "hidden": "hidden size",
    "q_heads": "query heads",
    "kv_heads": "key/value heads",
    "head_dim": "per-head size",
    "vocab": "vocabulary scores",
}

BATCH_SEQUENCE = ("batch", "sequence")
COMMENT_YELLOW = "\033[33m"
VARIABLE_BLUE = "\033[34m"
LAYER_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def print_comment(message: str) -> None:
    print(f"{COMMENT_YELLOW}# {message}{ANSI_RESET}")


def format_variable_name(name: str) -> str:
    return f"{VARIABLE_BLUE}{name}{ANSI_RESET}"


def format_layer_name(name: str) -> str:
    return f"{LAYER_GREEN}[{name}]{ANSI_RESET}"


def format_input_path(path: str) -> str:
    parameter_name, separator, suffix = path.partition("[")
    return (
        f"{format_variable_name(parameter_name)}{separator}{suffix}"
        if separator
        else format_variable_name(parameter_name)
    )


    if source_name.startswith("in@"):
        return source_name
    if source_name.startswith(("@", "op@")):
        return f"{source_name}.{path}"
    return f"{source_name}.input[{format_input_path(path)}]"


def format_runtime_local_reference(reference: str) -> str:
    function_name, separator, variable_name = reference.rpartition(".")
    if not separator:
        return reference
    return f"{function_name}.{format_variable_name(variable_name)}"

HIDDEN_SHAPE = ("batch", "sequence", "hidden")
POSITION_EMBEDDING_SHAPE = ("batch", "sequence", "head_dim")
LOGITS_SHAPE = ("batch", "sequence", "vocab")
TRACE_TENSOR_OPERATION_PREFIXES = (
    "aten.add",
    "aten.sub",
    "aten.mul",
    "aten.div",
    "aten.matmul",
    "aten.mm",
    "aten.bmm",
    "aten.addmm",
    "aten.addmv",
    "aten.mv",
    "aten.baddbmm",
    "aten.linear",
    "aten.einsum",
    "aten._scaled_dot_product_",
    "aten.unsqueeze",
    "aten.chunk",
    "aten.cat",
    "aten.neg",
)


def parse_layers(spec: str, layer_count: int) -> tuple[int, ...]:
    if spec.strip().lower() == "all":
        return tuple(range(layer_count))

    layers = tuple(sorted({int(item.strip()) for item in spec.split(",")}))
    invalid_layers = tuple(
        layer for layer in layers if layer < 0 or layer >= layer_count
    )
    if invalid_layers:
        raise ValueError(
            f"invalid layer index: {invalid_layers}; "
            f"valid range is 0..{layer_count - 1}"
        )
    return layers


def describe(value: object, sample_values: int) -> str:
    if isinstance(value, torch.Tensor):
        description = (
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )
        if sample_values > 0:
            sample = (
                value.detach()
                .flatten()[:sample_values]
                .float()
                .cpu()
                .tolist()
            )
            description += f", first_values={sample}"
        return description

    if isinstance(value, dict):
        entries = ", ".join(
            f"{key}={describe(item, sample_values)}"
            for key, item in value.items()
        )
        return "{" + entries + "}"

    if isinstance(value, tuple):
        return "(" + ", ".join(
            describe(item, sample_values) for item in value
        ) + ")"

    if isinstance(value, list):
        return "[" + ", ".join(
            describe(item, sample_values) for item in value
        ) + "]"

    if value is None:
        return "None"

    return type(value).__name__


def get_positional_parameter_names(
    module: torch.nn.Module,
) -> tuple[str, ...]:
    try:
        parameters = tuple(inspect.signature(module.forward).parameters.values())
    except (TypeError, ValueError):
        return ()
    return tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    )


def describe_named_inputs(
    module: torch.nn.Module,
    inputs: tuple[object, ...],
    sample_values: int,
) -> str:
    positional_parameter_names = get_positional_parameter_names(module)
    entries: list[str] = []
    for index, value in enumerate(inputs):
        if index < len(positional_parameter_names):
            parameter_name = positional_parameter_names[index]
        else:
            parameter_name = f"arg{index}"
        entries.append(
            f"{format_variable_name(parameter_name)}="
            f"{describe(value, sample_values)}"
        )
    return "(" + ", ".join(entries) + ")"


def describe_named_kwargs(
    keyword_inputs: dict[str, object],
    sample_values: int,
) -> str:
    entries = ", ".join(
        f"{format_variable_name(key)}={describe(value, sample_values)}"
        for key, value in keyword_inputs.items()
    )
    return "{" + entries + "}"


class RuntimeFlowTracer:
    def __init__(self) -> None:
        self._next_output_number = 1
        self._producers: dict[int, list[tuple[str, str, object]]] = {}
        self._next_operation_number = 1
        self._operation_producers: dict[
            int,
            list[tuple[str, str, weakref.ReferenceType[object]]],
        ] = {}
        self._active_input_sources: dict[
            int,
            list[tuple[str, str, object]],
        ] = {}
        self._next_external_input_number = 1
        self._external_inputs: dict[
            int,
            tuple[str, weakref.ReferenceType[object]],
        ] = {}
        self._input_scopes: dict[
            str,
            list[list[tuple[int, tuple[str, str, object]]]],
        ] = {}


    def _record_external_input(self, value: torch.Tensor) -> str:
        value_id = id(value)
        existing = self._external_inputs.get(value_id)
        if existing is not None and existing[1]() is value:
            return existing[0]

        input_label = f"in@{self._next_external_input_number}"
        self._next_external_input_number += 1
        self._external_inputs[value_id] = (input_label, weakref.ref(value))
        return input_label

    @staticmethod
    def _is_tracer_frame(frame: object) -> bool:
        code = getattr(frame, "f_code")
        qualname = code.co_qualname
        return (
            qualname.startswith(
                ("RuntimeFlowTracer.", "TensorOperationTracer.", "torch.")
            )
            or qualname.startswith(("DisableContext.", "_disable_dynamo"))
            or "_call_impl" in qualname
            or "_wrapped_call_impl" in qualname
            or qualname.endswith(".__call__")
            or qualname.endswith("._fn")
        )

    def runtime_function_name(self) -> str:
        frame = sys._getframe()
        try:
            frame = frame.f_back
            while frame is not None:
                if not self._is_tracer_frame(frame):
                    return frame.f_code.co_qualname
                frame = frame.f_back
        finally:
            del frame
        return "<unknown>"

    def runtime_local_reference(self, value: object) -> str | None:
        frame = sys._getframe()
        try:
            frame = frame.f_back
            while frame is not None:
                if not self._is_tracer_frame(frame):
                    for name, candidate in frame.f_locals.items():
                        if candidate is value:
                            return f"{frame.f_code.co_qualname}.{name}"
                frame = frame.f_back
        finally:
            del frame
        return None

    @staticmethod
    def _is_trackable(value: object) -> bool:
        return isinstance(value, (torch.Tensor, tuple, list, dict))

    @classmethod
    def _iter_references(
        cls,
        value: object,
        path: str,
        seen: set[int],
    ) -> Iterator[tuple[object, str]]:
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if cls._is_trackable(value):
            yield value, path

        if isinstance(value, dict):
            for key, item in value.items():
                yield from cls._iter_references(
                    item,
                    f"{path}[{key!r}]",
                    seen,
                )
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from cls._iter_references(
                    item,
                    f"{path}[{index}]",
                    seen,
                )

    @staticmethod
    def _named_inputs(
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        keyword_inputs: dict[str, object],
    ) -> Iterator[tuple[str, object]]:
        parameter_names = get_positional_parameter_names(module)
        for index, value in enumerate(inputs):
            parameter_name = (
                parameter_names[index]
                if index < len(parameter_names)
                else f"arg{index}"
            )
            yield parameter_name, value
        yield from keyword_inputs.items()

    def begin_module_call(
        self,
        module_name: str,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        keyword_inputs: dict[str, object],
    ) -> None:
        self.observe_inputs(module_name, module, inputs, keyword_inputs)
        input_scope: list[tuple[int, tuple[str, str, object]]] = []
        for parameter_name, value in self._named_inputs(
            module,
            inputs,
            keyword_inputs,
        ):
            for input_value, input_path in self._iter_references(
                value,
                parameter_name,
                set(),
            ):
                source = (
                    module_name,
                    input_path,
                    input_value,
                )
                self._active_input_sources.setdefault(
                    id(input_value),
                    [],
                ).append(source)
                input_scope.append((id(input_value), source))
        self._input_scopes.setdefault(module_name, []).append(input_scope)

    def end_module_call(self, module_name: str) -> None:
        input_scopes = self._input_scopes[module_name]
        input_scope = input_scopes.pop()
        if not input_scopes:
            del self._input_scopes[module_name]

        for value_id, source in input_scope:
            active_sources = self._active_input_sources[value_id]
            for index in range(len(active_sources) - 1, -1, -1):
                candidate = active_sources[index]
                if (
                    candidate[0] == source[0]
                    and candidate[1] == source[1]
                    and candidate[2] is source[2]
                ):
                    del active_sources[index]
                    break
            if not active_sources:
                del self._active_input_sources[value_id]

    def observe_inputs(
        self,
        module_name: str,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
        keyword_inputs: dict[str, object],
    ) -> None:
        printed_edges: set[tuple[str, str, str]] = set()
        for parameter_name, value in self._named_inputs(
            module,
            inputs,
            keyword_inputs,
        ):
            for input_value, input_path in self._iter_references(
                value,
                parameter_name,
                set(),
            ):
                for output_label, output_path in self._find_producers(
                    input_value,
                ):
                    edge = (output_label, output_path, input_path)
                    if edge in printed_edges:
                        continue
                    printed_edges.add(edge)
                    source_reference = format_flow_reference(
                        output_label,
                        output_path,
                    )
                    target_reference = (
                        f"{module_name}.input["
                        f"{format_input_path(input_path)}]"
                    )
                    print(
                        "runtime identity flow: "
                        f"{source_reference} -> {target_reference}"
                    )

    def _find_producers(
        self,
        value: object,
    ) -> Iterator[tuple[str, str]]:
        active_sources = self._active_input_sources.get(id(value), ())
        for source_label, source_path, source_value in reversed(active_sources):
            if source_value is value:
                yield source_label, source_path
                return

        for output_label, output_path, producer_value in reversed(
            self._producers.get(id(value), ())
        ):
            if producer_value is value:
                yield output_label, output_path
                return

        for operation_label, output_path, producer_reference in reversed(
            self._operation_producers.get(id(value), ())
        ):
            if producer_reference() is value:
                yield operation_label, output_path
                return

    def record_output(self, output: object) -> str:
        output_label = f"@{self._next_output_number}"
        self._next_output_number += 1
        for value, path in self._iter_references(output, "output", set()):
            self._producers.setdefault(id(value), []).append(
                (output_label, path, value)
            )
        return output_label

    def record_operation(self, operator_name: str, output: object) -> str:
        operation_label = (
            f"op@{self._next_operation_number}<{operator_name}>"
        )
        self._next_operation_number += 1
        for value, path in self._iter_references(output, "output", set()):
            if not isinstance(value, torch.Tensor):
                continue
            try:
                value_reference = weakref.ref(value)
            except TypeError:
                continue
            self._operation_producers.setdefault(id(value), []).append(
                (operation_label, path, value_reference)
            )
        return operation_label

    def print_operation(
        self,
        operation_label: str,
        args: tuple[object, ...],
        keyword_args: dict[str, object],
        output: object,
    ) -> None:
        print(f"runtime tensor op: {operation_label}")
        named_values = tuple(
            (f"arg{index}", value) for index, value in enumerate(args)
        ) + tuple(keyword_args.items())
        for input_name, value in named_values:
            for input_value, input_path in self._iter_references(
                value,
                input_name,
                set(),
            ):
                if not isinstance(input_value, torch.Tensor):
                    continue
                producer = next(self._find_producers(input_value), None)
                if producer is not None:
                    source = format_flow_reference(*producer)
                else:
                    local_reference = self.runtime_local_reference(input_value)
                    if local_reference is not None:
                        source = format_runtime_local_reference(local_reference)
                    else:
                        source = self._record_external_input(input_value)
                print(
                    f"  input {format_input_path(input_path)} <- "
                    f"{source}: {describe(input_value, 0)}"
                )
        print(
            f"  output {format_flow_reference(operation_label, 'output')}: "
            f"{describe(output, 0)}"
        )


class TensorOperationTracer(TorchDispatchMode):
    def __init__(self, flow_tracer: RuntimeFlowTracer) -> None:
        super().__init__()
        self._flow_tracer = flow_tracer

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        keyword_args = {} if kwargs is None else kwargs
        output = func(*args, **keyword_args)
        operator_name = str(func)
        runtime_function_name = self._flow_tracer.runtime_function_name()
        operation_label = self._flow_tracer.record_operation(
            f"{runtime_function_name}:{operator_name}",
            output,
        )
        if operator_name.startswith(TRACE_TENSOR_OPERATION_PREFIXES):
            self._flow_tracer.print_operation(
                operation_label,
                args,
                keyword_args,
                output,
            )
        return output


def print_new_shape_meanings(
    prefix: str,
    value: object,
    labels: tuple[str, ...] | None,
    explained_dimensions: set[str],
) -> None:
    if labels is None:
        return

    if isinstance(value, torch.Tensor):
        if len(value.shape) != len(labels):
            return
        new_lines = []
        for index, label in enumerate(labels):
            if label in explained_dimensions:
                continue
            explained_dimensions.add(label)
            new_lines.append(
                f"  dim {index}: {DIMENSION_NAMES[label]} = "
                f"{int(value.shape[index])}"
            )
        if new_lines:
            print_comment(f"{prefix} new dimension meanings:")
            for line in new_lines:
                print_comment(line)
        return

    if isinstance(value, dict):
        for item in value.values():
            print_new_shape_meanings(
                prefix,
                item,
                labels,
                explained_dimensions,
            )
        return

    if isinstance(value, (tuple, list)):
        for item in value:
            print_new_shape_meanings(
                prefix,
                item,
                labels,
                explained_dimensions,
            )


def make_boundary_hook(
    name: str,
    sample_values: int,
    explained_dimensions: set[str],
    tracer: RuntimeFlowTracer,
    input_labels: tuple[str, ...] | None = None,
    keyword_labels: dict[str, tuple[str, ...]] | None = None,
    output_labels: tuple[str, ...] | None = None,
):
    def hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
        keyword_inputs: dict[str, object],
        output: object,
    ) -> None:
        print(f"\n{format_layer_name(name)}")
        if name == "model.model.rotary_emb" and "position_ids" in keyword_inputs:
            print_comment(
                "position_ids 来源：本脚本调用 model(**inputs) 未显式传入；"
                "LlamaModel.forward 按 token 序列生成位置编号，"
                "再以关键字参数传给 rotary_emb。"
            )
        output_label = tracer.record_output(output)
        tracer.end_module_call(name)

        print(f"input args:   {describe_named_inputs(_module, inputs, sample_values)}")
        print(
            f"input kwargs: {describe_named_kwargs(keyword_inputs, sample_values)}"
        )
        print(f"output ref:   {output_label} ({name}.forward return)")
        print(f"output:       {describe(output, sample_values)}")

        print_new_shape_meanings(
            "input args",
            inputs,
            input_labels,
            explained_dimensions,
        )
        if keyword_labels is not None:
            for key, labels in keyword_labels.items():
                if key in keyword_inputs:
                    print_new_shape_meanings(
                        f"input kwargs[{key}]",
                        keyword_inputs[key],
                        labels,
                        explained_dimensions,
                    )
        print_new_shape_meanings(
            "output",
            output,
            output_labels,
            explained_dimensions,
        )

    return hook


def make_projection_hook(
    name: str,
    head_dim: int,
    head_label: str,
    sample_values: int,
    explained_dimensions: set[str],
    tracer: RuntimeFlowTracer,
):
    def hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
        keyword_inputs: dict[str, object],
        output: object,
    ) -> None:
        print(f"\n{format_layer_name(name)}")
        output_label = tracer.record_output(output)
        tracer.end_module_call(name)

        print(f"input args:   {describe_named_inputs(_module, inputs, sample_values)}")
        print(
            f"input kwargs: {describe_named_kwargs(keyword_inputs, sample_values)}"
        )
        print(f"output ref:   {output_label} ({name}.forward return)")
        print(f"output:       {describe(output, sample_values)}")
        print_new_shape_meanings(
            "input args",
            inputs,
            HIDDEN_SHAPE,
            explained_dimensions,
        )
        print_new_shape_meanings(
            "output",
            output,
            HIDDEN_SHAPE,
            explained_dimensions,
        )

        if not isinstance(output, torch.Tensor):
            return

        projected_size = output.shape[-1]
        if projected_size % head_dim != 0:
            print(
                f"reshaped output: unavailable; "
                f"projected_size={projected_size} is not divisible by "
                f"head_dim={head_dim}"
            )
            return

        head_count = projected_size // head_dim
        reshaped = output.view(
            *output.shape[:-1],
            head_count,
            head_dim,
        ).transpose(-3, -2)
        print_comment(
            f"reshaped output: after view(..., {head_count}, {head_dim})"
            f".transpose(-3, -2): {describe(reshaped, sample_values)}"
        )
        print_new_shape_meanings(
            "reshaped output",
            reshaped,
            ("batch", head_label, "sequence", "head_dim"),
            explained_dimensions,
        )

    return hook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print activation input/output and QKV shapes."
    )
    parser.add_argument(
        "--prompt",
        default="Llama 2 is",
        help="text used for one forward pass",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="layer indices, for example 0,7,31; default: all",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="model device; default: cuda",
    )
    parser.add_argument(
        "--sample-values",
        type=int,
        default=0,
        help="print this many flattened values from each tensor; default: 0",
    )
    parser.add_argument(
        "--trace-tensor-ops",
        action="store_true",
        help=(
            "trace arithmetic and matrix Tensor operations; "
            "adds runtime overhead"
        ),
    )
    args = parser.parse_args()

    if args.sample_values < 0:
        raise ValueError("--sample-values must be non-negative")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
    )
    model = model.to(args.device).eval()

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(MODEL_PATH),
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    layer_indices = parse_layers(args.layers, len(model.model.layers))

    input_ids = inputs["input_ids"][0].tolist()
    print(f"prompt: {args.prompt!r}")
    print(f"input_ids: {input_ids}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")
    print(f"watched layers: {layer_indices}")

    handles = []
    explained_dimensions: set[str] = set()
    flow_tracer = RuntimeFlowTracer()

    def watch(
        name: str,
        module: torch.nn.Module,
        input_labels: tuple[str, ...] | None = None,
        keyword_labels: dict[str, tuple[str, ...]] | None = None,
        output_labels: tuple[str, ...] | None = None,
    ) -> None:
        def begin_call(
            called_module: torch.nn.Module,
            called_inputs: tuple[object, ...],
            called_keyword_inputs: dict[str, object],
        ) -> None:
            flow_tracer.begin_module_call(
                name,
                called_module,
                called_inputs,
                called_keyword_inputs,
            )

        handles.append(
            module.register_forward_pre_hook(begin_call, with_kwargs=True)
        )

        handles.append(
            module.register_forward_hook(
                make_boundary_hook(
                    name,
                    args.sample_values,
                    explained_dimensions,
                    flow_tracer,
                    input_labels,
                    keyword_labels,
                    output_labels,
                ),
                with_kwargs=True,
            )
        )

    def watch_projection(
        name: str,
        module: torch.nn.Module,
        head_dim: int,
        head_label: str,
    ) -> None:
        def begin_call(
            called_module: torch.nn.Module,
            called_inputs: tuple[object, ...],
            called_keyword_inputs: dict[str, object],
        ) -> None:
            flow_tracer.begin_module_call(
                name,
                called_module,
                called_inputs,
                called_keyword_inputs,
            )

        handles.append(
            module.register_forward_pre_hook(begin_call, with_kwargs=True)
        )

        handles.append(
            module.register_forward_hook(
                make_projection_hook(
                    name,
                    head_dim,
                    head_label,
                    args.sample_values,
                    explained_dimensions,
                    flow_tracer,
                ),
                with_kwargs=True,
            )
        )

    watch(
        "model.embed_tokens",
        model.model.embed_tokens,
        input_labels=BATCH_SEQUENCE,
        output_labels=HIDDEN_SHAPE,
    )
    watch(
        "model.model.rotary_emb",
        model.model.rotary_emb,
        input_labels=HIDDEN_SHAPE,
        keyword_labels={"position_ids": BATCH_SEQUENCE},
        output_labels=POSITION_EMBEDDING_SHAPE,
    )

    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]
        attention = layer.self_attn
        attention_head_dim = int(vars(attention)["head_dim"])

        watch(
            f"model.layers[{layer_index}]",
            layer,
            input_labels=HIDDEN_SHAPE,
            keyword_labels={
                "position_ids": BATCH_SEQUENCE,
                "position_embeddings": POSITION_EMBEDDING_SHAPE,
            },
            output_labels=HIDDEN_SHAPE,
        )
        watch(
            f"model.layers[{layer_index}].input_layernorm",
            layer.input_layernorm,
            input_labels=HIDDEN_SHAPE,
            output_labels=HIDDEN_SHAPE,
        )
        watch(
            f"model.layers[{layer_index}].post_attention_layernorm",
            layer.post_attention_layernorm,
            input_labels=HIDDEN_SHAPE,
            output_labels=HIDDEN_SHAPE,
        )
        watch(
            f"model.layers[{layer_index}].self_attn",
            attention,
            keyword_labels={
                "hidden_states": HIDDEN_SHAPE,
                "position_ids": BATCH_SEQUENCE,
                "position_embeddings": POSITION_EMBEDDING_SHAPE,
            },
            output_labels=HIDDEN_SHAPE,
        )
        watch(
            f"model.layers[{layer_index}].mlp",
            layer.mlp,
            input_labels=HIDDEN_SHAPE,
            output_labels=HIDDEN_SHAPE,
        )

        watch_projection(
            f"model.layers[{layer_index}].self_attn.q_proj",
            attention.q_proj,
            attention_head_dim,
            "q_heads",
        )
        watch_projection(
            f"model.layers[{layer_index}].self_attn.k_proj",
            attention.k_proj,
            attention_head_dim,
            "kv_heads",
        )
        watch_projection(
            f"model.layers[{layer_index}].self_attn.v_proj",
            attention.v_proj,
            attention_head_dim,
            "kv_heads",
        )

    watch(
        "model.model.norm",
        model.model.norm,
        input_labels=HIDDEN_SHAPE,
        output_labels=HIDDEN_SHAPE,
    )
    watch(
        "lm_head",
        model.lm_head,
        input_labels=HIDDEN_SHAPE,
        output_labels=LOGITS_SHAPE,
    )

    try:
        tensor_operation_context = (
            TensorOperationTracer(flow_tracer)
            if args.trace_tensor_ops
            else nullcontext()
        )
        with torch.inference_mode(), tensor_operation_context:
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
