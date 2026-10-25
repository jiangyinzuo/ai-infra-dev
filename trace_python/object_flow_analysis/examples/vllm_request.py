"""追踪一个真实 Request 在 EngineCore、调度器、KV cache 与 connector 间的传播。"""
import ast
from dataclasses import replace
from pathlib import Path
import runpy

from trace_python.object_flow_analysis import Origin

# 复用同一模型和请求，不重复维护模型参数；场景的项目导入仍发生在 run 内。
base = runpy.run_path(str(Path(__file__).with_name("vllm_hisparse.py")))["config"]
file = "vllm/v1/engine/core.py"
tree = ast.parse((base.root / file).read_bytes())
line = next(node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "from_engine_core_request"
            and any(isinstance(target, ast.Name) and target.id == "req" for target in node.targets))
config = replace(
    base,
    origin=Origin(file, "EngineCore.preprocess_add_request", line, "req"),
    include=(
        file, "vllm/v1/engine/core_client.py", "vllm/v1/request.py",
        "vllm/v1/core/sched", "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/kv_cache_coordinator.py", "vllm/v1/core/single_type_kv_cache_manager.py",
        "vllm/v1/core/block_pool.py", "vllm/v1/core/encoder_cache_manager.py",
        "vllm/v1/hisparse/coordinator.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/hisparse",
    ),
    max_events=10000,
    max_objects=2000,
)
