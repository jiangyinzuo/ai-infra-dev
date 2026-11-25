"""单卡 tiny DeepSeek：记录 HiSparse 各类从初始化到 Host/hot 推理的入口顺序。"""

import ast
import json
import os
import runpy
import sys
from argparse import Namespace
from pathlib import Path
from typing import cast

from trace_python.call_stack_analysis import Target, TraceConfig

REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = REPOSITORY / "code/vllm-workspace/vllm"
SOURCES = (
    "vllm/config/attention.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/hisparse/stats.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/attention/backends/mla/index_group.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/hisparse/backend.py",
    "vllm/v1/hisparse/coordinator.py",
    "vllm/v1/hisparse/layout.py",
    "vllm/v1/hisparse/runtime.py",
)
ENTRY_FIELDS = (
    "self", "request_id", "request_ids", "page_idx", "restore",
    "num_tokens", "num_computed_tokens", "num_local_computed_tokens",
    "num_external_computed_tokens", "num_host_pages", "num_blocks",
    "block_size", "layer_index", "num_input_rows", "num_slot_rows",
    "descriptor_count", "hits", "misses", "host_to_device_bytes",
    "size", "is_host_writer",
)
DETAIL_FIELDS = {
    "HiSparseConnector.__init__": ("role.name",),
    "HiSparseConnectorScheduler.build_connector_meta": (
        "scheduler_output.num_scheduled_tokens",
    ),
    "HiSparseCoordinator.update_spills": ("enqueued_counts", "completed_counts"),
    "HiSparseConnectorWorker.start_step": ("metadata.source_block_ids",),
    "HiSparseRuntime._resolve_residency": ("topk_indices",),
    "HiSparseRuntime.swap_in": ("logical_topk_indices",),
    "HiSparseCacheHandle.swap_in": ("logical_topk_indices",),
}
NOTES = {
    "HiSparseConnector.__init__": "创建 Scheduler 或 Worker 侧连接器",
    "HiSparseConnectorScheduler.build_connector_meta": "汇总本步迁移和映射计划",
    "HiSparseConnectorScheduler.update_connector_output": "回收 Worker 完成通知",
    "HiSparseHostPool.allocate": "分配并固定 Host KV 内存",
    "HiSparseSourceManager.allocate_new_blocks": "分配 Host KV block",
    "HiSparseResidentManager.allocate_new_blocks": "分配 GPU resident block",
    "HiSparseHotManager.allocate_new_blocks": "分配 GPU hot buffer",
    "HiSparseCoordinator.plan_prefix_materialization": "准备前缀复用所需 KV",
    "HiSparseCoordinator.update_residency": "规划 Host/GPU 驻留状态",
    "HiSparseCoordinator._plan_page_transfer": "规划页面写回或恢复",
    "HiSparseCoordinator.update_spills": "处理页面迁移完成通知",
    "HiSparseCoordinator._page_became_clean": "确认 Host 页面可读",
    "HiSparseCoordinator.free": "释放请求的驻留与迁移状态",
    "HiSparseConnectorWorker.initialize": "绑定 Host、resident 和 hot 区域",
    "HiSparseConnectorWorker.start_step": "接收本步迁移和映射",
    "HiSparseConnectorWorker._submit_dma_descriptors": "提交 GPU→Host 异步拷贝",
    "HiSparseConnectorWorker._restore_pages": "Host→GPU resident 整页恢复",
    "HiSparseCacheHandle.swap_in": "Attention 请求本层选中 KV",
    "HiSparseRuntime._resolve_residency": "解析选中 KV 的驻留位置",
    "HiSparseRuntime._swap_rows": "Host→GPU hot 选中行换入",
    "HiSparseRuntime.wait_for_staged_swap": "计算流等待换入完成",
    "HiSparseConnectorWorker.take_transfer_updates": "报告已提交和已完成的迁移",
    "HiSparseConnectorWorker.shutdown": "同步搬运流并释放缓存绑定",
}


def _discover():
    targets = []
    inventory = []
    for file in SOURCES:
        tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            if not (
                cls.name.startswith("HiSparse")
                or cls.name == "CUDAHiSparseBackend"
            ):
                continue
            methods = []
            for method in cls.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = f"{cls.name}.{method.name}"
                parameters = {
                    arg.arg for arg in (
                        *method.args.posonlyargs,
                        *method.args.args,
                        *method.args.kwonlyargs,
                    )
                }
                fields = tuple(f for f in ENTRY_FIELDS if f in parameters)
                fields += DETAIL_FIELDS.get(name, ())
                # 较高额度保留本场景内完整顺序；验收时比较命中总数和保存数。
                targets.append(Target(
                    file, name, fields, limit=1_000_000, note=NOTES.get(name),
                ))
                methods.append(name)
            inventory.append({
                "class": cls.name,
                "file": file,
                "bases": [ast.unparse(base) for base in cls.bases],
                "source_methods": methods,
                "related_backend": cls.name == "CUDAHiSparseBackend",
            })
    return tuple(targets), tuple(inventory)


CLASS_TARGETS, CLASS_INVENTORY = _discover()


def _phase(name: str, request_id: str | None = None) -> None:
    print("TRACE_PHASE", json.dumps({
        "name": name, "request_id": request_id,
    }), flush=True)


def run(start):
    # 先启用再导入/构造引擎，不能漏掉各类的创建和绑定顺序。
    start()
    _phase("engine_init")
    sys.path.insert(0, str(REPOSITORY / "code/dev"))
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.outputs import RequestOutput
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine

    serve = runpy.run_path(str(
        REPOSITORY / "code/dev/serve_config/deepseek-v3.2-hisparse.py"
    ))["config"]
    arguments = list(serve.serve_command[3:])
    arguments[arguments.index("--max-model-len") + 1] = "16384"
    index = arguments.index("--attention-config") + 1
    attention = json.loads(arguments[index])
    attention["hisparse_config"] = {
        "device_buffer_size": 2048, "eager_host_mirror": False,
    }
    arguments[index] = json.dumps(attention)
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = cast(Namespace, parser.parse_args([
        "--model", serve.model, *arguments,
        "--distributed-executor-backend", "uni",
        "--no-enable-extensible-kv-cache", "--num-gpu-blocks-override", "512",
    ]))
    print("CLASS_INVENTORY", json.dumps(CLASS_INVENTORY), flush=True)
    engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(parsed))
    try:
        # Prefill 先留在 resident；Decode 增长后触发 Host/hot，避免长 Prefill staging。
        # 3585=56*64+1，重复前缀保留一个 token 走 Decode 而非 Host Prefill。
        prompt = [32000 + (index * 17 + 661) % 1000 for index in range(3585)]
        for request_id, output_tokens in (
            ("hisparse-order-cold", 768),
            ("hisparse-order-prefix", 8),
        ):
            _phase("request", request_id)
            engine.add_request(
                request_id, {"prompt_token_ids": prompt},
                SamplingParams(
                    temperature=0, max_tokens=output_tokens, ignore_eos=True,
                ),
            )
            completed = False
            while engine.has_unfinished_requests():
                for output in engine.step():
                    output = cast(RequestOutput, output)
                    if not output.finished:
                        continue
                    completed = True
                    generated = len(output.outputs[0].token_ids)
                    print("REQUEST_RESULT", json.dumps({
                        "request_id": output.request_id,
                        "prompt_tokens": len(prompt),
                        "cached_tokens": output.num_cached_tokens,
                        "generated_tokens": generated,
                        "finish_reason": output.outputs[0].finish_reason,
                    }), flush=True)
                    if generated != output_tokens:
                        raise RuntimeError(f"{request_id}: 输出 token 数不符")
            if not completed:
                raise RuntimeError(f"{request_id}: 没有最终结果")
            _phase("request_finished", request_id)
        engine.do_log_stats()
    finally:
        _phase("engine_shutdown")
        engine.engine_core.shutdown()
        _phase("finished")


config = TraceConfig(
    root=ROOT,
    run=run,
    env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    process_label=os.environ.get("TRACE_PROCESS_LABEL", "hisparse-order"),
    targets=(
        Target(str(Path(__file__).resolve()), "_phase", ("name", "request_id"), 16),
        *CLASS_TARGETS,
    ),
)
