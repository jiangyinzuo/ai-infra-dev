"""Qwen3-Next tiny random：从初始化到请求回收观察混合 KV cache group。"""

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
PLANNING = "vllm/v1/core/kv_cache_planning.py"
COORDINATOR = "vllm/v1/core/kv_cache_coordinator.py"
SINGLE = "vllm/v1/core/single_type_kv_cache_manager.py"
MANAGER = "vllm/v1/core/kv_cache_manager.py"
RUNNER = "vllm/v1/worker/gpu/model_runner.py"
TABLE = "vllm/v1/worker/gpu/block_table.py"
POOL = "vllm/v1/core/block_pool.py"
GROUP = ("self.kv_cache_group_id", "self.block_size", "self.group_block_size", "self.scheduler_block_size")
ALLOCATION = ("request_id", "num_tokens", "num_tokens_main_model")


def _print_state(coordinator, request_id, step):
    rows = []
    for manager in coordinator.single_type_managers:
        blocks = manager.req_to_blocks.get(request_id, ())
        rows.append({
            "group_id": manager.kv_cache_group_id,
            "manager": type(manager).__name__,
            "num_blocks": len(blocks),
            "block_ids": [block.block_id for block in blocks],
        })
    print("GROUP_BLOCKS", json.dumps({
        "request_id": request_id, "step": step, "groups": rows,
        "free_blocks": coordinator.block_pool.free_block_queue.num_free_blocks,
    }), flush=True)


def run(start):
    # 与现有单进程场景一致：首次导入模型前设置执行上下文，不改 trace 核心。
    sys.path.insert(0, str(REPOSITORY / "code/dev"))
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.outputs import RequestOutput
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.kv_cache_interface import MambaSpec

    serve = runpy.run_path(str(
        REPOSITORY / "code/dev/serve_config/qwen3-next-kv-groups.py"
    ))["config"]
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = cast(Namespace, parser.parse_args([
        "--model", serve.model, *serve.serve_command[3:],
        "--distributed-executor-backend", "uni",
    ]))
    start()
    engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(parsed))
    try:
        core = engine.engine_core.engine_core
        kv_manager = core.scheduler.kv_cache_manager
        cache_config = kv_manager.kv_cache_config
        coordinator = kv_manager.coordinator
        managers = coordinator.single_type_managers
        groups = cache_config.kv_cache_groups
        layout = []
        full_block_sizes = []
        has_mamba = False
        for group_id, (group, manager) in enumerate(zip(groups, managers, strict=True)):
            layer_specs = list(group.kv_cache_specs.values())
            if not layer_specs:
                raise RuntimeError(f"KV cache group {group_id} 没有 layer spec")
            spec = layer_specs[0]
            has_mamba |= any(isinstance(s, MambaSpec) for s in layer_specs)
            if isinstance(manager, FullAttentionManager):
                full_block_sizes.append(group.block_size)
            layout.append({
                "group_id": group_id, "layers": group.layer_names,
                "spec": type(spec).__name__, "manager": type(manager).__name__,
                "block_size": group.block_size, "page_size_bytes": group.page_size_bytes,
                "host_resident": group.host_resident,
                "state_shapes": [s.shapes for s in layer_specs if isinstance(s, MambaSpec)],
            })
        print("GROUP_LAYOUT", json.dumps({
            "num_groups": len(groups), "pool_blocks": cache_config.num_blocks,
            "scheduler_block_size": coordinator.scheduler_block_size, "groups": layout,
        }), flush=True)
        if not full_block_sizes or not has_mamba:
            raise RuntimeError("预期同时存在 Full Attention 和 GDN/Mamba 状态缓存")
        if any(group.host_resident for group in groups):
            raise RuntimeError("本场景不应启用 Host/HiSparse 缓存")

        block_size = max(full_block_sizes)
        # 使用解析后的 Attention block size，避免把 CLI 默认值当成最终布局。
        boundary = max(2, 1024 // block_size) * block_size
        free_before = coordinator.block_pool.free_block_queue.num_free_blocks
        for request_id, prompt_len in (
            ("groups-short", boundary - 1),
            ("groups-long", boundary + block_size - 1),
        ):
            if prompt_len + 8 > engine.vllm_config.model_config.max_model_len:
                raise RuntimeError("解析后的 block size 太大，需调整本学习场景的上下文上限")
            engine.add_request(
                request_id, {"prompt_token_ids": [100] * prompt_len},
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            )
            # InputProcessor 会给内部 ID 加随机后缀；单请求场景直接取调度器的真实 key。
            internal_request_id, = core.scheduler.requests
            _print_state(coordinator, internal_request_id, 0)
            completed = False
            step = 0
            while engine.has_unfinished_requests():
                outputs = engine.step()
                step += 1
                _print_state(coordinator, internal_request_id, step)
                for output in outputs:
                    output = cast(RequestOutput, output)
                    if not output.finished:
                        continue
                    completed = True
                    generated = len(output.outputs[0].token_ids)
                    print("REQUEST_RESULT", json.dumps({
                        "request_id": output.request_id, "prompt_tokens": prompt_len,
                        "internal_request_id": internal_request_id,
                        "generated_tokens": generated,
                        "finish_reason": output.outputs[0].finish_reason,
                    }), flush=True)
                    if generated != 8 or output.outputs[0].finish_reason != "length":
                        raise RuntimeError(f"{request_id}: 请求未按预期生成 8 token")
            if not completed:
                raise RuntimeError(f"{request_id}: 缺少最终输出")
            if any(internal_request_id in manager.req_to_blocks for manager in managers):
                raise RuntimeError(f"{request_id}: 请求结束后仍持有 group block")
            free_after = coordinator.block_pool.free_block_queue.num_free_blocks
            if free_after != free_before:
                raise RuntimeError(f"{request_id}: block 未完全归还：{free_before} -> {free_after}")
    finally:
        engine.engine_core.shutdown()


config = TraceConfig(
    root=ROOT,
    run=run,
    env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    process_label=os.environ.get("TRACE_PROCESS_LABEL", "qwen-kv-groups"),
    targets=(
        # 初始化：输入逐层规格、页面对齐、最终分组与 manager 选择。
        Target(PLANNING, "get_kv_cache_groups", (
            "layer_configs", "vllm_config.scheduler_config.disable_hybrid_kv_cache_manager",
        ), 4, note="按缓存规格划分层组"),
        Target(PLANNING, "unify_kv_cache_spec_page_size", ("layer_configs",), 4,
               note="对齐每层物理页大小"),
        Target(PLANNING, "get_kv_cache_groups_from_builder", (
            "builder", "layer_configs",
            "vllm_config.model_config.supports_kv_cache_grouping",
        ), 4, note="通过当前 builder 入口构造缓存分组"),
        Target(SINGLE, "create_single_type_manager", (
            "kv_cache_group.layer_names", "kv_cache_group.manager_cls",
            "kv_cache_group.block_size", "max_in_flight_tokens", "max_model_len",
        ), 16, caller_fields=("i", "kv_cache_group.layer_names"),
               note="选择各组的缓存管理器"),
        Target(SINGLE, "SingleTypeKVCacheManager.__init__", (
            "kv_cache_group_id", "kv_cache_group.layer_names", "kv_cache_group.block_size",
            "scheduler_block_size", "enable_caching",
        ), 16),
        Target(RUNNER, "GPUModelRunner.initialize_kv_cache", (
            "kv_cache_config.num_blocks", "kv_cache_config.kv_cache_groups",
        ), 4),
        Target(TABLE, "BlockTables.__init__", (
            "block_sizes", "max_num_blocks_per_group", "slot_mapping_enabled",
        ), 4, note="为每组建立独立映射"),
        # 同一请求经 coordinator 分发到各组；Full 与 Mamba 的分配路径不同。
        Target(MANAGER, "KVCacheManager.allocate_slots", (
            "request.request_id", "num_new_tokens", "self.num_kv_cache_groups",
        ), 64),
        Target(COORDINATOR, "KVCacheCoordinator.allocate_new_blocks", ALLOCATION, 64,
               note="将请求分配到各组"),
        Target(SINGLE, "SingleTypeKVCacheManager.allocate_new_blocks", GROUP + ALLOCATION, 128,
               note="按 token 数扩展 block"),
        Target(SINGLE, "MambaManager.allocate_new_blocks", GROUP + ALLOCATION, 128,
               note="分配或复用 GDN 状态"),
        Target(POOL, "BlockPool.get_new_blocks", (
            "num_blocks", "self.free_block_queue.num_free_blocks",
        ), 128),
        Target(TABLE, "BlockTables.append_block_ids", (
            "req_index", "new_block_ids", "overwrite",
        ), 64, note="写入各组的物理 block ID"),
        Target(MANAGER, "KVCacheManager.free", ("request.request_id",), 4,
               note="请求结束，回收所有组"),
        Target(SINGLE, "SingleTypeKVCacheManager.free", GROUP + ("request_id",), 16),
    ),
)
