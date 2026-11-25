"""单卡 SM120：观察普通 GPU KV 分配、前缀复用、追加写入和回收；无 HiSparse/NIXL。"""

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
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
MANAGER = "vllm/v1/core/kv_cache_manager.py"
COORDINATOR = "vllm/v1/core/kv_cache_coordinator.py"
POOL = "vllm/v1/core/block_pool.py"
RUNNER = "vllm/v1/worker/gpu/model_runner.py"
TABLE = "vllm/v1/worker/gpu/block_table.py"
SM120 = "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
REQUEST = (
    "request.request_id",
    "request.num_prompt_tokens",
    "request.num_computed_tokens",
    "request.num_output_tokens",
)
FREE = ("self.free_block_queue.num_free_blocks",)
BATCH = ("scheduler_output.num_scheduled_tokens",)


def run(start):
    # 复用现有 tiny random 模型参数，不复制配方；只去掉 Host/跨节点缓存功能。
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
    index = arguments.index("--kv-transfer-config")
    del arguments[index:index + 2]
    index = arguments.index("--attention-config") + 1
    attention = json.loads(arguments[index])
    attention.pop("hisparse_config")
    arguments[index] = json.dumps(attention)
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = cast(Namespace, parser.parse_args([
        "--model", serve.model, *arguments,
        "--distributed-executor-backend", "uni",
    ]))
    engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(parsed))
    try:
        # 初始化/预热不占用入口采样额度；所有请求使用同一个引擎和 block pool。
        if not engine.reset_prefix_cache():
            raise RuntimeError("不能清空初始前缀缓存")
        start()
        # block=64；2303=36*64-1，生成过程中会跨过一个 block 边界。
        prompt = [100] * 2303
        scenarios = (
            ("kv-cold", prompt),
            ("kv-prefix-hit", prompt),
            ("kv-partial-hit", [100] * 2048 + [101] * 255),
            ("kv-after-reset", prompt),
        )
        for request_id, token_ids in scenarios:
            if request_id == "kv-after-reset":
                reset = engine.reset_prefix_cache()
                print("PREFIX_CACHE_RESET", reset, flush=True)
                if not reset:
                    raise RuntimeError("请求结束后仍不能清空前缀缓存")
            engine.add_request(
                request_id, {"prompt_token_ids": token_ids},
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
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
                        "prompt_tokens": len(token_ids),
                        "cached_tokens": output.num_cached_tokens,
                        "generated_tokens": generated,
                        "finish_reason": output.outputs[0].finish_reason,
                    }), flush=True)
                    if generated != 8:
                        raise RuntimeError(f"{request_id}: 输出不足 8 token")
            if not completed:
                raise RuntimeError(f"{request_id}: 没有收到最终输出")
    finally:
        engine.engine_core.shutdown()


config = TraceConfig(
    root=ROOT,
    run=run,
    env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    process_label=os.environ.get("TRACE_PROCESS_LABEL", "kv-cache"),
    targets=(
        # 请求进入、前缀查找；命中结果在 allocate_slots 的入口参数中观察。
        Target(SCHEDULER, "Scheduler.add_request", REQUEST, 4),
        Target(MANAGER, "KVCacheManager.get_computed_blocks", REQUEST, 4,
               note="查找可复用前缀"),
        Target(MANAGER, "KVCacheManager.allocate_slots", REQUEST + (
            "num_new_tokens", "num_new_computed_tokens",
            "self.block_pool.free_block_queue.num_free_blocks",
        ), 64, note="复用命中 block 并分配新增空间"),
        Target(COORDINATOR, "KVCacheCoordinator.get_num_blocks_to_allocate", (
            "request_id", "num_tokens", "total_computed_tokens",
            "num_local_computed_tokens",
        ), 64),
        Target(COORDINATOR, "KVCacheCoordinator.allocate_new_blocks", (
            "request_id", "num_tokens",
        ), 64),
        Target(POOL, "BlockPool.get_new_blocks", ("num_blocks",) + FREE, 64,
               note="从空闲池取得物理 block"),
        Target(POOL, "BlockPool.touch", FREE, 16, note="增加命中 block 的引用"),
        # cache_full_blocks 登记可复用的完整 block；不等同于 GPU 写入完成。
        Target(POOL, "BlockPool.cache_full_blocks", REQUEST + (
            "num_cached_blocks", "num_full_blocks", "block_size",
            "kv_cache_group_id",
        ), 128, note="登记完整 block 的前缀索引"),
        Target(SCHEDULER, "Scheduler._update_after_schedule", BATCH, 64),
        # CPU block ID 列表可直接读数；GPU tensor 仅保存 shape/dtype/device。
        Target(RUNNER, "GPUModelRunner.execute_model", BATCH, 64),
        Target(RUNNER, "GPUModelRunner.update_requests", (
            "scheduler_output.scheduled_cached_reqs.req_ids",
            "scheduler_output.scheduled_cached_reqs.num_computed_tokens",
            "scheduler_output.scheduled_cached_reqs.new_block_ids",
        ), 64),
        Target(TABLE, "BlockTables.append_block_ids", (
            "req_index", "new_block_ids", "overwrite",
        ), 64, note="更新请求的物理 block 映射"),
        Target(TABLE, "BlockTables.gather_block_tables", (
            "idx_mapping", "num_reqs_padded", "self.block_sizes",
        ), 64),
        Target(TABLE, "BlockTables.compute_slot_mappings", (
            "positions", "query_start_loc", "num_tokens_padded",
            "self.block_sizes",
        ), 64, note="计算新 token 的 KV 写入位置"),
        # 当前 DeepSeek-V3.2 非 PCP 路径在 fused_norm_rope 中融合写 MLA/Indexer KV。
        Target("vllm/models/deepseek_v32/common/kernels.py", "fused_norm_rope", (
            "positions", "kv_c", "k_pe", "slot_mapping", "mla_kv_cache",
            "mla_kv_cache_dtype", "indexer_slot_mapping", "indexer_k_cache",
        ), 256, note="融合归一化、RoPE 与 KV 写入"),
        Target(SM120, "FlashInferMLASparseSM120Impl.forward_mqa", (
            "q", "kv_c_and_k_pe_cache", "attn_metadata.num_decode_tokens",
            "attn_metadata.block_table",
        ), 256),
        Target(SM120, "FlashInferMLASparseSM120Impl._run_mqa_kernel", (
            "q", "kv_cache", "topk_indices_physical",
        ), 256, note="按物理 Top-k 读取 KV 计算 Attention"),
        Target(SCHEDULER, "Scheduler.update_from_output", BATCH, 64),
        # 释放请求引用不等同于清除前缀索引，也不等同于归还 GPU 显存。
        Target(SCHEDULER, "Scheduler._free_request", REQUEST, 4),
        Target(MANAGER, "KVCacheManager.free", REQUEST + (
            "self.block_pool.free_block_queue.num_free_blocks",
        ), 4, note="释放请求持有的 block 引用"),
        Target(COORDINATOR, "KVCacheCoordinator.free", ("request.request_id",), 4),
        Target(POOL, "BlockPool.free_blocks", FREE + ("evict_first",), 32),
        Target(MANAGER, "KVCacheManager.reset_prefix_cache", (), 2,
               note="使前缀缓存索引失效"),
        Target(POOL, "BlockPool.reset_prefix_cache", FREE, 2),
        # 只有实际重用已有缓存 block 时才可能命中；本场景不强行制造容量压力。
        Target(POOL, "BlockPool._maybe_evict_cached_block", (
            "block.block_id", "block.ref_cnt", "block.block_hash",
        ), 16),
    ),
)
