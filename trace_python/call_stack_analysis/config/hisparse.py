"""单请求 HiSparse 学习配置；用 vLLM 虚拟环境运行，设备由调用者选择。"""
import runpy
import sys
from argparse import Namespace
from pathlib import Path
from typing import cast
from trace_python.call_stack_analysis import Target, TraceConfig

ROOT = Path(__file__).resolve().parents[3] / "code/vllm-workspace/vllm"
REQUEST = ("request.request_id", "request.num_computed_tokens", "request.num_tokens",
           "request.num_prompt_tokens", "request.num_output_tokens")


def run(start):
    sys.path.insert(0, str(ROOT.parents[1] / "dev"))
    # 必须等追踪器设置 config.env 后再导入 vLLM，避免提前缓存多进程配置。
    # pylint: disable=import-outside-toplevel
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.outputs import RequestOutput
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine
    # pylint: enable=import-outside-toplevel

    serve = runpy.run_path(str(ROOT.parents[1] / "dev/serve_config/deepseek-v3.2-hisparse.py"))["config"]
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    # 上游 parse_args 的推断类型包含 None；这里使用 argparse 的 Namespace 返回约定。
    parsed = cast(Namespace, parser.parse_args([
        "--model", serve.model, *serve.serve_command[3:],
        "--distributed-executor-backend", "uni",
    ]))
    args = EngineArgs.from_cli_args(parsed)
    # 初始化与预热也属于本次观察范围，必须在创建引擎前启动采集。
    start()
    engine = LLMEngine.from_engine_args(args)
    engine.add_request("learn-hisparse", {"prompt_token_ids": [100] * 2304},
                       SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    # LLMEngine可以同时处理多个请求，只要还有unfinished requests，就继续处理。
    while engine.has_unfinished_requests():
        # 一次engine.step()可能返回多个请求的输出
        for output in engine.step():
            # 本配置仅提交生成请求；step 的公共类型也包含 pooling 请求。
            output = cast(RequestOutput, output)
            if output.finished:
                print("REQUEST_RESULT", output.request_id,
                      "generated_tokens", len(output.outputs[0].token_ids),
                      "finish_reason", output.outputs[0].finish_reason)


config = TraceConfig(
    root=ROOT, run=run,
    env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    targets=(
        Target("vllm/v1/engine/llm_engine.py", "LLMEngine.add_request", ("request_id",)),
        Target("vllm/v1/core/sched/scheduler.py", "Scheduler.add_request", REQUEST),
        Target("vllm/v1/core/sched/scheduler.py", "Scheduler.schedule", ("self.current_step",), 16),
        Target("vllm/v1/core/kv_cache_manager.py", "KVCacheManager.get_computed_blocks", REQUEST),
        Target("vllm/v1/core/kv_cache_manager.py", "KVCacheManager.allocate_slots",
               REQUEST + ("num_new_tokens", "full_sequence_must_fit"), 16),
        Target("vllm/v1/core/block_pool.py", "BlockPool.get_new_blocks", ("num_blocks",)),
        Target("vllm/v1/worker/gpu/model_runner.py", "GPUModelRunner.execute_model",
               ("dummy_run", "scheduler_output.num_scheduled_tokens")),
        # 从元数据准备追到 SM120 执行与 kernel 调用入口；只记录张量描述，不读取 GPU 数据。
        Target("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
               "FlashInferMLASparseSM120Impl.__init__",
               ("self", "num_heads", "head_size", "kv_cache_dtype", "indexer", "mla_args"), 16),
        Target("vllm/model_executor/layers/attention/sparse_mla_attention.py",
               "SparseMLACommonMetadataBuilder.build",
               ("common_prefix_len", "common_attn_metadata.num_actual_tokens"), 16),
        Target("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
               "FlashInferMLASparseSM120Impl.forward_mqa",
               ("q", "attn_metadata.num_decode_tokens", "self.num_heads"), 64),
        Target("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
               "FlashInferMLASparseSM120Impl._run_mqa_kernel",
               ("q", "kv_cache", "topk_indices_physical"), 64),
        Target("vllm/v1/hisparse/coordinator.py", "HiSparseCoordinator.plan_prefix_materialization",
               ("request_id", "num_computed_tokens")),
        Target("vllm/v1/hisparse/coordinator.py", "HiSparseCoordinator._plan_page_transfer",
               ("request_id", "page_idx", "after_forward", "restore")),
        Target("vllm/v1/hisparse/coordinator.py", "HiSparseCoordinator.build_row_mirrors"),
        # Scheduler 侧把调度结果转换为 residency、spill/restore 和 row-mirror metadata。
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py",
               "HiSparseConnectorScheduler.build_connector_meta",
               ("scheduler_output.num_scheduled_tokens",
                "scheduler_output.scheduled_new_reqs",
                "scheduler_output.scheduled_cached_reqs"), 16),
        # Worker 侧异步传输完成后，Scheduler 侧据此回收 Host/GPU block 引用。
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py",
               "HiSparseConnectorScheduler.update_connector_output",
               ("connector_output.finished_recving",
                "connector_output.failed_recving",
                "connector_output.kv_connector_worker_meta"), 16),
        # Worker 每步接收 Scheduler metadata；这里能看到 residency 更新和页面搬运计划是否真正下发。
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
               "HiSparseConnectorWorker.start_step",
               ("metadata.command", "metadata.host_block_copies",
                "metadata.residency_updates", "request_ids", "num_tokens"), 32),
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
               "HiSparseConnectorWorker._restore_pages", ("transfers",), 16),
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
               "HiSparseConnectorWorker._submit_transfers", ("transfers",), 16),
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
               "HiSparseConnectorWorker._submit_dma_descriptors", ("descriptor_count",)),
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
               "HiSparseConnectorWorker.finish_forward"),
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py",
               "HiSparseConnectorWorker.take_transfer_updates"),
        Target("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py",
               "HiSparseConnector.__init__",
               ("role", "kv_cache_config.hisparse_host_num_blocks")),
        Target("vllm/v1/worker/gpu/kv_connector.py", "ActiveKVConnector.__init__",
               ("vllm_config", "kv_caches_dict")),
        Target("vllm/v1/hisparse/runtime.py", "HiSparseRuntime.swap_in",
               ("logical_topk_indices", "block_size", "resident.decode_batch")),
        Target("vllm/v1/hisparse/runtime.py", "HiSparseRuntime._resolve_residency",
               ("topk_indices", "self.is_group_leader")),
        Target("vllm/v1/hisparse/runtime.py", "HiSparseRuntime._swap_rows"),
        Target("vllm/v1/core/sched/scheduler.py", "Scheduler.update_from_output",
               ("scheduler_output.num_scheduled_tokens",), 16),
        Target("vllm/v1/core/sched/scheduler.py", "Scheduler._free_request", REQUEST),
        Target("vllm/v1/core/kv_cache_manager.py", "KVCacheManager.free", REQUEST),
        Target("vllm/v1/hisparse/coordinator.py", "HiSparseCoordinator.free", ("request_id",)),
        Target("vllm/v1/engine/output_processor.py", "OutputProcessor.process_outputs"),
    ),
)
