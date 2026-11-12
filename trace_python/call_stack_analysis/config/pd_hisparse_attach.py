"""P/D 与 HiSparse 入口采集：角色、计算进度、容量和传输关联。"""

from pathlib import Path

from trace_python.call_stack_analysis import Target, TraceConfig

ROOT = Path(__file__).resolve().parents[3] / "code/vllm-workspace/vllm"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
WORKER = "vllm/v1/worker/gpu/kv_connector.py"
HISPARSE_WORKER = "vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py"
COORDINATOR = "vllm/v1/hisparse/coordinator.py"
RUNTIME = "vllm/v1/hisparse/runtime.py"
PROXY = "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
NIXL = "vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py"
NODE = ("self.vllm_config.kv_transfer_config.engine_id",)
REQUEST = (
    "request.request_id", "request.status.name", "request.num_prompt_tokens",
    "request.num_tokens", "request.num_output_tokens", "request.num_computed_tokens",
)
PROXY_FIELDS = ("endpoint", "request_id")
BATCH = (
    "scheduler_output.num_scheduled_tokens",
    "scheduler_output.has_sync_kv_loads",
)
RESIDENCY = (
    "self.transition_watermark",
    "self.gpu_pool.num_gpu_blocks",
    "self.gpu_pool.free_block_queue.num_free_blocks",
    "self.host_manager.block_pool.num_gpu_blocks",
    "self.host_manager.block_pool.free_block_queue.num_free_blocks",
)


def _attach_only(_start):
    raise RuntimeError(
        "pd_hisparse_attach.py 仅用于 python -m "
        "trace_python.call_stack_analysis.attach --pid ...，不启动模型服务"
    )


config = TraceConfig(
    root=ROOT,
    run=_attach_only,
    targets=(
        # 实际部署的 NIXL toy proxy；只采请求转发入口，不读请求体。
        Target(PROXY, "send_request_to_service", PROXY_FIELDS, 128),
        Target(PROXY, "stream_service_response", PROXY_FIELDS, 128),
        # 框架层交接；不打在 Connector 基类声明或具体传输实现上。
        Target(SCHEDULER, "Scheduler.add_request",
               NODE + ("request.request_id", "request.num_prompt_tokens",
                       "request.max_tokens", "request.kv_transfer_params"), 4),
        # 在 NIXL 子对象上读取，避免顶层 MultiConnector 的 kv_both 混淆身份。
        Target(NIXL, "NixlBaseConnector.get_num_new_matched_tokens",
               ("request.request_id", "self.kv_transfer_config.is_kv_producer",
                "self.kv_transfer_config.is_kv_consumer"), 4),
        Target(SCHEDULER, "Scheduler._connector_finished", REQUEST, 4),
        Target(SCHEDULER, "Scheduler._build_kv_connector_meta",
               ("self.current_step", "self.max_num_scheduled_tokens",
                "scheduler_output.num_scheduled_tokens", "self._kv_fetch_counts"), 8),
        Target("vllm/v1/core/kv_cache_manager.py", "KVCacheManager.allocate_slots",
               REQUEST + ("request.num_in_flight_tokens",
                          "request.num_output_placeholders", "num_new_tokens",
                          "num_new_computed_tokens", "num_external_computed_tokens",
                          "delay_cache_blocks", "full_sequence_must_fit",
                          "self.block_pool.num_gpu_blocks",
                          "self.block_pool.free_block_queue.num_free_blocks"), 8,
               caller_fields=("token_budget", "input_budget")),
        # 绑定 Scheduler metadata、调用通用 start_load_kv、收集异步传输结果。
        Target(WORKER, "ActiveKVConnector.pre_forward",
               BATCH, 8),
        Target(WORKER, "ActiveKVConnector._start_load_kv", (), 2),
        Target(WORKER, "ActiveKVConnector.post_forward", (), 2),
        # D 调度侧：Host import 完成、resident 压力策略、spill/restore 页面计划。
        Target(COORDINATOR, "HiSparseCoordinator.record_pending_host_import",
               ("request_id", "num_tokens"), 4),
        Target(COORDINATOR, "HiSparseCoordinator.complete_host_import",
               ("request_id", "num_computed_tokens"), 4),
        Target(COORDINATOR, "HiSparseCoordinator.update_residency",
               ("request_id",) + RESIDENCY, 8),
        Target(COORDINATOR, "HiSparseCoordinator._plan_page_transfer",
               ("request_id", "page_idx", "restore", "self.next_spill_id"), 64),
        Target(COORDINATOR, "HiSparseCoordinator.update_spills", (), 2),
        # 这里调用者已经检查完成计数，且 pending 尚在局部变量中。
        Target(COORDINATOR, "HiSparseCoordinator._page_became_clean",
               ("request_id", "page_idx"), 40,
               caller_fields=("pending.transfer_id", "pending.restore",
                              "pending.worker_completions",
                              "pending.expected_worker_completions")),
        Target(COORDINATOR, "HiSparseCoordinator._lose_page",
               ("request_id", "page_idx"), 8),
        # 入口前几次常是空 transfers；页面计划和 DMA ID 才能关联有效搬运。
        Target(HISPARSE_WORKER, "HiSparseConnectorWorker._restore_pages", (), 2),
        Target(HISPARSE_WORKER, "HiSparseConnectorWorker._enqueue_transfers", (), 2),
        # 空 transfers 也会进入上面的函数；此入口才有非空 DMA 描述符。
        Target(HISPARSE_WORKER, "HiSparseConnectorWorker._submit_dma_descriptors",
               ("descriptor_count", "transfer_ids", "self.kernel_block_size"), 16,
               caller_fields=("num_layers", "row_bytes", "source.device.type",
                              "destination.device.type")),
        # 只记录 Tensor 描述，不读 GPU 数值；入口命中不等于确有 miss/数据搬运。
        Target(RUNTIME, "HiSparseRuntime.swap_in",
               ("logical_topk_indices", "block_size",
                "resident.all_context_pages_resident", "resident.num_decode_tokens",
                "self.region_stride", "self.eager_host_mirror",
                "self.is_group_leader", "self.resident_source_index",
                "self._swap_staged"), 8),
        Target(RUNTIME, "HiSparseRuntime._resolve_residency", (), 2),
        Target(RUNTIME, "HiSparseRuntime._swap_rows", (), 2),
    ),
)
