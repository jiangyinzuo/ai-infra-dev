<!-- vllmclass -->
:vim:foldlevel=7 nowrap:

DeepSeek-V3.2缩小版 + HiSparse 0.32.1rc1.dev 2026-10-21

## Engine层：LLMEngine、EngineCore、InprocClient
    用户通过python库直接调用推理，在本脚本中，从 from vllm.v1.engine.llm_engine import LLMEngine 开始。
    InprocClient: 进程内客户端，访问 EngineCore。是class EngineCoreClient(ABC)的子类。

```vllmclass
class AsyncLLM(EngineClient):
        self.engine_core = EngineCoreClient.make_async_mp_client(...)
class LLMEngine:
        self.engine_core: InprocClient = EngineCoreClient.make_client(...)
    # VllmConfig: 进程级全局配置
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":

class InprocClient(EngineCoreClient):
        self.engine_core = EngineCore(...)
```

## Scheduler层：Scheduler

```vllmclass
class EngineCore:
        self.scheduler: SchedulerInterface = Scheduler(...)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())

# /home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py:85
# vllm.v1.core.sched.scheduler.Scheduler
class Scheduler(SchedulerInterface):
    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector
```

### Scheduler.schedule

`Scheduler.schedule`：
- 在本轮 token 计算预算和 KV cache 容量等约束下，决定哪些请求向前推进、各推进多少 token，为这些计算准备好 KV block，再生成交给模型执行器的 SchedulerOutput。
- 构造下一轮动态 batch 的执行计划，并同步更新调度侧状态。

prefill、decode、chunked prefill都可以放进同一个补齐计算进度的统一模型。

vllm/vllm:vllm/v1/core/sched/output.py:231-232
```
@dataclass
class SchedulerOutput:
```

## KV Cache管理层：KVCacheManager、KVCacheCoordinator、SingleTypeKVCacheManager、BlockPool

## KVConnector层

协调外部 KVCache 的复用、传输及相关 block 生命周期。
```
  subtypes (ty) in kv_connector.py
  󰌗 KVConnector
    󰌗 ActiveKVConnector

  subtypes (ty) in base.py
  󰌗 KVConnectorBase_V1
    󰌗 DummyKVConnector
    󰌗 _LegacyConnector
    󰌗 MockConnector
    󰌗 MockHMAConnector
    󰌗 DummyKVConnector
    󰌗 MockKVConnector
    󰌗 DecodeBenchConnector
    󰌗 ExampleConnector
    󰌗 ExampleHiddenStatesConnector
    󰌗 FlexKVConnectorV1
    󰌗 HF3FSKVConnector
    󰌗 HiSparseConnector
    󰌗 LMCacheConnectorV1
    󰌗 LMCacheMPConnectorUpstream
    󰌗 MooncakeConnector
    󰌗 MooncakeStoreConnector
    󰌗 MoRIIOConnector
    󰌗 MultiConnector
    󰌗 NixlBaseConnector
    󰌗 OffloadingConnector
    󰌗 SimpleCPUOffloadConnector
    󰌗 UMBPStoreConnector
```

配置文件：
```python
@config
class KVTransferConfig:
```

### MooncakeConnector / MooncakeStoreConnector


| Connector | 作用 | 数据路径 |
|---|---|---|
| `MooncakeConnector` | PD分离下，Prefill/Decode 之间的 **点对点 KV 传输** | Prefill GPU → Decode GPU |
| `MooncakeStoreConnector` | 通过 `MooncakeDistributedStore` 做 **共享 KV Store / 前缀缓存** | vLLM 实例 ↔ Mooncake Store |


### MultiConnector 组合多个Connector

组合方式包括但不限于
- HiSparseConnector + MooncakeConnector
- MooncakeConnector + MooncakeStoreConnector

### KVConnectorRole: SCHEDULER和WORKER

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:139-144
```python
class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1
```

通信的类

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:147-160
```python
class KVConnectorHandshakeMetadata(ABC):  # noqa: B024
    """Metadata used for out of band connector handshake between
    P/D workers. This needs to serializable.
    """

    pass


class KVConnectorMetadata(ABC):  # noqa: B024
    """Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """

    pass
```

## Executor层：UniProcExecutor、WorkerWrapperBase、Worker

## ModelRunner层：GPUModelRunner、NPUModelRunner

NPUModelRunner在vllm-ascend仓

### 模型：Module、DeepseekV32Model、DeepseekV32DecoderLayer、DeepseekV32Attention

## Attention后端: AttentionImplBase基类、FlashInferMLASparseSM120Impl

继承关系
```
  󰌗 FlashInferMLASparseSM120Impl
    󰌗 SparseMLACommonImpl
      󰌗 MLACommonBaseImpl
     │  󰌗 MLAAttentionImpl
     │    󰌗 AttentionImplBase
     │      󰌗 ABC
      󰌗 SharedTopkIndicesBuffer
        󰌗 object
```

## HiSparse

- 调度侧：HiSparseCoordinator、HiSparseSourceManager、HiSparseHotManager、HiSparseResidentManager
- Connector：HiSparseConnector、HiSparseConnectorScheduler、HiSparseConnectorWorker
- Runtime：HiSparseRuntime、HiSparseCacheHandle、HiSparseIndexGroup

### HiSparseConnector

```vllmclass
# /home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py:221
# vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector.HiSparseConnector
class HiSparseConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str: ...
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig) -> None:
        self.connector_scheduler: HiSparseConnectorScheduler | None = None
        self.connector_worker: HiSparseConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = HiSparseConnectorScheduler(...)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = HiSparseConnectorWorker(...)
```

### HiSparseConnectorScheduler

```python
# /home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py:95
# vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector.HiSparseConnectorScheduler
class HiSparseConnectorScheduler:
    def __init__(self, *, async_speculative: bool, draft_kv_lookahead: int=0) -> None:
        self.coordinator: HiSparseCoordinator | None = None
        self.async_speculative = async_speculative
        self.draft_kv_lookahead = draft_kv_lookahead
        self.requests: dict[str, Request] = {}
    def bind_coordinator(self, coordinator: HiSparseCoordinator) -> None: ...
    def get_kv_connector_stats(self) -> HiSparseKVConnectorStats | None: ...
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> HiSparseConnectorMetadata: ...
    def update_connector_output(self, connector_output: KVConnectorOutput) -> None: ...
```

## 输出：OutputProcessor、RequestOutput

## 调用路径树
```vllmclass
调用路径树（已保存事件的路径并集；按首次出现排列）
trace_started_at=2026-10-26T13:19:47+00:00
trace_command=python trace_python/call_stack_analysis --config trace_python/call_stack_analysis/config/hisparse.py --output-prefix /tmp/hisparse
trace_cwd=/home/server/ai-infra-dev
运行时长：51.383秒
samples=经过节点的已保存栈数，不是函数调用次数；events=事件序号范围，不表示连续命中。
🎯[n]=该路径首次作为采集终点的事件序号；caller=调用者源码行号。循环折叠，递归按深度保留。
status=ok events=259

└─_run_module_as_main (<frozen runpy>) samples=259 events=1..259
  └─_run_code (<frozen runpy>) [caller:198] samples=259 events=1..259
    └─<module> (/home/server/ai-infra-dev/trace_python/call_stack_analysis/__main__.py) [caller:88] samples=259 events=1..259
      └─main (/home/server/ai-infra-dev/trace_python/call_stack_analysis/__main__.py) [caller:341] samples=259 events=1..259
        └─run_trace (/home/server/ai-infra-dev/trace_python/call_stack_analysis/__main__.py) [caller:337] samples=259 events=1..259
          └─run (/home/server/ai-infra-dev/trace_python/call_stack_analysis/config/hisparse.py) [caller:311] samples=259 events=1..259
            ├─LLMEngine.from_engine_args (vllm/v1/engine/llm_engine.py) [caller:35] samples=18 events=1..18
            │ └─LLMEngine.__init__ (vllm/v1/engine/llm_engine.py) [caller:191] samples=18 events=1..18
            │   └─EngineCoreClient.make_client (vllm/v1/engine/core_client.py) [caller:116] samples=18 events=1..18
            │     └─InprocClient.__init__ (vllm/v1/engine/core_client.py) [caller:136] samples=18 events=1..18
            │       └─EngineCore.__init__ (vllm/v1/engine/core.py) [caller:386] samples=18 events=1..18
            │         ├─instrument_otel.<locals>.sync_wrapper (vllm/tracing/otel.py) [caller:147] samples=4 events=1..4
            │         │ └─Executor.__init__ (vllm/v1/executor/abstract.py) [caller:175] samples=4 events=1..4
            │         │   └─UniProcExecutor._init_executor (vllm/v1/executor/uniproc_executor.py) [caller:120] samples=4 events=1..4
            │         │     └─Worker.load_model (vllm/v1/worker/gpu_worker.py) [caller:74] samples=4 events=1..4
            │         │       └─GPUModelRunner.load_model (vllm/v1/worker/gpu/model_runner.py) [caller:601] samples=4 events=1..4
            │         │         └─instrument_otel.<locals>.sync_wrapper (vllm/tracing/otel.py) [caller:403] samples=4 events=1..4
            │         │           └─BaseModelLoader.load_model (vllm/model_executor/model_loader/base_loader.py) [caller:175] samples=4 events=1..4
            │         │             └─BaseModelLoader.create_model (vllm/model_executor/model_loader/base_loader.py) [caller:76] samples=4 events=1..4
            │         │               └─instrument_otel.<locals>.sync_wrapper (vllm/tracing/otel.py) [caller:53] samples=4 events=1..4
            │         │                 └─initialize_model (vllm/model_executor/model_loader/utils.py) [caller:175] samples=4 events=1..4
            │         │                   └─DeepseekV32ForCausalLM.__init__ (vllm/models/deepseek_v32/nvidia/model.py) [caller:101] samples=4 events=1..4
            │         │                     └─DeepseekV2ForCausalLM.__init__ (vllm/model_executor/models/deepseek_v2.py) [caller:441] samples=4 events=1..4
            │         │                       └─DeepseekV32Model.__init__ (vllm/models/deepseek_v32/nvidia/model.py) [caller:1914] samples=4 events=1..4
            │         │                         └─make_layers (vllm/model_executor/models/utils.py) [caller:216] samples=4 events=1..4
            │         │                           └─NoopOffloader.wrap_modules (vllm/model_executor/offloader/base.py) [caller:913] samples=4 events=1..4
            │         │                             └─make_layers.<locals>.<genexpr> (vllm/model_executor/models/utils.py) [caller:120] samples=4 events=1..4
            │         │                               └─DeepseekV32Model.__init__.<locals>.<lambda> (vllm/models/deepseek_v32/nvidia/model.py) [caller:914] samples=4 events=1..4
            │         │                                 └─DeepseekV32DecoderLayer.__init__ (vllm/models/deepseek_v32/nvidia/model.py) [caller:218] samples=4 events=1..4
            │         │                                   └─DeepseekV32Attention.__init__ (vllm/models/deepseek_v32/attention.py) [caller:83] samples=4 events=1..4
            │         │                                     └─MLAAttention.__init__ (vllm/model_executor/layers/attention/mla_attention.py) [caller:202] samples=4 events=1..4
            │         │                                       └─FlashInferMLASparseSM120Impl.__init__ 🎯[1] (vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py) [caller:576] samples=4 events=1..4
            │         ├─instrument_otel.<locals>.sync_wrapper (vllm/tracing/otel.py) [caller:160] samples=13 events=5..17
            │         │ └─EngineCore._initialize_kv_caches (vllm/v1/engine/core.py) [caller:175] samples=13 events=5..17
            │         │   ├─Executor.determine_available_memory (vllm/v1/executor/abstract.py) [caller:300] samples=1 events=5..5
            │         │   │ └─UniProcExecutor.collective_rpc (vllm/v1/executor/uniproc_executor.py) [caller:157] samples=1 events=5..5
            │         │   │   └─run_method (vllm/v1/serial_utils.py) [caller:103] samples=1 events=5..5
            │         │   │     └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:508] samples=1 events=5..5
            │         │   │       └─Worker.determine_available_memory (vllm/v1/worker/gpu_worker.py) [caller:124] samples=1 events=5..5
            │         │   │         └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:700] samples=1 events=5..5
            │         │   │           └─GPUModelRunner.profile_run (vllm/v1/worker/gpu/model_runner.py) [caller:124] samples=1 events=5..5
            │         │   │             └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:1027] samples=1 events=5..5
            │         │   │               └─step_eplb_after.<locals>.decorator.<locals>.wrapper (vllm/v1/worker/gpu/eplb_utils.py) [caller:124] samples=1 events=5..5
            │         │   │                 └─GPUModelRunner._dummy_run (vllm/v1/worker/gpu/model_runner.py) [caller:57] samples=1 events=5..5
            │         │   │                   └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:923] samples=1 events=5..5
            │         │   │                     └─GPUModelRunner.execute_model 🎯[5] (vllm/v1/worker/gpu/model_runner.py) [caller:124] samples=1 events=5..5
            │         │   ├─Executor.initialize_from_config (vllm/v1/executor/abstract.py) [caller:362] samples=2 events=6..7
            │         │   │ └─UniProcExecutor.collective_rpc (vllm/v1/executor/uniproc_executor.py) [caller:131] samples=2 events=6..7
            │         │   │   └─run_method (vllm/v1/serial_utils.py) [caller:103] samples=2 events=6..7
            │         │   │     └─WorkerWrapperBase.initialize_from_config (vllm/v1/worker/worker_base.py) [caller:508] samples=2 events=6..7
            │         │   │       └─instrument_otel.<locals>.sync_wrapper (vllm/tracing/otel.py) [caller:354] samples=2 events=6..7
            │         │   │         └─Worker.initialize_from_config (vllm/v1/worker/gpu_worker.py) [caller:175] samples=2 events=6..7
            │         │   │           ├─ensure_kv_transfer_initialized (vllm/distributed/kv_transfer/kv_transfer_state.py) [caller:887] samples=1 events=6..6
            │         │   │           │ └─KVConnectorFactory.create_connector (vllm/distributed/kv_transfer/kv_connector/factory.py) [caller:88] samples=1 events=6..6
            │         │   │           │   └─HiSparseConnector.__init__ 🎯[6] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:75] samples=1 events=6..6
            │         │   │           └─GPUModelRunner.initialize_kv_cache (vllm/v1/worker/gpu/model_runner.py) [caller:907] samples=1 events=7..7
            │         │   │             └─get_kv_connector (vllm/v1/worker/gpu/kv_connector.py) [caller:822] samples=1 events=7..7
            │         │   │               └─ActiveKVConnector.__init__ 🎯[7] (vllm/v1/worker/gpu/kv_connector.py) [caller:194] samples=1 events=7..7
            │         │   └─Executor.compile_or_warm_up_model (vllm/v1/executor/abstract.py) [caller:379] samples=10 events=8..17
            │         │     └─UniProcExecutor.collective_rpc (vllm/v1/executor/uniproc_executor.py) [caller:135] samples=10 events=8..17
            │         │       └─run_method (vllm/v1/serial_utils.py) [caller:103] samples=10 events=8..17
            │         │         └─instrument_otel.<locals>.sync_wrapper (vllm/tracing/otel.py) [caller:508] samples=10 events=8..17
            │         │           └─Worker.compile_or_warm_up_model (vllm/v1/worker/gpu_worker.py) [caller:175] samples=10 events=8..17
            │         │             └─Worker._compile_or_warm_up_model (vllm/v1/worker/gpu_worker.py) [caller:938] samples=10 events=8..17
            │         │               └─kernel_warmup (vllm/model_executor/warmup/kernel_warmup.py) [caller:982] samples=10 events=8..17
            │         │                 └─flashinfer_sparse_mla_decode_autotune_warmup (vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py) [caller:237] samples=10 events=8..17
            │         │                   └─_flashinfer_sparse_mla_decode_autotune (vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py) [caller:314] samples=10 events=8..17
            │         │                     └─_run_flashinfer_sparse_mla_decode_autotune (vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py) [caller:290] samples=10 events=8..17
            │         │                       └─_run_sparse_mla_decode_autotune_v2 (vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py) [caller:160] samples=10 events=8..17
            │         │                         └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:274] samples=10 events=8..17
            │         │                           └─step_eplb_after.<locals>.decorator.<locals>.wrapper (vllm/v1/worker/gpu/eplb_utils.py) [caller:124] samples=10 events=8..17
            │         │                             └─GPUModelRunner._dummy_run (vllm/v1/worker/gpu/model_runner.py) [caller:55] samples=10 events=8..17
            │         │                               └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:923] samples=10 events=8..17
            │         │                                 └─GPUModelRunner.execute_model 🎯[8] (vllm/v1/worker/gpu/model_runner.py) [caller:124] samples=10 events=8..17
            │         │                                   ├─DefaultModelState.prepare_attn (vllm/v1/worker/gpu/model_states/default.py) [caller:1954] samples=1 events=9..9
            │         │                                   │ └─build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py) [caller:222] samples=1 events=9..9
            │         │                                   │   └─AttentionGroup.build_metadata (vllm/v1/worker/utils.py) [caller:574] samples=1 events=9..9
            │         │                                   │     └─SparseMLACommonMetadataBuilder.build 🎯[9] (vllm/model_executor/layers/attention/sparse_mla_attention.py) [caller:348] samples=1 events=9..9
            │         │                                   └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:2084] samples=8 events=10..17
            │         │                                     └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=8 events=10..17
            │         │                                       └─DeepseekV2ForCausalLM.forward (vllm/model_executor/models/deepseek_v2.py) [caller:1794] samples=8 events=10..17
            │         │                                         └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1972] samples=8 events=10..17
            │         │                                           └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=8 events=10..17
            │         │                                             └─DeepseekV32Model.forward (vllm/models/deepseek_v32/nvidia/model.py) [caller:1794] samples=8 events=10..17
            │         │                                               └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:293] samples=8 events=10..17
            │         │                                                 └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=8 events=10..17
            │         │                                                   └─DeepseekV32DecoderLayer.forward (vllm/models/deepseek_v32/nvidia/model.py) [caller:1794] samples=8 events=10..17
            │         │                                                     └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:151] samples=8 events=10..17
            │         │                                                       └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=8 events=10..17
            │         │                                                         └─DeepseekV32Attention.forward (vllm/models/deepseek_v32/attention.py) [caller:1794] samples=8 events=10..17
            │         │                                                           └─eager_break_during_capture.<locals>.wrapper (vllm/compilation/breakable_cudagraph.py) [caller:450] samples=8 events=10..17
            │         │                                                             └─DeepseekV32Attention._sparse_indexer_and_attn (vllm/models/deepseek_v32/attention.py) [caller:103] samples=8 events=10..17
            │         │                                                               └─FlashInferMLASparseSM120Impl.forward_mqa 🎯[10] (vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py) [caller:597] samples=8 events=10..17
            │         │                                                                 └─FlashInferMLASparseSM120Impl._run_mqa_kernel 🎯[11] (vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py) [caller:141] samples=4 events=11..17
            │         └─AsyncScheduler.__init__ (vllm/v1/core/sched/async_scheduler.py) [caller:177] samples=1 events=18..18
            │           └─Scheduler.__init__ (vllm/v1/core/sched/scheduler.py) [caller:16] samples=1 events=18..18
            │             └─KVConnectorFactory.create_connector (vllm/distributed/kv_transfer/kv_connector/factory.py) [caller:177] samples=1 events=18..18
            │               └─HiSparseConnector.__init__ 🎯[18] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:75] samples=1 events=18..18
            ├─LLMEngine.add_request 🎯[19] (vllm/v1/engine/llm_engine.py) [caller:36] samples=2 events=19..20
            │ └─InprocClient.add_request (vllm/v1/engine/core_client.py) [caller:299] samples=1 events=20..20
            │   └─EngineCore.add_request (vllm/v1/engine/core.py) [caller:408] samples=1 events=20..20
            │     └─Scheduler.add_request 🎯[20] (vllm/v1/core/sched/scheduler.py) [caller:548] samples=1 events=20..20
            └─LLMEngine.step (vllm/v1/engine/llm_engine.py) [caller:41] samples=239 events=21..259
              ├─InprocClient.get_output (vllm/v1/engine/core_client.py) [caller:327] samples=237 events=21..259
              │ └─EngineCore.step_with_batch_queue (vllm/v1/engine/core.py) [caller:399] samples=237 events=21..259
              │   ├─Scheduler.schedule 🎯[21] (vllm/v1/core/sched/scheduler.py) [caller:725] samples=47 events=21..251
              │   │ ├─Scheduler._get_local_prefix_cache_hit (vllm/v1/core/sched/scheduler.py) [caller:1100] samples=1 events=22..22
              │   │ │ └─KVCacheManager.get_computed_blocks 🎯[22] (vllm/v1/core/kv_cache_manager.py) [caller:652] samples=1 events=22..22
              │   │ ├─KVCacheManager.allocate_slots 🎯[23] (vllm/v1/core/kv_cache_manager.py) [caller:1408] samples=3 events=23..25
              │   │ │ └─KVCacheCoordinator.allocate_new_blocks (vllm/v1/core/kv_cache_coordinator.py) [caller:602] samples=2 events=24..25
              │   │ │   └─KVCacheCoordinator.allocate_new_blocks.<locals>.<genexpr> (vllm/v1/core/kv_cache_coordinator.py) [caller:335] samples=2 events=24..25
              │   │ │     └─SingleTypeKVCacheManager.allocate_new_blocks (vllm/v1/core/single_type_kv_cache_manager.py) [caller:336] samples=2 events=24..25
              │   │ │       └─BlockPool.get_new_blocks 🎯[24] (vllm/v1/core/block_pool.py) [caller:406] samples=2 events=24..25
              │   │ ├─Scheduler._build_kv_connector_meta (vllm/v1/core/sched/scheduler.py) [caller:1712] samples=19 events=26..251
              │   │ │ └─HiSparseConnector.build_connector_meta (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:1744] samples=19 events=26..251
              │   │ │   └─HiSparseConnectorScheduler.build_connector_meta 🎯[26] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:395] samples=19 events=26..251
              │   │ │     ├─HiSparseCoordinator.advance_scheduled (vllm/v1/hisparse/coordinator.py) [caller:152] samples=4 events=27..51
              │   │ │     │ └─HiSparseCoordinator.plan_prefix_materialization 🎯[27] (vllm/v1/hisparse/coordinator.py) [caller:454] samples=4 events=27..51
              │   │ │     │   └─HiSparseCoordinator._plan_page_transfer 🎯[28] (vllm/v1/hisparse/coordinator.py) [caller:486] samples=2 events=28..29
              │   │ │     └─HiSparseCoordinator.build_row_mirrors 🎯[30] (vllm/v1/hisparse/coordinator.py) [caller:179] samples=2 events=30..52
              │   │ └─KVCacheManager.allocate_slots 🎯[49] (vllm/v1/core/kv_cache_manager.py) [caller:892] samples=11 events=49..235
              │   ├─UniProcExecutor.execute_model (vllm/v1/executor/uniproc_executor.py) [caller:727] samples=160 events=31..253
              │   │ └─UniProcExecutor.collective_rpc (vllm/v1/executor/uniproc_executor.py) [caller:122] samples=160 events=31..253
              │   │   └─run_method (vllm/v1/serial_utils.py) [caller:109] samples=160 events=31..253
              │   │     └─WorkerWrapperBase.execute_model (vllm/v1/worker/worker_base.py) [caller:508] samples=160 events=31..253
              │   │       └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:383] samples=160 events=31..253
              │   │         └─Worker.execute_model (vllm/v1/worker/gpu_worker.py) [caller:124] samples=160 events=31..253
              │   │           └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:1360] samples=160 events=31..253
              │   │             └─GPUModelRunner.execute_model (vllm/v1/worker/gpu/model_runner.py) [caller:124] samples=160 events=31..253
              │   │               ├─DefaultModelState.prepare_attn (vllm/v1/worker/gpu/model_states/default.py) [caller:1954] samples=12 events=31..237
              │   │               │ └─build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py) [caller:222] samples=12 events=31..237
              │   │               │   └─AttentionGroup.build_metadata (vllm/v1/worker/utils.py) [caller:574] samples=12 events=31..237
              │   │               │     └─SparseMLACommonMetadataBuilder.build 🎯[31] (vllm/model_executor/layers/attention/sparse_mla_attention.py) [caller:348] samples=12 events=31..237
              │   │               ├─ActiveKVConnector.pre_forward (vllm/v1/worker/gpu/kv_connector.py) [caller:2068] samples=40 events=32..239
              │   │               │ └─ActiveKVConnector._start_load_kv (vllm/v1/worker/gpu/kv_connector.py) [caller:94] samples=40 events=32..239
              │   │               │   └─HiSparseConnector.start_load_kv (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:124] samples=40 events=32..239
              │   │               │     └─HiSparseConnectorWorker.start_step 🎯[32] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:338] samples=40 events=32..239
              │   │               │       ├─HiSparseConnectorWorker._finish_previous_step (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:416] samples=8 events=33..173
              │   │               │       │ └─HiSparseConnectorWorker._submit_transfers 🎯[33] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:945] samples=8 events=33..173
              │   │               │       ├─HiSparseConnectorWorker._restore_pages 🎯[34] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:438] samples=12 events=34..239
              │   │               │       └─HiSparseConnectorWorker._submit_transfers 🎯[35] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:444] samples=8 events=35..175
              │   │               ├─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:2084] samples=102 events=36..247
              │   │               │ └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=102 events=36..247
              │   │               │   └─DeepseekV2ForCausalLM.forward (vllm/model_executor/models/deepseek_v2.py) [caller:1794] samples=102 events=36..247
              │   │               │     └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1972] samples=102 events=36..247
              │   │               │       └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=102 events=36..247
              │   │               │         └─DeepseekV32Model.forward (vllm/models/deepseek_v32/nvidia/model.py) [caller:1794] samples=102 events=36..247
              │   │               │           └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:293] samples=102 events=36..247
              │   │               │             └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=102 events=36..247
              │   │               │               └─DeepseekV32DecoderLayer.forward (vllm/models/deepseek_v32/nvidia/model.py) [caller:1794] samples=102 events=36..247
              │   │               │                 └─Module._wrapped_call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:151] samples=102 events=36..247
              │   │               │                   └─Module._call_impl (.venv/lib/python3.12/site-packages/torch/nn/modules/module.py) [caller:1783] samples=102 events=36..247
              │   │               │                     └─DeepseekV32Attention.forward (vllm/models/deepseek_v32/attention.py) [caller:1794] samples=102 events=36..247
              │   │               │                       └─eager_break_during_capture.<locals>.wrapper (vllm/compilation/breakable_cudagraph.py) [caller:450] samples=102 events=36..247
              │   │               │                         └─DeepseekV32Attention._sparse_indexer_and_attn (vllm/models/deepseek_v32/attention.py) [caller:103] samples=102 events=36..247
              │   │               │                           └─FlashInferMLASparseSM120Impl.forward_mqa 🎯[36] (vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py) [caller:597] samples=102 events=36..247
              │   │               │                             ├─FlashInferMLASparseSM120Impl._run_mqa_kernel 🎯[37] (vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py) [caller:141] samples=48 events=37..247
              │   │               │                             └─HiSparseMLAIndexGroup.convert_logical_to_physical_topk (vllm/v1/attention/backends/mla/index_group.py) [caller:132] samples=6 events=135..142
              │   │               │                               └─HiSparseCacheHandle.swap_in (vllm/v1/hisparse/runtime.py) [caller:256] samples=6 events=135..142
              │   │               │                                 └─HiSparseRuntime.swap_in 🎯[135] (vllm/v1/hisparse/runtime.py) [caller:1288] samples=6 events=135..142
              │   │               │                                   └─HiSparseRuntime._resolve_and_stage_group (vllm/v1/hisparse/runtime.py) [caller:1112] samples=4 events=136..142
              │   │               │                                     ├─HiSparseRuntime._resolve_residency 🎯[136] (vllm/v1/hisparse/runtime.py) [caller:1052] samples=2 events=136..141
              │   │               │                                     └─HiSparseRuntime._swap_rows 🎯[137] (vllm/v1/hisparse/runtime.py) [caller:1067] samples=2 events=137..142
              │   │               ├─ActiveKVConnector.finish_forward (vllm/v1/worker/gpu/kv_connector.py) [caller:2086] samples=4 events=44..67
              │   │               │ └─HiSparseConnector.finish_forward (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:131] samples=4 events=44..67
              │   │               │   └─HiSparseConnectorWorker.finish_forward 🎯[44] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:287] samples=4 events=44..67
              │   │               │     └─HiSparseConnectorWorker._finish_mirror_phase (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:920] samples=2 events=45..67
              │   │               │       └─HiSparseConnectorWorker._enqueue_host_mirror (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:914] samples=2 events=45..67
              │   │               │         └─HiSparseConnectorWorker._enqueue_row_dma (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:897] samples=2 events=45..67
              │   │               │           └─HiSparseConnectorWorker._submit_dma_descriptors 🎯[45] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:711] samples=2 events=45..67
              │   │               └─ActiveKVConnector.no_forward (vllm/v1/worker/gpu/kv_connector.py) [caller:1781] samples=2 events=252..253
              │   │                 └─ActiveKVConnector.pre_forward (vllm/v1/worker/gpu/kv_connector.py) [caller:172] samples=2 events=252..253
              │   │                   └─ActiveKVConnector._start_load_kv (vllm/v1/worker/gpu/kv_connector.py) [caller:94] samples=2 events=252..253
              │   │                     └─HiSparseConnector.start_load_kv (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:127] samples=2 events=252..253
              │   │                       └─HiSparseConnectorWorker.start_step 🎯[252] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:338] samples=2 events=252..253
              │   │                         └─HiSparseConnectorWorker._restore_pages 🎯[253] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:438] samples=1 events=253..253
              │   ├─UniProcExecutor.sample_tokens (vllm/v1/executor/uniproc_executor.py) [caller:743] samples=2 events=46..68
              │   │ └─UniProcExecutor.collective_rpc (vllm/v1/executor/uniproc_executor.py) [caller:137] samples=2 events=46..68
              │   │   └─run_method (vllm/v1/serial_utils.py) [caller:109] samples=2 events=46..68
              │   │     └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:508] samples=2 events=46..68
              │   │       └─Worker.sample_tokens (vllm/v1/worker/gpu_worker.py) [caller:124] samples=2 events=46..68
              │   │         └─context_decorator.<locals>.decorate_context (.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py) [caller:1330] samples=2 events=46..68
              │   │           └─step_eplb_after.<locals>.decorator.<locals>.wrapper (vllm/v1/worker/gpu/eplb_utils.py) [caller:124] samples=2 events=46..68
              │   │             └─GPUModelRunner.sample_tokens (vllm/v1/worker/gpu/model_runner.py) [caller:57] samples=2 events=46..68
              │   │               └─ActiveKVConnector.post_forward (vllm/v1/worker/gpu/kv_connector.py) [caller:2244] samples=2 events=46..68
              │   │                 └─HiSparseConnector.build_connector_worker_meta (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:156] samples=2 events=46..68
              │   │                   └─HiSparseConnectorWorker.take_transfer_updates 🎯[46] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) [caller:363] samples=2 events=46..68
              │   └─Scheduler.update_from_output 🎯[69] (vllm/v1/core/sched/scheduler.py) [caller:783] samples=28 events=69..259
              │     ├─Scheduler._update_from_kv_xfer_finished (vllm/v1/core/sched/scheduler.py) [caller:2525] samples=12 events=70..259
              │     │ └─HiSparseConnector.update_connector_output (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:3418] samples=12 events=70..259
              │     │   └─HiSparseConnectorScheduler.update_connector_output 🎯[70] (vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:399] samples=12 events=70..259
              │     └─Scheduler._free_request 🎯[255] (vllm/v1/core/sched/scheduler.py) [caller:2419] samples=4 events=255..258
              │       └─Scheduler._free_blocks (vllm/v1/core/sched/scheduler.py) [caller:2938] samples=3 events=256..258
              │         └─Scheduler._free_request_blocks (vllm/v1/core/sched/scheduler.py) [caller:2944] samples=3 events=256..258
              │           └─KVCacheManager.free 🎯[256] (vllm/v1/core/kv_cache_manager.py) [caller:2967] samples=3 events=256..258
              │             └─KVCacheCoordinator.free (vllm/v1/core/kv_cache_coordinator.py) [caller:636] samples=2 events=257..258
              │               └─SingleTypeKVCacheManager.free (vllm/v1/core/single_type_kv_cache_manager.py) [caller:408] samples=2 events=257..258
              │                 ├─HiSparseSourceManager.pop_blocks_for_free (vllm/v1/core/single_type_kv_cache_manager.py) [caller:598] samples=1 events=257..257
              │                 │ └─HiSparseCoordinator.free 🎯[257] (vllm/v1/hisparse/coordinator.py) [caller:2345] samples=1 events=257..257
              │                 └─HiSparseResidentManager.pop_blocks_for_free (vllm/v1/core/single_type_kv_cache_manager.py) [caller:598] samples=1 events=258..258
              │                   └─HiSparseCoordinator.free 🎯[258] (vllm/v1/hisparse/coordinator.py) [caller:2550] samples=1 events=258..258
              └─OutputProcessor.process_outputs 🎯[47] (vllm/v1/engine/output_processor.py) [caller:334] samples=2 events=47..71

目标完整命中数（包含超过 limit、未保存调用栈的命中）：
       4  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:FlashInferMLASparseSM120Impl.__init__
      16  vllm/v1/worker/gpu/model_runner.py:GPUModelRunner.execute_model
       2  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py:HiSparseConnector.__init__
       1  vllm/v1/worker/gpu/kv_connector.py:ActiveKVConnector.__init__
      13  vllm/model_executor/layers/attention/sparse_mla_attention.py:SparseMLACommonMetadataBuilder.build
      52  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:FlashInferMLASparseSM120Impl.forward_mqa
      52  vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:FlashInferMLASparseSM120Impl._run_mqa_kernel
       1  vllm/v1/engine/llm_engine.py:LLMEngine.add_request
       1  vllm/v1/core/sched/scheduler.py:Scheduler.add_request
      13  vllm/v1/core/sched/scheduler.py:Scheduler.schedule
       1  vllm/v1/core/kv_cache_manager.py:KVCacheManager.get_computed_blocks
      12  vllm/v1/core/kv_cache_manager.py:KVCacheManager.allocate_slots
      40  vllm/v1/core/block_pool.py:BlockPool.get_new_blocks
      13  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py:HiSparseConnectorScheduler.build_connector_meta
      12  vllm/v1/hisparse/coordinator.py:HiSparseCoordinator.plan_prefix_materialization
      36  vllm/v1/hisparse/coordinator.py:HiSparseCoordinator._plan_page_transfer
      12  vllm/v1/hisparse/coordinator.py:HiSparseCoordinator.build_row_mirrors
      13  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py:HiSparseConnectorWorker.start_step
      26  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py:HiSparseConnectorWorker._submit_transfers
      13  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py:HiSparseConnectorWorker._restore_pages
      13  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py:HiSparseConnectorWorker.finish_forward
      12  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py:HiSparseConnectorWorker._submit_dma_descriptors
      13  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py:HiSparseConnectorWorker.take_transfer_updates
      13  vllm/v1/engine/output_processor.py:OutputProcessor.process_outputs
      12  vllm/v1/core/sched/scheduler.py:Scheduler.update_from_output
      12  vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py:HiSparseConnectorScheduler.update_connector_output
      28  vllm/v1/hisparse/runtime.py:HiSparseRuntime.swap_in
      28  vllm/v1/hisparse/runtime.py:HiSparseRuntime._resolve_residency
      28  vllm/v1/hisparse/runtime.py:HiSparseRuntime._swap_rows
       1  vllm/v1/core/sched/scheduler.py:Scheduler._free_request
       1  vllm/v1/core/kv_cache_manager.py:KVCacheManager.free
       5  vllm/v1/hisparse/coordinator.py:HiSparseCoordinator.free
```

## Object-Flow Call Tree

对Request对象的追踪
```vllmclass
对象流调用树（观察路径并集，不是连续执行路径）
samples=已保存事件栈数；同路径同源码行聚合，kind=N 为该类事件数；编号范围不保证连续；历史载体不保证当前持有目标。
events=873 incomplete=False
├─LLMEngine.add_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/llm_engine.py) samples=14 events=1..14
│ └─InprocClient.add_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/core_client.py) [caller:299] samples=14 events=1..14
│   ├─EngineCore.preprocess_add_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/core.py) [caller:405] samples=5 events=1..5
│   │ ├─req = Request.from_engine_core_request(request, self.request_block_hasher) 🚥[1] :1128 capture=1 objects=O1 events=1..1
│   │ ├─Request.use_structured_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1129] samples=1 events=2..2
│   │ │ └─if req.use_structured_output: 🚩[2] :1129 argument=1 objects=O1 events=2..2
│   │ └─return req, request.current_wave 🚩[3] :1136 observed_write=1 return=1 historical_carrier_use=1 objects=O1,O2 events=3..5
│   ├─req, request_wave = self.engine_core.preprocess_add_request(request) 🚩[6] :405 bind=1 objects=O1 events=6..6
│   └─EngineCore.add_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/core.py) [caller:406] samples=8 events=7..14
│     ├─self.engine_core.add_request(req, request_wave) 🚩[7] :406 argument=1 objects=O1 events=7..7
│     └─Scheduler.add_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:548] samples=7 events=8..14
│       ├─self.scheduler.add_request(request) 🚩[8] :548 argument=1 objects=O1 events=8..8
│       ├─Scheduler._enqueue_waiting_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2733] samples=3 events=9..11
│       │ ├─self._enqueue_waiting_request(request) 🚩[9] :2733 argument=1 objects=O1 events=9..9
│       │ ├─Scheduler._holds_kv_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2559] samples=1 events=10..10
│       │ │ └─if self._holds_kv_blocks(request): 🚩[10] :2559 argument=1 objects=O1 events=10..10
│       │ └─FCFSRequestQueue.add_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/request_queue.py) [caller:2562] samples=1 events=11..11
│       │   └─self.waiting.add_request(request) 🚩[11] :2562 argument=1 objects=O1 events=11..11
│       ├─self.requests[request.request_id] = request 🚩[12] :2734 observed_write=1 objects=O1 events=12..12
│       ├─KVConnectorBase_V1.on_new_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py) [caller:2740] samples=1 events=13..13
│       │ └─self.connector.on_new_request(request) 🚩[13] :2740 argument=1 objects=O1 events=13..13
│       └─Request.record_event (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2747] samples=1 events=14..14
│         └─request.record_event(EngineCoreEventType.QUEUED) 🚩[14] :2747 argument=1 objects=O1 events=14..14
└─LLMEngine.step (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/llm_engine.py) samples=859 events=15..873
  └─InprocClient.get_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/core_client.py) [caller:327] samples=859 events=15..873
    └─EngineCore.step_with_batch_queue (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/engine/core.py) [caller:397] samples=859 events=15..873
      ├─Scheduler.schedule (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:725] samples=493 events=15..812
      │ ├─FCFSRequestQueue.peek_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/request_queue.py) [caller:983] samples=1 events=15..15
      │ │ └─return self[0] 🚩[15] :90 return=1 objects=O1 events=15..15
      │ ├─request = request_queue.peek_request() 🚩[16] :983 bind=1 objects=O1 events=16..16
      │ ├─Scheduler._handle_blocked_waiting_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:986] samples=1 events=17..17
      │ │ └─ready_to_schedule = self._handle_blocked_waiting_request(request) 🚩[17] :986 argument=1 objects=O1 events=17..17
      │ ├─Scheduler._get_local_prefix_cache_hit (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1024] samples=4 events=18..21
      │ │ ├─( 🚩[18] :1024 argument=1 objects=O1 events=18..18
      │ │ │     new_computed_blocks,
      │ │ │     num_new_local_computed_tokens,
      │ │ │     request.shared_prefix_boundary,
      │ │ │     hit_diverged,
      │ │ │ ) = self._get_local_prefix_cache_hit(request)
      │ │ └─KVCacheManager.get_computed_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:597] samples=3 events=19..21
      │ │   ├─blocks, num_local, shared_prefix_boundary = ( 🚩[19] :597 argument=1 objects=O1 events=19..19
      │ │   │     self.kv_cache_manager.get_computed_blocks(request)
      │ │   │ )
      │ │   ├─KVCacheManager.prefix_cache_lookup_enabled (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:286] samples=1 events=20..20
      │ │   │ └─if not self.prefix_cache_lookup_enabled(request): 🚩[20] :286 argument=1 objects=O1 events=20..20
      │ │   └─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:295] samples=1 events=21..21
      │ │     └─max_cache_hit_length = request.num_tokens - 1 🚩[21] :295 argument=1 objects=O1 events=21..21
      │ ├─HiSparseConnector.get_num_new_matched_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:1036] samples=1 events=22..22
      │ │ └─ext_tokens, load_kv_async = ( 🚩[22] :1036 argument=1 objects=O1 events=22..22
      │ │       self.connector.get_num_new_matched_tokens(
      │ │           request, block_aligned_local
      │ │       )
      │ │   )
      │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1097] samples=1 events=23..23
      │ │ └─connector_prefix_cache_queries = ( 🚩[23] :1097 argument=1 objects=O1 events=23..23
      │ │       request.num_tokens - num_new_local_computed_tokens
      │ │   )
      │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1105] samples=1 events=24..24
      │ │ └─assert num_computed_tokens <= request.num_tokens 🚩[24] :1105 argument=1 objects=O1 events=24..24
      │ ├─Scheduler._ec_transfer_pending (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1132] samples=1 events=25..25
      │ │ └─if self._ec_transfer_pending(request, ec_computed_tokens): 🚩[25] :1132 argument=1 objects=O1 events=25..25
      │ ├─Scheduler._set_kv_fetch_stage (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1144] samples=1 events=26..26
      │ │ └─self._set_kv_fetch_stage( 🚩[26] :1144 argument=1 objects=O1 events=26..26
      │ │       request, KV_FETCH_WAITING_TO_START if load_kv_async else None
      │ │   )
      │ ├─Scheduler._mark_prefix_replay (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1183] samples=1 events=27..27
      │ │ └─num_replay_tokens = self._mark_prefix_replay( 🚩[27] :1183 argument=1 objects=O1 events=27..27
      │ │       request, num_computed_tokens, window_loaded
      │ │   )
      │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1191] samples=1 events=28..28
      │ │ └─num_new_tokens = request.num_tokens - num_computed_tokens 🚩[28] :1191 argument=1 objects=O1 events=28..28
      │ ├─Request.has_encoder_inputs (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1258] samples=2 events=29..30
      │ │ ├─if request.has_encoder_inputs: 🚩[29] :1258 argument=1 objects=O1 events=29..29
      │ │ └─Request.num_encoder_inputs (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:312] samples=1 events=30..30
      │ │   └─return self.num_encoder_inputs > 0 🚩[30] :312 argument=1 objects=O1 events=30..30
      │ ├─Scheduler._reserve_prefill_lookahead (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1275] samples=2 events=31..32
      │ │ ├─num_new_tokens = self._reserve_prefill_lookahead( 🚩[31] :1275 argument=1 objects=O1 events=31..31
      │ │ │     request, num_computed_tokens, num_new_tokens
      │ │ │ )
      │ │ └─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:617] samples=1 events=32..32
      │ │   └─remaining = request.num_tokens - num_computed_tokens - num_new_tokens 🚩[32] :617 argument=1 objects=O1 events=32..32
      │ ├─KVCacheManager.allocate_slots (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:1320] samples=22 events=33..54
      │ │ ├─new_blocks = self.kv_cache_manager.allocate_slots( 🚩[33] :1320 argument=1 objects=O1 events=33..33
      │ │ │     request,
      │ │ │     num_tokens_past_hit,
      │ │ │     num_new_computed_tokens=num_new_local_computed_tokens,
      │ │ │     new_computed_blocks=new_computed_blocks,
      │ │ │     num_lookahead_tokens=effective_lookahead_tokens,
      │ │ │     num_external_computed_tokens=num_external_computed_tokens,
      │ │ │     delay_cache_blocks=load_kv_async,
      │ │ │     skip_zeroing_group_ids=(
      │ │ │         self.connector.get_loaded_kv_cache_group_ids(request)
      │ │ │         if load_kv_async and self.connector is not None
      │ │ │         else ()
      │ │ │     ),
      │ │ │     num_encoder_tokens=num_encoder_tokens,
      │ │ │     full_sequence_must_fit=self.scheduler_reserve_full_isl,
      │ │ │     reserved_blocks=reserved_blocks,
      │ │ │     has_scheduled_reqs=bool(self.running),
      │ │ │ )
      │ │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:520] samples=1 events=34..34
      │ │ │ └─prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1) 🚩[34] :520 argument=1 objects=O1 events=34..34
      │ │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:524] samples=1 events=35..35
      │ │ │ └─full_num_tokens = min(request.num_tokens, self.max_model_len) 🚩[35] :524 argument=1 objects=O1 events=35..35
      │ │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:614] samples=1 events=36..36
      │ │ │ └─num_tokens_to_cache = min( 🚩[36] :614 argument=1 objects=O1 events=36..36
      │ │ │       total_computed_tokens + num_new_tokens,
      │ │ │       request.num_tokens,
      │ │ │   )
      │ │ └─KVCacheCoordinator.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py) [caller:616] samples=18 events=37..54
      │ │   ├─self.coordinator.cache_blocks(request, num_tokens_to_cache) 🚩[37] :616 argument=1 objects=O1 events=37..37
      │ │   ├─HiSparseSourceManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=6 events=38..43
      │ │   │ ├─manager.cache_blocks( 🚩[38] :372 argument=1 objects=O1 events=38..38
      │ │   │ │     request,
      │ │   │ │     self.get_num_cacheable_tokens(num_computed_tokens, group_id),
      │ │   │ │     retention_interval=self.retention_interval,
      │ │   │ │     replay_boundaries=boundaries,
      │ │   │ │ )
      │ │   │ └─HiSparseCoordinator.publish_when_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:2317] samples=5 events=39..43
      │ │   │   ├─self.coordinator.publish_when_ready( 🚩[39] :2317 argument=1 objects=O1 events=39..39
      │ │   │   │     request,
      │ │   │   │     num_tokens,
      │ │   │   │     retention_interval,
      │ │   │   │     replay_boundaries=replay_boundaries,
      │ │   │   │ )
      │ │   │   └─HiSparseCoordinator._publish_host_blocks_if_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:509] samples=4 events=40..43
      │ │   │     ├─self.host_manager.publish_blocks( 🚩[40] :530 read_attribute=1 objects=O1 events=40..40
      │ │   │     │     publication.request,
      │ │   │     │     num_tokens,
      │ │   │     │     retention_interval=publication.retention_interval,
      │ │   │     │     replay_boundaries=publication.replay_boundaries,
      │ │   │     │ )
      │ │   │     └─HiSparseSourceManager.publish_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:529] samples=3 events=41..43
      │ │   │       ├─self.host_manager.publish_blocks( 🚩[41] :529 argument=1 objects=O1 events=41..41
      │ │   │       │     publication.request,
      │ │   │       │     num_tokens,
      │ │   │       │     retention_interval=publication.retention_interval,
      │ │   │       │     replay_boundaries=publication.replay_boundaries,
      │ │   │       │ )
      │ │   │       └─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:2332] samples=2 events=42..43
      │ │   │         ├─super().cache_blocks( 🚩[42] :2332 argument=1 objects=O1 events=42..42
      │ │   │         │     request,
      │ │   │         │     num_tokens,
      │ │   │         │     retention_interval=retention_interval,
      │ │   │         │     replay_boundaries=replay_boundaries,
      │ │   │         │ )
      │ │   │         └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=1 events=43..43
      │ │   │           └─super().cache_blocks( 🚩[43] :847 argument=1 objects=O1 events=43..43
      │ │   │                 request,
      │ │   │                 num_tokens,
      │ │   │                 retention_interval=retention_interval,
      │ │   │                 replay_boundaries=replay_boundaries,
      │ │   │             )
      │ │   ├─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=3 events=44..46
      │ │   │ ├─manager.cache_blocks( 🚩[44] :372 argument=1 objects=O1 events=44..44
      │ │   │ │     request,
      │ │   │ │     self.get_num_cacheable_tokens(num_computed_tokens, group_id),
      │ │   │ │     retention_interval=self.retention_interval,
      │ │   │ │     replay_boundaries=boundaries,
      │ │   │ │ )
      │ │   │ └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=2 events=45..46
      │ │   │   ├─super().cache_blocks( 🚩[45] :847 argument=1 objects=O1 events=45..45
      │ │   │   │     request,
      │ │   │   │     num_tokens,
      │ │   │   │     retention_interval=retention_interval,
      │ │   │   │     replay_boundaries=replay_boundaries,
      │ │   │   │ )
      │ │   │   └─BlockPool.cache_full_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/block_pool.py) [caller:531] samples=1 events=46..46
      │ │   │     └─self.block_pool.cache_full_blocks( 🚩[46] :531 argument=1 objects=O1 events=46..46
      │ │   │           request=request,
      │ │   │           blocks=self.req_to_blocks[request.request_id],
      │ │   │           num_cached_blocks=num_cached_blocks,
      │ │   │           num_full_blocks=num_full_blocks,
      │ │   │           block_size=self.block_size,
      │ │   │           kv_cache_group_id=self.kv_cache_group_id,
      │ │   │           block_mask=block_mask,
      │ │   │       )
      │ │   └─_HiSparseAuxiliaryManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=8 events=47..54
      │ │     └─manager.cache_blocks( 🚩[47] :372 argument=8 objects=O1 events=47..54
      │ │           request,
      │ │           self.get_num_cacheable_tokens(num_computed_tokens, group_id),
      │ │           retention_interval=self.retention_interval,
      │ │           replay_boundaries=boundaries,
      │ │       )
      │ ├─HiSparseConnector.update_state_after_alloc (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:1353] samples=2 events=55..56
      │ │ ├─self.connector.update_state_after_alloc( 🚩[55] :1353 argument=1 objects=O1 events=55..55
      │ │ │     request,
      │ │ │     self.kv_cache_manager.get_blocks(request_id),
      │ │ │     num_external_computed_tokens,
      │ │ │ )
      │ │ └─self.connector_scheduler.requests[request.request_id] = request 🚩[56] :389 observed_write=1 objects=O1 events=56..56
      │ ├─KVCacheManager.record_prefix_cache_stats (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:1380] samples=3 events=57..59
      │ │ ├─self.kv_cache_manager.record_prefix_cache_stats( 🚩[57] :1380 argument=1 objects=O1 events=57..57
      │ │ │     request, num_new_local_computed_tokens
      │ │ │ )
      │ │ ├─KVCacheManager.prefix_cache_lookup_enabled (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:255] samples=1 events=58..58
      │ │ │ └─if not self.log_stats or not self.prefix_cache_lookup_enabled(request): 🚩[58] :255 argument=1 objects=O1 events=58..58
      │ │ └─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:259] samples=1 events=59..59
      │ │   └─self.prefix_cache_stats.record( 🚩[59] :259 argument=1 objects=O1 events=59..59
      │ │         num_tokens=request.num_tokens,
      │ │         num_hits=num_hits,
      │ │         preempted=request.num_preemptions > 0,
      │ │     )
      │ ├─FCFSRequestQueue.pop_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/request_queue.py) [caller:1413] samples=1 events=60..60
      │ │ └─return self.popleft() 🚩[60] :84 return=1 objects=O1 events=60..60
      │ ├─request = request_queue.pop_request() 🚩[61] :1413 bind=1 objects=O1 events=61..61
      │ ├─self.running.append(request) 🚩[62] :1415 observed_write=1 objects=O1 events=62..62
      │ ├─Request.record_event (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1420] samples=1 events=63..63
      │ │ └─request.record_event( 🚩[63] :1420 argument=1 objects=O1 events=63..63
      │ │       EngineCoreEventType.SCHEDULED, scheduled_timestamp
      │ │   )
      │ ├─scheduled_new_reqs.append(request) 🚩[64] :1424 observed_write=1 objects=O1 events=64..64
      │ ├─Scheduler._set_kv_fetch_stage (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1440] samples=1 events=65..65
      │ │ └─self._set_kv_fetch_stage(request, None) 🚩[65] :1440 argument=1 objects=O1 events=65..65
      │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1453] samples=1 events=66..66
      │ │ └─prefill_admitted |= num_computed_tokens < request.num_tokens - 1 🚩[66] :1453 argument=1 objects=O1 events=66..66
      │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1455] samples=1 events=67..67
      │ │ └─if num_computed_tokens + num_new_tokens < request.num_tokens: 🚩[67] :1455 argument=1 objects=O1 events=67..67
      │ ├─self._inflight_prefills.add(request) 🚩[68] :1456 observed_write=1 objects=O1 events=68..68
      │ ├─NewRequestData.from_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/output.py) [caller:1509] samples=1 events=69..69
      │ │ └─new_reqs_data = [ 🚩[69] :1509 argument=1 objects=O1 events=69..69
      │ │       NewRequestData.from_request(
      │ │           req,
      │ │           req_to_new_blocks[req.request_id].get_block_ids(),
      │ │           req._all_token_ids,
      │ │           uses_mrope=self.model_uses_mrope,
      │ │       )
      │ │       for req in scheduled_new_reqs
      │ │   ]
      │ ├─AsyncScheduler._update_after_schedule (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/async_scheduler.py) [caller:1642] samples=80 events=70..773
      │ │ ├─Scheduler._update_after_schedule (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:26] samples=48 events=70..770
      │ │ │ ├─request = self.requests[req_id] 🚩[70] :1715 read_subscript=12 bind=12 objects=O1 events=70..768
      │ │ │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1722] samples=12 events=72..769
      │ │ │ │ └─request.is_prefill_chunk = request.num_computed_tokens < ( 🚩[72] :1722 argument=12 objects=O1 events=72..769
      │ │ │ │       request.num_tokens + request.num_output_placeholders
      │ │ │ │   )
      │ │ │ └─Request.use_structured_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1725] samples=12 events=73..770
      │ │ │   └─scheduler_output.has_structured_output_requests |= ( 🚩[73] :1725 argument=12 objects=O1 events=73..770
      │ │ │         request.use_structured_output and not request.is_prefill_chunk
      │ │ │     )
      │ │ ├─request = self.requests[req_id] 🚩[74] :33 read_subscript=12 bind=12 objects=O1 events=74..772
      │ │ └─Request.use_structured_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:38] samples=8 events=257..773
      │ │   └─scheduler_output.pending_structured_output_tokens |= ( 🚩[257] :38 argument=8 objects=O1 events=257..773
      │ │         request.use_structured_output and request.num_output_placeholders > 0
      │ │     )
      │ ├─request = self.running[req_index] 🚩[76] :716 read_subscript=12 bind=12 objects=O1 events=76..812
      │ ├─Request.num_tokens_with_spec (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:760] samples=11 events=78..737
      │ │ └─num_new_tokens = ( 🚩[78] :760 argument=11 objects=O1 events=78..737
      │ │       request.num_tokens_with_spec
      │ │       + request.num_output_placeholders
      │ │       - request.num_computed_tokens
      │ │   )
      │ ├─Request.has_encoder_inputs (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:790] samples=22 events=79..739
      │ │ ├─if request.has_encoder_inputs: 🚩[79] :790 argument=11 objects=O1 events=79..738
      │ │ └─Request.num_encoder_inputs (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:312] samples=11 events=80..739
      │ │   └─return self.num_encoder_inputs > 0 🚩[80] :312 argument=11 objects=O1 events=80..739
      │ ├─Scheduler._reserve_prefill_lookahead (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:807] samples=22 events=81..741
      │ │ ├─num_new_tokens = self._reserve_prefill_lookahead( 🚩[81] :807 argument=11 objects=O1 events=81..740
      │ │ │     request, request.num_computed_tokens, num_new_tokens
      │ │ │ )
      │ │ └─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:617] samples=11 events=82..741
      │ │   └─remaining = request.num_tokens - num_computed_tokens - num_new_tokens 🚩[82] :617 argument=11 objects=O1 events=82..741
      │ ├─KVCacheManager.allocate_slots (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:834] samples=224 events=83..761
      │ │ ├─new_blocks = self.kv_cache_manager.allocate_slots( 🚩[83] :834 argument=11 objects=O1 events=83..742
      │ │ │     request,
      │ │ │     num_new_tokens,
      │ │ │     num_lookahead_tokens=self.num_lookahead_tokens,
      │ │ │ )
      │ │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:520] samples=11 events=84..743
      │ │ │ └─prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1) 🚩[84] :520 argument=11 objects=O1 events=84..743
      │ │ ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:614] samples=11 events=85..744
      │ │ │ └─num_tokens_to_cache = min( 🚩[85] :614 argument=11 objects=O1 events=85..744
      │ │ │       total_computed_tokens + num_new_tokens,
      │ │ │       request.num_tokens,
      │ │ │   )
      │ │ └─KVCacheCoordinator.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py) [caller:616] samples=191 events=86..761
      │ │   ├─self.coordinator.cache_blocks(request, num_tokens_to_cache) 🚩[86] :616 argument=11 objects=O1 events=86..745
      │ │   ├─HiSparseSourceManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=66 events=87..751
      │ │   │ ├─manager.cache_blocks( 🚩[87] :372 argument=11 objects=O1 events=87..746
      │ │   │ │     request,
      │ │   │ │     self.get_num_cacheable_tokens(num_computed_tokens, group_id),
      │ │   │ │     retention_interval=self.retention_interval,
      │ │   │ │     replay_boundaries=boundaries,
      │ │   │ │ )
      │ │   │ └─HiSparseCoordinator.publish_when_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:2317] samples=55 events=88..751
      │ │   │   ├─self.coordinator.publish_when_ready( 🚩[88] :2317 argument=11 objects=O1 events=88..747
      │ │   │   │     request,
      │ │   │   │     num_tokens,
      │ │   │   │     retention_interval,
      │ │   │   │     replay_boundaries=replay_boundaries,
      │ │   │   │ )
      │ │   │   └─HiSparseCoordinator._publish_host_blocks_if_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:509] samples=44 events=89..751
      │ │   │     ├─self.host_manager.publish_blocks( 🚩[89] :530 read_attribute=11 objects=O1 events=89..748
      │ │   │     │     publication.request,
      │ │   │     │     num_tokens,
      │ │   │     │     retention_interval=publication.retention_interval,
      │ │   │     │     replay_boundaries=publication.replay_boundaries,
      │ │   │     │ )
      │ │   │     └─HiSparseSourceManager.publish_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:529] samples=33 events=90..751
      │ │   │       ├─self.host_manager.publish_blocks( 🚩[90] :529 argument=11 objects=O1 events=90..749
      │ │   │       │     publication.request,
      │ │   │       │     num_tokens,
      │ │   │       │     retention_interval=publication.retention_interval,
      │ │   │       │     replay_boundaries=publication.replay_boundaries,
      │ │   │       │ )
      │ │   │       └─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:2332] samples=22 events=91..751
      │ │   │         ├─super().cache_blocks( 🚩[91] :2332 argument=11 objects=O1 events=91..750
      │ │   │         │     request,
      │ │   │         │     num_tokens,
      │ │   │         │     retention_interval=retention_interval,
      │ │   │         │     replay_boundaries=replay_boundaries,
      │ │   │         │ )
      │ │   │         └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=11 events=92..751
      │ │   │           └─super().cache_blocks( 🚩[92] :847 argument=11 objects=O1 events=92..751
      │ │   │                 request,
      │ │   │                 num_tokens,
      │ │   │                 retention_interval=retention_interval,
      │ │   │                 replay_boundaries=replay_boundaries,
      │ │   │             )
      │ │   ├─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=26 events=93..753
      │ │   │ ├─manager.cache_blocks( 🚩[93] :372 argument=11 objects=O1 events=93..752
      │ │   │ │     request,
      │ │   │ │     self.get_num_cacheable_tokens(num_computed_tokens, group_id),
      │ │   │ │     retention_interval=self.retention_interval,
      │ │   │ │     replay_boundaries=boundaries,
      │ │   │ │ )
      │ │   │ └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=15 events=94..753
      │ │   │   ├─super().cache_blocks( 🚩[94] :847 argument=11 objects=O1 events=94..753
      │ │   │   │     request,
      │ │   │   │     num_tokens,
      │ │   │   │     retention_interval=retention_interval,
      │ │   │   │     replay_boundaries=replay_boundaries,
      │ │   │   │ )
      │ │   │   └─BlockPool.cache_full_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/block_pool.py) [caller:531] samples=4 events=95..237
      │ │   │     └─self.block_pool.cache_full_blocks( 🚩[95] :531 argument=4 objects=O1 events=95..237
      │ │   │           request=request,
      │ │   │           blocks=self.req_to_blocks[request.request_id],
      │ │   │           num_cached_blocks=num_cached_blocks,
      │ │   │           num_full_blocks=num_full_blocks,
      │ │   │           block_size=self.block_size,
      │ │   │           kv_cache_group_id=self.kv_cache_group_id,
      │ │   │           block_mask=block_mask,
      │ │   │       )
      │ │   └─_HiSparseAuxiliaryManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=88 events=96..761
      │ │     └─manager.cache_blocks( 🚩[96] :372 argument=88 objects=O1 events=96..761
      │ │           request,
      │ │           self.get_num_cacheable_tokens(num_computed_tokens, group_id),
      │ │           retention_interval=self.retention_interval,
      │ │           replay_boundaries=boundaries,
      │ │       )
      │ ├─scheduled_running_reqs.append(request) 🚩[104] :907 observed_write=11 objects=O1 events=104..762
      │ └─Scheduler._make_cached_request_data (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:1528] samples=44 events=105..766
      │   ├─cached_reqs_data = self._make_cached_request_data( 🚩[105] :1528 argument=11 historical_carrier_use=11 objects=O8,O9,O10,O11,O12,O13,O14,O15,O16,O17,O18 events=105..764
      │   │     scheduled_running_reqs,
      │   │     scheduled_resumed_reqs,
      │   │     num_scheduled_tokens,
      │   │     scheduled_spec_decode_tokens,
      │   │     req_to_new_blocks,
      │   │ )
      │   ├─for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)): 🚩[107] :1797 bind=11 objects=O1 events=107..765
      │   └─Request.num_output_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:1826] samples=11 events=108..766
      │     └─num_output_tokens.append( 🚩[108] :1826 argument=11 objects=O1 events=108..766
      │           req.num_output_tokens + req.num_output_placeholders
      │       )
      └─Scheduler.update_from_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:783] samples=366 events=115..873
        ├─request = self.requests.get(req_id) 🚩[115] :2172 builtin_result=12 bind=12 objects=O1 events=115..814
        ├─Request.is_finished (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2184] samples=12 events=117..815
        │ └─if request is None or request.is_finished(): 🚩[117] :2184 argument=12 objects=O1 events=117..815
        ├─Request.has_encoder_inputs (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2245] samples=24 events=118..817
        │ ├─if request.has_encoder_inputs: 🚩[118] :2245 argument=12 objects=O1 events=118..816
        │ └─Request.num_encoder_inputs (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:312] samples=12 events=119..817
        │   └─return self.num_encoder_inputs > 0 🚩[119] :312 argument=12 objects=O1 events=119..817
        ├─Scheduler._update_from_kv_xfer_finished (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2429] samples=25 events=164..430
        │ └─HiSparseConnector.update_connector_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:3322] samples=25 events=164..430
        │   └─HiSparseConnectorScheduler.update_connector_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:399] samples=25 events=164..430
        │     └─HiSparseCoordinator.update_spills (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:215] samples=25 events=164..430
        │       └─HiSparseCoordinator._complete_host_writes (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:781] samples=25 events=164..430
        │         └─HiSparseCoordinator._publish_host_blocks_if_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:827] samples=25 events=164..430
        │           ├─self.host_manager.publish_blocks( 🚩[164] :530 read_attribute=5 objects=O1 events=164..426
        │           │     publication.request,
        │           │     num_tokens,
        │           │     retention_interval=publication.retention_interval,
        │           │     replay_boundaries=publication.replay_boundaries,
        │           │ )
        │           └─HiSparseSourceManager.publish_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:529] samples=20 events=165..430
        │             ├─self.host_manager.publish_blocks( 🚩[165] :529 argument=5 objects=O1 events=165..427
        │             │     publication.request,
        │             │     num_tokens,
        │             │     retention_interval=publication.retention_interval,
        │             │     replay_boundaries=publication.replay_boundaries,
        │             │ )
        │             └─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:2332] samples=15 events=166..430
        │               ├─super().cache_blocks( 🚩[166] :2332 argument=5 objects=O1 events=166..428
        │               │     request,
        │               │     num_tokens,
        │               │     retention_interval=retention_interval,
        │               │     replay_boundaries=replay_boundaries,
        │               │ )
        │               └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=10 events=167..430
        │                 ├─super().cache_blocks( 🚩[167] :847 argument=5 objects=O1 events=167..429
        │                 │     request,
        │                 │     num_tokens,
        │                 │     retention_interval=retention_interval,
        │                 │     replay_boundaries=replay_boundaries,
        │                 │ )
        │                 └─BlockPool.cache_full_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/block_pool.py) [caller:531] samples=5 events=168..430
        │                   └─self.block_pool.cache_full_blocks( 🚩[168] :531 argument=5 objects=O1 events=168..430
        │                         request=request,
        │                         blocks=self.req_to_blocks[request.request_id],
        │                         num_cached_blocks=num_cached_blocks,
        │                         num_full_blocks=num_full_blocks,
        │                         block_size=self.block_size,
        │                         kv_cache_group_id=self.kv_cache_group_id,
        │                         block_mask=block_mask,
        │                     )
        ├─AsyncScheduler._update_request_with_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/async_scheduler.py) [caller:2260] samples=223 events=312..844
        │ ├─new_token_ids, stopped = self._update_request_with_output( 🚩[312] :2260 argument=8 objects=O1 events=312..818
        │ │     request, new_token_ids, is_stale=output_is_stale
        │ │ )
        │ ├─Scheduler._update_request_with_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:65] samples=71 events=313..826
        │ │ ├─new_token_ids, stopped = super()._update_request_with_output( 🚩[313] :65 argument=8 objects=O1 events=313..819
        │ │ │     request, new_token_ids, is_stale
        │ │ │ )
        │ │ ├─Request.append_output_token_ids (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2599] samples=32 events=314..823
        │ │ │ ├─request.append_output_token_ids(output_token_id) 🚩[314] :2599 argument=8 objects=O1 events=314..820
        │ │ │ └─Request.update_block_hashes (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:283] samples=24 events=315..823
        │ │ │   ├─self.update_block_hashes() 🚩[315] :283 argument=8 objects=O1 events=315..821
        │ │ │   └─get_request_block_hasher.<locals>.request_block_hasher (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py) [caller:288] samples=16 events=316..823
        │ │ │     ├─self.block_hashes.extend(self._block_hasher(self)) 🚩[316] :288 argument=8 objects=O1 events=316..822
        │ │ │     └─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:968] samples=8 events=317..823
        │ │ │       └─num_tokens = request.num_tokens 🚩[317] :968 argument=8 objects=O1 events=317..823
        │ │ └─check_stop (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/utils.py) [caller:2603] samples=31 events=318..826
        │ │   ├─stopped = check_stop(request, self.max_model_len) 🚩[318] :2603 argument=8 objects=O1 events=318..824
        │ │   ├─Request.num_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:115] samples=8 events=319..825
        │ │   │ └─request.num_tokens >= max_model_len 🚩[319] :115 argument=8 objects=O1 events=319..825
        │ │   ├─Request.num_output_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:116] samples=8 events=320..826
        │ │   │ └─or request.num_output_tokens >= request.max_tokens 🚩[320] :116 argument=8 objects=O1 events=320..826
        │ │   └─Request.num_output_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:126] samples=7 events=321..788
        │ │     └─if request.num_output_tokens < sampling_params.min_tokens: 🚩[321] :126 argument=7 objects=O1 events=321..788
        │ └─KVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:77] samples=144 events=322..844
        │   ├─self.kv_cache_manager.cache_blocks( 🚩[322] :77 argument=8 objects=O1 events=322..827
        │   │     request, request.num_computed_tokens - request.num_output_placeholders
        │   │ )
        │   └─KVCacheCoordinator.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py) [caller:792] samples=136 events=323..844
        │     ├─self.coordinator.cache_blocks(request, num_computed_tokens) 🚩[323] :792 argument=8 objects=O1 events=323..828
        │     ├─HiSparseSourceManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=48 events=324..834
        │     │ ├─manager.cache_blocks( 🚩[324] :372 argument=8 objects=O1 events=324..829
        │     │ │     request,
        │     │ │     self.get_num_cacheable_tokens(num_computed_tokens, group_id),
        │     │ │     retention_interval=self.retention_interval,
        │     │ │     replay_boundaries=boundaries,
        │     │ │ )
        │     │ └─HiSparseCoordinator.publish_when_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:2317] samples=40 events=325..834
        │     │   ├─self.coordinator.publish_when_ready( 🚩[325] :2317 argument=8 objects=O1 events=325..830
        │     │   │     request,
        │     │   │     num_tokens,
        │     │   │     retention_interval,
        │     │   │     replay_boundaries=replay_boundaries,
        │     │   │ )
        │     │   └─HiSparseCoordinator._publish_host_blocks_if_ready (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/hisparse/coordinator.py) [caller:509] samples=32 events=326..834
        │     │     ├─self.host_manager.publish_blocks( 🚩[326] :530 read_attribute=8 objects=O1 events=326..831
        │     │     │     publication.request,
        │     │     │     num_tokens,
        │     │     │     retention_interval=publication.retention_interval,
        │     │     │     replay_boundaries=publication.replay_boundaries,
        │     │     │ )
        │     │     └─HiSparseSourceManager.publish_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:529] samples=24 events=327..834
        │     │       ├─self.host_manager.publish_blocks( 🚩[327] :529 argument=8 objects=O1 events=327..832
        │     │       │     publication.request,
        │     │       │     num_tokens,
        │     │       │     retention_interval=publication.retention_interval,
        │     │       │     replay_boundaries=publication.replay_boundaries,
        │     │       │ )
        │     │       └─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:2332] samples=16 events=328..834
        │     │         ├─super().cache_blocks( 🚩[328] :2332 argument=8 objects=O1 events=328..833
        │     │         │     request,
        │     │         │     num_tokens,
        │     │         │     retention_interval=retention_interval,
        │     │         │     replay_boundaries=replay_boundaries,
        │     │         │ )
        │     │         └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=8 events=329..834
        │     │           └─super().cache_blocks( 🚩[329] :847 argument=8 objects=O1 events=329..834
        │     │                 request,
        │     │                 num_tokens,
        │     │                 retention_interval=retention_interval,
        │     │                 replay_boundaries=replay_boundaries,
        │     │             )
        │     ├─FullAttentionManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=16 events=330..836
        │     │ ├─manager.cache_blocks( 🚩[330] :372 argument=8 objects=O1 events=330..835
        │     │ │     request,
        │     │ │     self.get_num_cacheable_tokens(num_computed_tokens, group_id),
        │     │ │     retention_interval=self.retention_interval,
        │     │ │     replay_boundaries=boundaries,
        │     │ │ )
        │     │ └─SingleTypeKVCacheManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:847] samples=8 events=331..836
        │     │   └─super().cache_blocks( 🚩[331] :847 argument=8 objects=O1 events=331..836
        │     │         request,
        │     │         num_tokens,
        │     │         retention_interval=retention_interval,
        │     │         replay_boundaries=replay_boundaries,
        │     │     )
        │     └─_HiSparseAuxiliaryManager.cache_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py) [caller:372] samples=64 events=332..844
        │       └─manager.cache_blocks( 🚩[332] :372 argument=64 objects=O1 events=332..844
        │             request,
        │             self.get_num_cacheable_tokens(num_computed_tokens, group_id),
        │             retention_interval=self.retention_interval,
        │             replay_boundaries=boundaries,
        │         )
        ├─StructuredOutputManager.accept_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/structured_output/__init__.py) [caller:2279] samples=16 events=340..846
        │ ├─if new_token_ids and not self.structured_output_manager.accept_tokens( 🚩[340] :2279 argument=8 objects=O1 events=340..845
        │ └─Request.use_structured_output (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:535] samples=8 events=341..846
        │   └─if not request.use_structured_output or not new_token_ids: 🚩[341] :535 argument=8 objects=O1 events=341..846
        ├─Request.take_prefill_stats (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2301] samples=8 events=342..847
        │ └─prefill_stats = request.take_prefill_stats() 🚩[342] :2301 argument=8 objects=O1 events=342..847
        ├─KVCacheManager.estimate_cached_tokens (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:2304] samples=1 events=343..343
        │ └─prefill_stats.finalize( 🚩[343] :2304 argument=1 objects=O1 events=343..343
        │       self.kv_cache_manager.estimate_cached_tokens(request)
        │   )
        ├─Request.take_events (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2369] samples=8 events=344..867
        │ └─outputs[request.client_index].append( 🚩[344] :2369 argument=8 objects=O1 events=344..867
        │       EngineCoreOutput(
        │           request_id=req_id,
        │           new_token_ids=new_token_ids,
        │           finish_reason=finish_reason,
        │           new_logprobs=new_logprobs,
        │           new_sampling_mask=new_sampling_mask,
        │           new_prompt_logprobs_tensors=prompt_logprobs_tensors,
        │           prompt_token_id_logprobs=prompt_token_id_logprobs,
        │           pooling_output=pooler_output,
        │           stop_reason=request.stop_reason,
        │           events=request.take_events(),
        │           prefill_stats=prefill_stats,
        │           spec_decode_metrics=(
        │               request.spec_decode_metrics
        │               if finish_reason is not None
        │               else None
        │           ),
        │           kv_transfer_params=kv_transfer_params,
        │           ec_transfer_params=ec_transfer_params,
        │           trace_headers=request.trace_headers,
        │           routed_experts=routed_experts,
        │           num_nans_in_logits=request.num_nans_in_logits,
        │       )
        │   )
        ├─Request.get_finished_reason (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2320] samples=1 events=848..848
        │ └─finish_reason = request.get_finished_reason() 🚩[848] :2320 argument=1 objects=O1 events=848..848
        ├─Scheduler._handle_stopped_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2321] samples=1 events=849..849
        │ └─finished = self._handle_stopped_request(request) 🚩[849] :2321 argument=1 objects=O1 events=849..849
        ├─Scheduler._free_request (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2323] samples=16 events=850..865
        │ ├─kv_transfer_params, ec_transfer_params = self._free_request(request) 🚩[850] :2323 argument=1 objects=O1 events=850..850
        │ ├─Request.is_finished (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2817] samples=1 events=851..851
        │ │ └─assert request.is_finished() 🚩[851] :2817 argument=1 objects=O1 events=851..851
        │ ├─Scheduler._set_kv_fetch_stage (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2822] samples=1 events=852..852
        │ │ └─self._set_kv_fetch_stage(request, None) 🚩[852] :2822 argument=1 objects=O1 events=852..852
        │ ├─Scheduler._connector_finished (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2823] samples=6 events=853..858
        │ │ ├─connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request) 🚩[853] :2823 argument=1 objects=O1 events=853..853
        │ │ ├─KVCacheManager.finalize_partial_tail_offloads (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:3120] samples=1 events=854..854
        │ │ │ └─finished_partial_tails = ( 🚩[854] :3120 argument=1 objects=O1 events=854..854
        │ │ │       self.kv_cache_manager.finalize_partial_tail_offloads(request)
        │ │ │   )
        │ │ └─HiSparseConnector.request_finished_all_groups (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py) [caller:3155] samples=4 events=855..858
        │ │   ├─delay_free, kv_xfer_params = self.connector.request_finished_all_groups( 🚩[855] :3155 argument=1 objects=O1 events=855..855
        │ │   │     request, block_ids
        │ │   │ )
        │ │   └─self.connector_scheduler.requests.pop(request.request_id, None) 🚩[856] :407 read_attribute=1 historical_carrier_use=1 builtin_result=1 objects=O4,O1 events=856..858
        │ ├─EncoderCacheManager.free (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/encoder_cache_manager.py) [caller:2834] samples=2 events=859..860
        │ │ ├─self.encoder_cache_manager.free(request) 🚩[859] :2834 argument=1 objects=O1 events=859..859
        │ │ └─EncoderCacheManager.get_cached_input_ids (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/encoder_cache_manager.py) [caller:293] samples=1 events=860..860
        │ │   └─for input_id in list(self.get_cached_input_ids(request)): 🚩[860] :293 argument=1 objects=O1 events=860..860
        │ └─Scheduler._free_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2842] samples=5 events=861..865
        │   ├─self._free_blocks(request) 🚩[861] :2842 argument=1 objects=O1 events=861..861
        │   ├─Request.is_finished (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/request.py) [caller:2847] samples=1 events=862..862
        │   │ └─assert request.is_finished() 🚩[862] :2847 argument=1 objects=O1 events=862..862
        │   └─Scheduler._free_request_blocks (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2848] samples=3 events=863..865
        │     ├─self._free_request_blocks(request) 🚩[863] :2848 argument=1 objects=O1 events=863..863
        │     ├─Scheduler._request_blocks_can_be_freed (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py) [caller:2870] samples=1 events=864..864
        │     │ └─if self._request_blocks_can_be_freed(request): 🚩[864] :2870 argument=1 objects=O1 events=864..864
        │     └─KVCacheManager.free (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py) [caller:2871] samples=1 events=865..865
        │       └─self.kv_cache_manager.free(request) 🚩[865] :2871 argument=1 objects=O1 events=865..865
        ├─stopped_running_reqs.add(request) 🚩[866] :2339 observed_write=1 objects=O1 events=866..866
        └─remove_all (/home/server/ai-infra-dev/code/vllm-workspace/vllm/vllm/v1/core/sched/utils.py) [caller:2391] samples=6 events=868..873
          ├─self.running = remove_all(self.running, stopped_running_reqs) 🚩[868] :2391 argument=2 historical_carrier_use=2 objects=O5,O19 events=868..871
          └─return lst 🚩[872] :93 return=1 historical_carrier_use=1 objects=O5 events=872..873
```
