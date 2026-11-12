## entrypoint

vllm/vllm:pyproject.toml:42-43
```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

AsyncEngineArgs.create_engine_config
- CLI 参数转成 VllmConfig
- 确定模型、并行、调度、缓存等配置

## API server

run_multi_api_server
- 多 API server / Rust frontend 的启动入口
- 创建 EngineCore 后启动 frontend manager

run_server
run_server_worker
- 单 API server 启动路径
- API server worker 内构建 AsyncLLM

run_headless
- 只启动 EngineCore，不启动 API server

**单 API / 多 API 的区别是前端 API 进程数，不是模型 Worker 数，也不是 DP/TP/EP 并行度。**

| | 单 API server | 多 API server |
|---|---|---|
| 进程结构 | 同一个进程运行 API 服务和 `AsyncLLM` 前端 | launcher 进程单独管理 EngineCore，并启动多个 API server 子进程 |
| 引擎连接 | `AsyncLLM` 创建 EngineCore 客户端；EngineCore 和 Worker 仍是后台进程 | 多个 API server 通过各自的 ZMQ listener/client 连接到 EngineCore |
| 主要差别 | 前端处理集中在一个进程 | 前端处理分散到多个进程；需要管理共享引擎连接和多进程指标 |
| 资源代价 | 前端进程开销较小 | 增加 API 进程与 CPU/内存开销；不代表模型权重或 EngineCore 必然复制 |

```text
单 API：
vllm serve 进程
├── API / HTTP 服务
├── AsyncLLM / EngineCoreClient
└── EngineCore 进程
    └── Executor
        └── Worker 进程

多 API：
vllm serve launcher
├── EngineCore 进程
│   └── Executor
│       └── Worker 进程
├── API server 进程 0
└── API server 进程 1 ...
```

在当前代码中，单 API 分支会运行 `run_server()` → `run_server_worker()`，由 API server 内的 `AsyncLLM` 创建 engine client。多 API 分支运行 `run_multi_api_server()`：先调用 `launch_core_engines()`，再由 `APIServerProcessManager` 启动多个 API server 进程。

- `--api-server-count` 控制 API 前端进程数；它不控制 TP/EP Worker 数。DP 配置会影响 API server 数的默认值，但两者不是同一个配置概念。
- 多 API server 不是线性吞吐保证：它增加前端并发处理能力，实际效果仍受共享 EngineCore、GPU 和请求负载限制。当前代码也限制多 API server 与 runtime LoRA updating 同时使用。

APIServerProcessManager
RustFrontendProcessManager
- 管理 API server / Rust frontend 进程

CoreEngineProcManager
- 创建和管理 EngineCore 进程
- 处理启动、就绪、关闭

vllm/vllm:vllm/v1/engine/utils.py:1205-1212
```python
@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Iterator[CoreEngineLaunch]:
    """Launch engine and DP coordinator processes as needed."""
```

build_async_engine_client_from_engine_args
- 创建 VllmConfig 和 AsyncLLM

class AsyncLLM
- 初始化 renderer、InputProcessor、OutputProcessor
- 创建 EngineCoreClient
    self.renderer 会话模板

class InputProcessor
class OutputProcessor

EngineCoreClient.make_async_mp_client
- 创建 frontend 到 EngineCore 的客户端连接

build_and_serve
- 获取支持的任务
- 构造 app、初始化 app state、启动 HTTP 服务

OnlineRenderer
OnlineDerenderer
ServingTokenization
- API 请求侧的渲染、反渲染、tokenization

不管worker进程，APIServer和EngineCore是怎么启动的？

## EngineCore

DPEngineCoreProc

vllm/vllm:vllm/v1/engine/core.py:1343-1344
```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
    ...
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]() 请求
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
    ...
            engine_core.run_busy_loop()
```

EngineCoreProc._perform_handshakes
- frontend、EngineCore、DP coordinator 交换 ZMQ 地址与配置

### EngineCore初始化Model

vllm/vllm:vllm/v1/engine/core.py:162-163
```python
        # Setup Model.
        self.model_executor = executor_class(vllm_config)
```

### EngineCore初始化KVCache

vllm/vllm:vllm/v1/engine/core.py:175-176
```python
        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
```

EngineCore._initialize_kv_caches
- 收集 Worker 的 KV cache specs
- 确定 KV cache layout
- profiling 可用显存
- 生成并初始化 KV cache 配置
- 执行 compile / warmup

### EngineCore初始化调度器

一个DP组一个调度器
- 每个 DP engine 对应自己的调度与执行入口

vllm/vllm:vllm/v1/engine/core.py:179-201
```python
        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
```

### EngineCore初始化Prefix Hash

vllm/vllm:vllm/v1/engine/core.py:241-251
```python
            self.request_block_hasher = get_request_block_hasher(
                hash_block_size,
                caching_hash_fn,
                vllm_config.num_hash_lookahead_tokens,
            )
            self.prefix_hash_config = PrefixHashConfig(
                hash_algo=vllm_config.cache_config.prefix_caching_hash_algo,
                hash_block_size=hash_block_size,
                none_hash=kv_cache_utils.NONE_HASH,
                num_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
            )
```

### EngineCore.step:

step函数

vllm/vllm:vllm/v1/engine/core.py:253-255
```python
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
```

- 调度器调度请求 scheduler.schedule
- self.model_executor.execute_model


## Executor

默认

vllm/vllm:vllm/v1/executor/multiproc_executor.py:149-149
```python
class MultiprocExecutor(Executor):
```

vllm/vllm:vllm/v1/executor/multiproc_executor.py:229-234
```python
            def start_worker(local_rank: int) -> UnreadyWorkerProcHandle:
                global_rank = global_start_rank + local_rank
                with cpu_omp_manager.configure_omp_envs(
                    rank=global_rank, local_rank=local_rank
                ):
                    return WorkerProc.make_worker_process(
```

通信域

vllm/vllm:vllm/v1/executor/multiproc_executor.py:867-868
```python
            "local_rank": local_rank,
            "rank": rank,
```

EngineCore是独立进程，worker也是独立进程

vllm/vllm:vllm/v1/executor/multiproc_executor.py:878-884
```python
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )
```

vllm/vllm:vllm/v1/executor/abstract.py:53-58
```python
class Executor(ABC):
    """Abstract base class for vLLM executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """
```


## Worker

Worker
- 进程拥有自己的 rank、ModelRunner
- TP / PP / EP 等是 Worker 间通信组

MultiprocExecutor._init_executor
- 创建 Worker 进程
- 等待 Worker ready

NPUPlatform._setup_worker_and_scheduler

vllm/vllm:vllm/v1/executor/multiproc_executor.py:742-742
```python
class WorkerProc:
    worker_main
```

vllm/vllm:vllm/v1/worker/worker_base.py:240-246
```python
class WorkerWrapperBase:
    """This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """
```
vllm/vllm:vllm/v1/worker/worker_base.py:288-289
```python
    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
```

通信域放在`class Worker`

vllm/vllm:vllm/v1/worker/gpu_worker.py:216-216
```python
class Worker(WorkerBase):
```

Worker.determine_available_memory
- dummy forward / profiling
- 估算 KV cache 可用显存

Worker.initialize_from_config
- 按 KV cache 配置初始化 Worker 侧缓存

创建通信域

vllm/vllm:vllm/distributed/parallel_state.py:2085-2092
```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """Initialize model parallel groups.
```

每个Worker都有自己的通信域、ModelRunner

vllm/vllm:vllm/v1/worker/gpu_worker.py:567-579
```python
    def _init_workspace_and_model_runner(self) -> None:
        """Set up the workspace manager, build the model runner, report usage."""
        # DSpark target and draft CUDA graphs retain workspace views concurrently.
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(
            self.device,
            num_ubatches,
            _num_workspace_lanes(self.vllm_config),
            # Scratch holds no state across steps: discard it on sleep.
            alloc_context=lambda: self._maybe_get_memory_pool_context("workspace"),
        )
        with self._maybe_get_memory_pool_context(tag="runtime"):
            self.model_runner: GPUModelRunner = self._make_model_runner()
```

GPUModelRunner.load_model
BaseModelLoader.load_model


## BaseModelLoader

ModelLoader和模型权重切分有关

create_model initialize_model

## KVCacheManager

## KVConnector

## DP coordinator
- DP rank 协调与负载均衡
- 是否存在取决于 DP / load-balancing 配置

## DP2 TP2 EP4为例

vllm serve主进程
- launch_core_engines 进程
    - DP0 EngineCoreProc 进程, EngineCore
        - 调度器
        - executor
            - Worker进程
                - init_worker
                - init_device
                - init 并行策略、通信域
                - init_model_runner
                - load_model
        - prefixcache
        - _initialize_kv_caches
    - DP1 EngineCoreProc 进程
    - TP
        - process_input_queue
        - process_engine_step busy loop
- api_server_manager 进程，对外暴露的服务
    - api0: input processor, output processor, renderer, 注册路由
    - api1
