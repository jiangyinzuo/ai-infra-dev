# vLLM Architecture Overview

```text
OpenAI/API frontend
        ↓
LLMEngine / AsyncLLM
        ↓
EngineCoreClient
        ↓
EngineCore
   ├── Scheduler
   ├── KVCacheManager / Coordinator
   └── Executor
          ↓
       Worker / ModelRunner
          ├── model weights
          ├── KV cache tensors
          └── BlockTables
```

- Scheduler 管逻辑请求和 KV block。
- Executor 管 Worker 生命周期与 RPC。
- Worker／ModelRunner 执行设备侧模型和 KV cache。
- KV cache group 是缓存管理维度，不等于 Worker 数量。
