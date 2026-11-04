# vLLM多进程架构

```
API
EngineCore
- Scheduler
Worker 0 / NPU 0 本卡权重、KV和运行状态
Worker 1 / NPU 1
```

API <- ZMQ -> EngineCore <- MQ RPC -> Worker 0

Worker 0 <- HCCL -> Worker 1
