WSL运行jaeger

hostname -I 查看ip，

以172.22.240.192为例

http://172.22.240.192:16686/search

# Trace Level

sgl-project/sglang:python/sglang/srt/observability/req_time_stats.py:79-94
```python
@dataclass
class RequestStageConfig:
    """Configuration for a request pipeline stage.

    Attributes:
        stage_name: Name used for metrics labels and trace span names.
        level: Trace hierarchy depth.
            1 = leaf stages (atomic operations, e.g. TOKENIZE, PREFILL_FORWARD),
            2 = parent/dispatch stages (e.g. API_SERVER_DISPATCH, REQUEST_PROCESS),
            3 = composite/nested stages (e.g. DECODE_LOOP, PREFILL_CHUNKED_FORWARD).
        metrics_is_observed: Whether to call metrics_collector.observe_per_stage_req_latency.
    """

    stage_name: str
    level: int = 0
    metrics_is_observed: bool = False
```

设置trace level

- set_global_trace_level(1)  # 只记录最基础的操作
- set_global_trace_level(2)  # 记录基础操作 + 调度分发
- set_global_trace_level(3)  # 记录所有细节（默认）

level字段是一个trace过滤机制，允许用户根据需要动态调整trace的详细程度，在性能开销和可观测性之间取得平衡。这在分布式系统的observability设计中是一个常见且重要的模式。

```bash
curl "http://localhost:30000/set_trace_level?level=4"
```
