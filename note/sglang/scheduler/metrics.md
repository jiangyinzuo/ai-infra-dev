

sgl-project/sglang:python/sglang/srt/managers/scheduler.py:389-390
```python
        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)
```


- record_bubble_metrics: 统计bubble时间
- record_forward_metrics: 实际forward的GPU执行时间

sgl-project/sglang:python/sglang/srt/managers/scheduler.py:2735-2742
```python
                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
```
