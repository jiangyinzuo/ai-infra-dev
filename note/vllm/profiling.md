## record_function_or_nullcontext

vllm/vllm:vllm/v1/core/sched/scheduler.py:832-838
```python
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
```
