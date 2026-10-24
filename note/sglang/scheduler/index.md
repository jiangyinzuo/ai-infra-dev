# Scheduler

mixin设计模式，达到关注点分离的效果。防止“上帝类”scheduler混合了各方面的代码逻辑。
如果是C#，可以用`partial class`

每个Schduler...Mixin类似乎都只有方法，没有新增成员变量。

sgl-project/sglang:python/sglang/srt/managers/scheduler.py:297-310
```python
class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
    SchedulerDllmMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""
```

sgl-project/sglang:python/sglang/srt/managers/scheduler.py:3651-3665
```python
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )
```
sgl-project/sglang:python/sglang/srt/managers/scheduler.py:3692-3708
```python
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until shutdown)
        scheduler.run_event_loop()
```

## Event Loop

```
   SchedulerPPMixin::event_loop_pp
    󰊕 dispatch_event_loop ×2
      󰊕 Scheduler::run_event_loop
        󰊕 run_scheduler_process ×2
        󰊕 SchedulerActor::run_event_loop ×2
```

## run_batch

```
   Scheduler::run_batch
    󰊕 SchedulerDisaggregationDecodeMixin::event_loop_normal_disagg_decode ×2
    󰊕 SchedulerDisaggregationDecodeMixin::event_loop_overlap_disagg_decode ×2
    󰊕 SchedulerDisaggregationDecodeMixin::_run_batch_prebuilt ×2
    󰊕 SchedulerDisaggregationPrefillMixin::event_loop_normal_disagg_prefill ×2
    󰊕 SchedulerDisaggregationPrefillMixin::event_loop_overlap_disagg_prefill ×2
    󰊕 Scheduler::event_loop_normal ×2
    󰊕 Scheduler::event_loop_overlap ×2
    󰊕 SchedulerPPMixin::_pp_launch_batch ×2
    󰊕 SchedulerMultiplexMixin::event_loop_pdmux ×4
```
