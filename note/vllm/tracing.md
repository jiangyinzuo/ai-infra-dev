# vLLM tracing

## VLLM_TRACE_FUNCTION

基于Python `sys.settrace()`
```bash
VLLM_TRACE_FUNCTION=1 .venv/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

能证明实际执行了哪些函数，但不够直观地展示 vLLM 的完整执行流程。
日志约 35 MB、13.7 万行，对应不到 4 秒的活跃区间。
| 问题 | 对理解流程的影响 |
| --- | --- |
| 逐行记录 call / return，没有缩进 | 很难一眼看清调用层级 |
| 包含 PyTorch、Triton 等依赖代码 | 关键流程被大量实现细节淹没 |
| 同一调度循环反复执行 | 有 117 次 schedule，直接阅读非常重复 |
| 只记录 EngineCore 的一个线程 | HTTP 接收、聊天模板处理、API 输出不在这份日志里 |
| 没有参数、request ID、请求状态值 | 看得到调用，无法直接知道调度了哪些 token、请求何时完成 |
| 编译与 CUDA Graph 隐藏内部执行 | 不能据此展开每一层模型计算或 GPU kernel |
```
2026-10-07 12:50:57.504939 Call to _handle_client_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1611 from _process_input_queue in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1526
2026-10-07 12:50:57.505119 Call to _reject_add_in_shutdown in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1645 from _handle_client_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1619
2026-10-07 12:50:57.505191 Return from _reject_add_in_shutdown in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1647 to _handle_client_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1619
2026-10-07 12:50:57.505250 Call to add_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:486 from _handle_client_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:1621
2026-10-07 12:50:57.505333 Call to add_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/core/sched/scheduler.py:2598 from add_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/engine/core.py:526
2026-10-07 12:50:57.505391 Call to _enqueue_waiting_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/core/sched/scheduler.py:2439 from add_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/core/sched/scheduler.py:2615
2026-10-07 12:50:57.505443 Call to _is_blocked_waiting_status in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/core/sched/scheduler.py:2418 from _enqueue_waiting_request in /home/server/code/ai-infra-dev/code/vllm/vllm/v1/core/sched/scheduler.py:2440
```
