# Python call stack analysis

Python 3.12+

Attach进程
```bash
mkdir -p /tmp/runtime/pd-trace
python -m trace_python.call_stack_analysis.attach \
  --pid 645792 645764 645014 \
  --process-label 645792=P --process-label 645764=D \
  --config trace_python/call_stack_analysis/config/pd_hisparse_attach.py \
  --output-prefix /tmp/runtime/pd-trace/capture \
  --duration 60 --timeout 10
```

`TraceSession(..., process_label="P")` 设置业务标签；自动前缀为 `BASE.[label.]process_name.PID-UUID`，名称经过文件名清理。
采集入口刷新进程名并重命名文件，不依赖正常退出；run 保留初始身份，event/summary 记录当前身份。
`capture(output_prefix=...)` 保持指定前缀不变。

`.tree.txt` 用 `🔎[事件号]` 在函数下展示变量，每条路径最多 2 次；其余记录查看 `.txt` 或 `.jsonl`。
文本中每个变量单独一行，容器使用 Python 语法；最多展示 16 项，超出以 `...` 和 `length` 标明。
`Target(..., caller_fields=("needed", "available"))` 读取直接调用者变量，以 `👆` 标识；不跳过包装函数，与 `🔎` 共用 2 次展示上限。

## Multi-API startup trace

默认 TP=2、API 进程数=2；`TRACE_VLLM_STARTUP_TENSOR_PARALLEL_SIZE=1` 改用单卡。
启动配置默认标签为 `multi-api`、`dp2-ep1` 或 `hisparse`；`TRACE_PROCESS_LABEL=P` 覆盖默认值，根进程与子进程共用标签。
```bash
cd "$HOME/ai-infra-dev/trace_python"
python call_stack_analysis/ --config call_stack_analysis/config/startup_multi_api.py --output-prefix /tmp/startup-multi-api
```

## DP=2, EP=1 startup trace

This installs the tracer before vLLM starts; attach mode cannot capture completed startup calls.

```bash
cd "$HOME/ai-infra-dev"
mkdir -p /tmp/runtime/startup-trace
TRACE_VLLM_STARTUP_DEVICES="${GPU0_UUID:?set GPU0 UUID},${GPU1_UUID:?set GPU1 UUID}" \
TRACE_VLLM_STARTUP_SECONDS=180 \
  "$HOME/ai-infra-dev/code/vllm-workspace/vllm/.venv/bin/python" \
  -m trace_python.call_stack_analysis \
  --config trace_python/call_stack_analysis/config/startup_dp2_ep1.py \
  --output-prefix /tmp/runtime/startup-trace/dp2-ep1
```

Uses the matching `dev/serve_config/deepseek-v3.2-dp2-ep1.py`; records launcher, DP EngineCore, Worker, model and KV-cache initialization. `TRACE_VLLM_STARTUP_SECONDS=0` waits for Ctrl-C. The merged `.timeline.txt` shows call order; per-process `.jsonl` files retain stacks and DP-rank fields.



## GLM-5.3 HiSparse + push 启动失败

仅运行 `glm-5.3-hisparse-pd-push.py` 的 Decode 原命令；不启动 P/Proxy，不发送请求。

```bash
cd "$HOME/ai-infra-dev"
mkdir -p /tmp/runtime/startup-trace
TRACE_VLLM_STARTUP_DEVICES="${D_GPU_UUID:?设置 D 卡 UUID}" \
TRACE_VLLM_STARTUP_SECONDS=120 \
TRACE_VLLM_STARTUP_OUTPUT_PREFIX="/tmp/runtime/startup-trace/glm-push" \
  code/vllm-workspace/vllm/.venv/bin/python -m trace_python.call_stack_analysis \
  --config trace_python/call_stack_analysis/config/startup_glm_hisparse_push.py \
  --output-prefix /tmp/runtime/startup-trace/controller
```

查看 `glm-push-*.D.EngineCore.*.tree.txt`：入口事件记录 KV 注册过程、`_mixed_mem_types` 和 DRAM/VRAM 区域。
预期原始异常为 `NixlPushConnector does not support mixed-memory KV caches`；树不采集异常事件，需结合进程日志阅读。`status=ok` 和控制端退出码只表示采集结束，不表示服务启动成功。

## Limit

- 仅实现了函数入口trace，还没实现行级采集
