# Python call stack analysis

Python 3.12+

Attach进程
```bash
python -m trace_python.call_stack_analysis.attach \
  --pid 645792 645764 645014 \
  --config trace_python/call_stack_analysis/config/pd_hisparse_attach.py \
  --output-prefix runtime/pd-trace/capture \
  --duration 60 --timeout 10
```

`.tree.txt` 用 `🔎[事件号]` 在函数下展示变量，每条路径最多 2 次；其余记录查看 `.txt` 或 `.jsonl`。
文本中每个变量单独一行，容器使用 Python 语法；最多展示 16 项，超出以 `...` 和 `length` 标明。
`Target(..., caller_fields=("needed", "available"))` 读取直接调用者变量，以 `👆` 标识；不跳过包装函数，与 `🔎` 共用 2 次展示上限。


## vLLM startup process order

Run with the vLLM virtualenv and two visible GPUs (`CUDA_VISIBLE_DEVICES=<id0>,<id1>`). The configs use the tiny dummy DeepSeek-V3.2 recipe with tensor parallel size 2; spawned processes each load the same entry tracer.
```bash
cd "$HOME/ai-infra-dev"
mkdir -p runtime/startup-trace
TRACE_VLLM_STARTUP_SECONDS=180 \
  "$HOME/ai-infra-dev/code/vllm-workspace/vllm/.venv/bin/python" \
  -m trace_python.call_stack_analysis \
  --config trace_python/call_stack_analysis/config/startup_single_api.py \
  --output-prefix runtime/startup-trace/single-api
```

`startup_headless.py` currently has no frontend client, so it can stop at the EngineCore handshake and may not reach model initialization. Use either API config to trace the full EngineCore/Worker initialization sequence.

Set `TRACE_VLLM_STARTUP_TENSOR_PARALLEL_SIZE=1` to use one GPU. Set `TRACE_VLLM_STARTUP_SECONDS=0` to keep the service running until Ctrl-C.

## Limit

- 仅实现了函数入口trace，还没实现行级采集
