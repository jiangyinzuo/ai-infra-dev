PD 分离模式下需要启动三个独立进程：prefill server、decode server 和 router。每个 server 都是完整的 pp-size=2 实例。

1. 安装 Mooncake（KV 传输后端）

pip install mooncake-transfer-engine

2. 启动 Prefill Server（GPU 0,1 作为 PP rank 0 和 1）

CUDA_VISIBLE_DEVICES=0,1 \
python -m sglang.launch_server \
  --model-path /root/autodl-tmp/sglang/Qwen2.5-7B-Instruct \
  --tp-size 1 \
  --dp-size 1 \
  --pp-size 2 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --host 127.0.0.1 \
  --port 30000 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --mem-fraction-static 0.75 \
  --enable-trace \
  --otlp-traces-endpoint 127.0.0.1:4317
3. 启动 Decode Server（GPU 2 作为 PP rank 0）
CUDA_VISIBLE_DEVICES=2 \
python -m sglang.launch_server \
  --model-path /root/autodl-tmp/sglang/Qwen2.5-7B-Instruct \
  --tp-size 1 \
  --dp-size 1 \
  --pp-size 1 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --host 127.0.0.1 \
  --port 30001 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --mem-fraction-static 0.75 \
  --enable-trace \
  --otlp-traces-endpoint 127.0.0.1:4317
4. 启动 Router（分发请求）
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-trace \
  --otlp-traces-endpoint 127.0.0.1:4317
注意事项
- 同时占用 GPU：Prefill 和 Decode 两个 server 会各自占用全部 2 个 GPU（作为 PP rank 0/1）。这意味着需要两台机器各自运行一个
 server，或者你只能先跑 prefill 再跑 decode（但这样就无法通过 router 联合使用了）。如果只是测试 tracing 功能，可以用
--disaggregation-transfer-backend fake 避免安装 mooncake。
- Tracing：两个 server 和 router 都要加 --enable-trace --otlp-traces-endpoint，这样 trace 会被聚合到同一个 OpenTelemetry
collector。
