关闭 FlashInfer autotune，可能降低推理性能。

## DeepSeek-V3.2 P/D + HiSparse

- tiny random model（模型名tiny-dsa）
- NIXL

```bash
uv pip install --python vllm-workspace/vllm/.venv/bin/python 'nixl==1.5.2' 'nixl-cu13==1.5.2'
vllm-workspace/vllm/.venv/bin/python dev vllm serve \
  --script deepseek-v3.2-hisparse-pd.py \
  --devices "${P_GPU_UUID:?请设置P卡完整UUID},${D_GPU_UUID:?请设置D卡完整UUID}"
```

## GLM-5.3 P/D

tiny random BF16，模型名 `tiny-glm-5.3`；两张 SM120 GPU，设备顺序 P、D。

| 配方 | NIXL | HiSparse | 状态 |
| --- | --- | --- | --- |
| `glm-5.3-nixl-pd-pull.py` | pull | 关 | 可运行 |
| `glm-5.3-nixl-pd-push.py` | push | 关 | 可运行 |
| `glm-5.3-hisparse-pd.py` | pull | 开 | 可运行 |
| `glm-5.3-hisparse-pd-push.py` | push | 开 | 保留失败复现 |

在 `code/` 下运行，替换 `SCRIPT` 切换配方；共用端口，不能同时启动：

```bash
SCRIPT=glm-5.3-nixl-pd-pull.py
vllm-workspace/vllm/.venv/bin/python dev vllm serve \
  --script "$SCRIPT" \
  --devices "${P_GPU_UUID:?请设置P卡完整UUID},${D_GPU_UUID:?请设置D卡完整UUID}"
```

P/D HTTP：8100/8200；NIXL side-channel：5600/5601；请求入口：`http://127.0.0.1:8192/v1/completions`。
日志：`dev/serve_config/runtime/hisparse-pd-*/{prefill,decode,proxy}.log`；状态：

```bash
vllm-workspace/vllm/.venv/bin/python dev vllm serve status
vllm-workspace/vllm/.venv/bin/python dev vllm serve stop
```

pull 由 D 发起 NIXL READ；push 由 P 发起 NIXL WRITE，代理并发提交 P/D 请求。
纯 NIXL 两套使用相同模型、KV dtype、block size、容量和 seed，仅切换传输模式及配套代理。
现有代理带 tracing；用于路径学习，不能直接用请求耗时评价传输性能。

HiSparse + push 在 Decode 启动阶段保留框架原始错误，监管脚本随后停止整组服务：

```text
NotImplementedError: NixlPushConnector does not support mixed-memory KV caches
```

## GLM-5.3 P/D + Mooncake 池 + HiSparse

P：NIXL + Mooncake Store；D：NIXL + HiSparse，不写回池。单机 TCP 功能验证，非 RDMA 性能测试。
在 `code/` 下运行；master、client 分别保持前台运行，再启动 P/D：

```bash
uv pip install --python vllm-workspace/vllm/.venv/bin/python 'mooncake-transfer-engine-cuda13==0.3.14'
vllm-workspace/vllm/.venv/bin/mooncake_master --rpc_address=127.0.0.1 --rpc_port=50051 --metrics_host=127.0.0.1 --metrics_port=9003 --logtostderr
vllm-workspace/vllm/.venv/bin/mooncake_client --host=127.0.0.1 --port=50052 --master_server_address=127.0.0.1:50051 --metadata_server=P2PHANDSHAKE --protocol=tcp --global_segment_size=512MB --local_buffer_size=16MB --logtostderr
MOONCAKE_CONFIG_PATH="$PWD/dev/serve_config/mooncake-pd.json" vllm-workspace/vllm/.venv/bin/python dev vllm serve --script glm-5.3-hisparse-pd-mooncake.py --devices "${P_GPU_UUID:?},${D_GPU_UUID:?}"
```

`serve stop` 只停止 P/D 与代理；master/provider 单独 Ctrl-C。保留 provider 可跨 P/D 重启复用池，停止 provider 会丢失内存 KV。

## DeepSeek-V3.2 DP=2, EP=1

两张卡各运行一个 DP rank；MoE 专家不跨 DP rank 切分（不启用 `--enable-expert-parallel`）。

```bash
vllm-workspace/vllm/.venv/bin/python dev vllm serve \
  --script deepseek-v3.2-dp2-ep1.py \
  --devices "${GPU0_UUID:?设置 GPU0 完整 UUID},${GPU1_UUID:?设置 GPU1 完整 UUID}"
```
