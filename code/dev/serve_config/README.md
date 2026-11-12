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

## GLM-5.3 P/D + HiSparse

```bash
vllm-workspace/vllm/.venv/bin/python dev vllm serve \
  --script glm-5.3-hisparse-pd.py \
  --devices "${P_GPU_UUID:?请设置P卡完整UUID},${D_GPU_UUID:?请设置D卡完整UUID}"
```
