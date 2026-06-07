https://docs.vllm.ai/en/latest/contributing/index.html#documentation

```bash
pip install uv
git clone https://github.com/vllm-project/vllm.git --depth 1000 && cd vllm
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
# 如果VLLM_USE_PRECOMPILED=0，会从头编译项目
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 VLLM_PRECOMPILED_WHEEL_COMMIT=nightly uv pip install -e . -v --extra-index-url https://wheels.vllm.ai/nightly
```

测试服务器 `curl --noproxy '*' -sS -v http://127.0.0.1:8000/v1/models`

