## 编译和本地安装

https://docs.vllm.ai/en/latest/contributing/index.html#documentation

setup.py 会去 https://wheels.vllm.ai/{commit}/{variant}/vllm/metadata.json 取预编译 wheel。
如果当前commit还没产出wheel，安装可能报错。
```bash
# 如果VLLM_USE_PRECOMPILED=0，会从头编译项目
# VLLM_PRECOMPILED_WHEEL_COMMIT请替换为实际的commit
cd ~/vllm
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=58b3298457dde7b4554b3b4e20b238c0ac2c3a65 \
uv pip install -e .
```

## 测试

测试服务器 `curl --noproxy '*' -sS -v http://127.0.0.1:8000/v1/models`

