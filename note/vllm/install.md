## 编译和本地安装

https://docs.vllm.ai/en/latest/contributing/index.html#documentation

### 创建venv并`pip install -e .`

#### GPU环境

setup.py 会去 https://wheels.vllm.ai/{commit}/{variant}/vllm/metadata.json 取预编译 wheel。
如果当前commit还没产出wheel，安装可能报错。
```bash
# 如果VLLM_USE_PRECOMPILED=0，会从头编译项目
# 从此工作区的 vLLM 目录运行
cd code/vllm-workspace/vllm
uv venv --python 3.12 --seed --managed-python
# 企业内网可能需要用系统证书
uv venv --python 3.12 --seed --managed-python --system-certs
source .venv/bin/activate
python3 ~/ai-infra-dev/code/dev/ vllm install
# 自动发现已发布的 CUDA wheel，失败时不会执行安装
# env 输出同时固定 commit 和 CUDA variant，避免安装时选择不同索引
if wheel_env=$(python3 ../../../scripts/find_vllm_precompiled_commit.py --format env); then
    eval "$wheel_env"
    uv pip install -e .
fi
# 退出当前虚拟环境
deactivate
```

脚本读取官方 nightly 元数据中的完整 commit，再验证该 commit 的索引和当前架构的 wheel 下载地址（HEAD，不下载整个 wheel）。默认按当前 `setup.py` 的规则检测 CUDA variant；也可显式指定：

```bash
# 在工作区根目录，仅输出完整 commit；诊断写到 stderr
python3 scripts/find_vllm_precompiled_commit.py --variant cu129
# 打印安装命令，不执行安装
python3 scripts/find_vllm_precompiled_commit.py --variant cu129 --format command
```

可用参数：`--variant cu129|cu130|default`、`--arch x86_64|aarch64`、`--timeout 20`、`--format commit|env|command`。`default` 仅用于旧版使用无 variant 根索引的 `setup.py`；当前版本需要具体 CUDA variant。此脚本针对 CUDA wheel，不用于 Ascend/ROCm 安装。nightly 表示该 variant 当前已发布的构建，不保证是 GitHub `main` 的最新提交，也不保证与旧 checkout 的 C++/CUDA 接口兼容。

#### 纯CPU环境

仅用于python代码阅读，不编译C++
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend cpu
```

如果要源码编译，需要`sudo apt-get install -y libnuma-dev`

## 部署与测试

```bash
# 查看当前机器有哪些模型
hf cache ls
# 运行服务
.venv/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct
```
测试服务器 `curl --noproxy '*' -sS -v http://127.0.0.1:8000/v1/models`

