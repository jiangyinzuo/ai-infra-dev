# Triton 学习环境

使用 uv 创建独立虚拟环境，安装 PyTorch 及其配套的 Triton，不修改系统 Python 或 vLLM 环境。

## 当前服务器

- Linux x86_64，Python 3.12.3。
- 两张 NVIDIA RTX PRO 2000 Blackwell，各约 16 GB 显存，Compute Capability 12.0。
- NVIDIA 驱动 580.178.04，已安装 CUDA 13.0 Toolkit。
- 已安装 uv，可直接使用 `uv` 命令。

以下使用 PyTorch 官方 CUDA 13.0 wheel 源。预编译包会安装所需的用户态 CUDA 依赖，通常不需要设置 `CUDA_HOME` 或手动修改 `LD_LIBRARY_PATH`。

## 初始化与安装

在仓库根目录执行：

```bash
cd learn-triton

# 创建本目录专用的虚拟环境，不使用系统 site-packages。
uv venv --python 3.12 .venv

# 直接指定目标解释器；安装时无需先激活环境，也无需安装 pip。
# Linux x86_64 下，PyTorch 会通过依赖安装配套的 Triton。
uv pip install --python .venv/bin/python \
  torch --default-index https://download.pytorch.org/whl/cu130
```

这里使用普通虚拟环境，不需要 `uv init` 创建 Python 包项目。暂不单独安装或升级 `triton`，避免破坏 PyTorch 的版本配套。

## 激活与检查

在 `learn-triton/` 目录执行：

```bash
source .venv/bin/activate

python - <<'PY'
import sys
import torch
import triton

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("PyTorch CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch 无法使用 CUDA，请检查驱动和安装包"
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

x = torch.arange(1024, device="cuda", dtype=torch.float32)
torch.testing.assert_close(x + x, x * 2)
torch.cuda.synchronize()
print("CUDA 运算检查通过")
PY
```

以上验证包导入和基础 CUDA 运算，**不代表已经验证 Triton kernel 编译**。下一步运行官方 [Vector Addition 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)，确认实际 Triton kernel 的编译、执行与结果校验。

## 运行朴素矩阵乘法

`naive_matmul.py` 使用标准 Triton kernel 语法，可在兼容的 CUDA Triton 或 Triton-Ascend 环境运行。默认 `--device auto` 优先选 CUDA，否则尝试 Ascend NPU；也可显式指定：

```bash
# CUDA；当前虚拟环境按上文安装
.venv/bin/python naive_matmul.py --device cuda

# Ascend；在已配置 CANN、torch_npu 和 Triton-Ascend 的环境运行
python naive_matmul.py --device npu --m 65 --n 97 --k 33
```

非整块尺寸示例用于验证边界 mask。脚本用 FP32 逐个 k 累加外积，每个 Triton program 计算一个 16 × 16 输出块，不使用 `tl.dot` 或 Tensor Core；运行后将结果复制到 CPU，与 PyTorch FP32 矩阵乘法校验并打印最大绝对误差。首次运行包含 JIT 编译。CUDA 环境安装说明仅适用于 NVIDIA；Ascend 需使用与硬件/CANN 匹配的 Triton-Ascend 软件栈。

## 日常使用

每次打开终端后，在 `learn-triton/` 目录激活环境：

```bash
source .venv/bin/activate
python your_kernel.py
```

也可以不激活，直接使用环境内的解释器：

```bash
.venv/bin/python your_kernel.py
```

需要指定一张 GPU 时，仅对当前命令设置：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python your_kernel.py
```

退出激活状态：

```bash
deactivate
```

## 注意事项

- 首次安装需要联网并下载较大的 PyTorch / CUDA 依赖，预留磁盘空间。
- 当前 GPU 是 Blackwell `sm_120`，不要假定针对数据中心 Blackwell `sm_100` 等架构的专用 kernel 可以直接运行。
- 不要将 `.venv/` 提交到 Git。
- 上述命令是安装步骤；本文不表示已在当前服务器完成安装或通过运行验证。

参考：[Triton 安装文档](https://triton-lang.org/main/getting-started/installation.html)、[PyTorch 安装入口](https://docs.pytorch.org/get-started/locally/)。
