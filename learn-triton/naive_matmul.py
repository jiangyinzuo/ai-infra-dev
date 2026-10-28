"""朴素 FP32 矩阵乘法：逐个 k 累加外积，不使用 tl.dot 或 Tensor Core。

在 learn-triton/ 目录的 CUDA 或 Ascend Triton 环境中运行：
    python naive_matmul.py
    python naive_matmul.py --device npu --m 65 --n 97 --k 33
    python naive_matmul.py --device cuda
"""

import argparse
import importlib
import importlib.util

import torch

_TORCH_NPU_IMPORT_ERROR: Exception | None = None
if importlib.util.find_spec("torch_npu") is not None:
    try:
        importlib.import_module("torch_npu")
    except (ImportError, OSError) as exc:
        # Ascend Triton examples import torch_npu before importing Triton.
        _TORCH_NPU_IMPORT_ERROR = exc

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 每个 program 负责 C 的一个二维块；输入都是连续的行主序矩阵。
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)

    for k in range(K):
        # A 的一列与 B 的一行广播成外积，再累加到输出块。
        # mask 让非整块尺寸也安全；越界元素用 0 填充。
        a = tl.load(a_ptr + rows * K + k, mask=rows < M, other=0.0)
        b = tl.load(b_ptr + k * N + cols, mask=cols < N, other=0.0)
        acc += a[:, None] * b[None, :]

    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptr + offsets, acc, mask=mask)


def resolve_device(requested: str, parser: argparse.ArgumentParser):
    # --device=auto preserves the existing preference for CUDA.
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda:0"), torch.cuda
    if requested == "cuda":
        parser.error(
            "CUDA 不可用；请安装支持 CUDA 的 PyTorch/Triton，或使用 --device npu"
        )
    npu_backend = getattr(torch, "npu", None)
    if npu_backend is not None and npu_backend.is_available():
        return torch.device("npu:0"), npu_backend
    if requested == "npu":
        detail = (
            f"；导入 torch_npu 失败：{_TORCH_NPU_IMPORT_ERROR}"
            if _TORCH_NPU_IMPORT_ERROR
            else ""
        )
        parser.error(
            "NPU 不可用；请按兼容版本配置 CANN、torch_npu 和 Triton-Ascend" + detail
        )
    parser.error("未检测到可用 CUDA GPU 或 Ascend NPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128, help="A 和 C 的行数")
    parser.add_argument("--n", type=int, default=128, help="B 和 C 的列数")
    parser.add_argument("--k", type=int, default=128, help="归约维度")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "npu"),
        default="auto",
        help="计算后端；auto 优先 CUDA，否则尝试 Ascend NPU",
    )
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("m、n、k 必须为正整数")
    device, backend = resolve_device(args.device, parser)

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float32)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float32)
    c = torch.empty((args.m, args.n), device=device, dtype=torch.float32)

    block_m, block_n = 16, 16
    grid = (triton.cdiv(args.m, block_m), triton.cdiv(args.n, block_n))
    matmul_kernel[grid](a, b, c, args.m, args.n, args.k, block_m, block_n)
    backend.synchronize()

    # 在 CPU 上生成 FP32 参考值，避免 CUDA TF32 或 NPU matmul 模式影响校验。
    a_cpu, b_cpu, c_cpu = a.cpu(), b.cpu(), c.cpu()
    expected = a_cpu @ b_cpu
    torch.testing.assert_close(c_cpu, expected, rtol=1e-3, atol=1e-3)
    max_error = (c_cpu - expected).abs().max().item()
    print(f"设备: {device}")
    print(f"A: {tuple(a.shape)}, B: {tuple(b.shape)}, C: {tuple(c.shape)}")
    print(f"校验通过，最大绝对误差: {max_error:.6g}")


if __name__ == "__main__":
    main()


