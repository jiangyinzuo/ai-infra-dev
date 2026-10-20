"""朴素 FP32 矩阵乘法：逐个 k 累加外积，不使用 tl.dot 或 Tensor Core。

在 learn-triton/ 目录激活环境后运行：
    python naive_matmul.py
    python naive_matmul.py --m 65 --n 97 --k 33
"""

import argparse

import torch
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128, help="A 和 C 的行数")
    parser.add_argument("--n", type=int, default=128, help="B 和 C 的列数")
    parser.add_argument("--k", type=int, default=128, help="归约维度")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("m、n、k 必须为正整数")
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA GPU，请先安装 CUDA 版 PyTorch 和配套 Triton")

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float32)
    c = torch.empty((args.m, args.n), device="cuda", dtype=torch.float32)

    block_m, block_n = 16, 16
    grid = (triton.cdiv(args.m, block_m), triton.cdiv(args.n, block_n))
    matmul_kernel[grid](
        a, b, c, args.m, args.n, args.k, block_m, block_n, num_warps=4
    )
    torch.cuda.synchronize()

    # 用 FP32 参考结果校验，避免 TF32 精度差异干扰入门实验。
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = a @ b
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)
    max_error = (c - expected).abs().max().item()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"A: {tuple(a.shape)}, B: {tuple(b.shape)}, C: {tuple(c.shape)}")
    print(f"校验通过，最大绝对误差: {max_error:.6g}")


if __name__ == "__main__":
    main()
