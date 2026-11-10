"""用 4 个 CPU rank 各处理 4×8 矩阵的一行，验证 AllReduce 的分解。"""

from datetime import timedelta

import torch
import torch.distributed as dist

from cpu_collectives import _print_rank_lines


def print_comment(message: str) -> None:
    if dist.get_rank() == 0:
        print(f"\033[33m{message}\033[0m", flush=True)


def verify_decomposition() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"本例需要 4 个 rank，实际 world_size={world_size}")

    original = torch.arange(1, 9, dtype=torch.int64, device="cpu")
    original *= 10**rank

    print_comment(
        "输入按 rank 堆叠为 4×8 矩阵，每个 rank 只持有自己的 1 行（8 个元素）。"
        "规约将各行的同一列相加，不是对每行内部的元素求和。"
    )
    _print_rank_lines(f"[input] rank={rank}: matrix_row={original.tolist()}")

    print_comment(
        "1. AllReduce(sum)：将 4 行按列求和，每个 rank 接收完整的 8 元素结果。"
    )
    # AllReduce 原地修改张量；保留 original，供两条路径使用相同输入。
    all_reduce_output = original.clone()
    dist.all_reduce(all_reduce_output, op=dist.ReduceOp.SUM)
    _print_rank_lines(
        f"[all_reduce] rank={rank}: input={original.tolist()}, "
        f"output={all_reduce_output.tolist()}"
    )

    print_comment(
        "2. ReduceScatter(sum)：每行沿第 0 维分为 4 块，每块 2 个元素；"
        "逐元素求和后，rank i 只接收第 i 块。"
    )
    shard = torch.empty(original.numel() // world_size, dtype=original.dtype, device="cpu")
    dist.reduce_scatter_single(shard, original, op=dist.ReduceOp.SUM)
    _print_rank_lines(
        f"[reduce_scatter] rank={rank}: input={original.tolist()}, "
        f"input_chunks={[chunk.tolist() for chunk in original.chunk(world_size)]}, "
        f"output={shard.tolist()}"
    )

    print_comment(
        "3. AllGather：输入是上一步实际得到的 shard；"
        "按来源 rank 收集并拼接，不再求和。"
    )
    gathered_shards = [torch.empty_like(shard) for _ in range(world_size)]
    dist.all_gather(gathered_shards, shard)
    reconstructed = torch.cat(gathered_shards)
    _print_rank_lines(
        f"[all_gather] rank={rank}: input={shard.tolist()}, "
        f"received_chunks={[chunk.tolist() for chunk in gathered_shards]}, "
        f"output={reconstructed.tolist()}"
    )

    matches = torch.equal(all_reduce_output, reconstructed)
    if not matches:
        raise AssertionError(f"rank={rank}: 两条路径的结果不相等")
    expected = torch.tensor(
        [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888],
        dtype=original.dtype,
        device="cpu",
    )
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
    _print_rank_lines(
        f"[verify] rank={rank}: all_reduce={all_reduce_output.tolist()}, "
        f"reduce_scatter_then_all_gather={reconstructed.tolist()}, equal={matches}"
    )
    print_comment(
        "验证通过：4 个 rank 的两条路径结果均为 "
        "[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]。"
        "本例使用 int64 精确比较；浮点规约可能因运算顺序不同产生舍入差异。"
    )


def main() -> None:
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=60))
    try:
        verify_decomposition()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
