"""用 torchrun 启动多个 CPU 进程，观察最基础的集合通信。"""

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist


def _print_rank_lines(*messages: str) -> None:
    # 每个算子只汇总一次日志；这次额外通信仅用于演示输出排序。
    received: list[tuple[str, ...]] | None = (
        [()] * dist.get_world_size() if dist.get_rank() == 0 else None
    )
    dist.gather_object(messages, object_gather_list=received, dst=0)
    if received is not None:
        for rank_messages in received:
            for message in rank_messages:
                print(message, flush=True)


def broadcast() -> None:
    rank = dist.get_rank()
    tensor = torch.tensor([rank + 1, 10 * (rank + 1)], device="cpu")
    before = tensor.tolist()

    # 所有 rank 都调用；src=0 的张量覆盖每个 rank 的 tensor。
    dist.broadcast(tensor, src=0)

    _print_rank_lines(f"[broadcast] rank={rank}: input={before}, output={tensor.tolist()}")


def _reduction_input() -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return torch.tensor(
        [
            [(rank + 1) * (destination + 1), 10 * (rank + 1) * (destination + 1)]
            for destination in range(world_size)
        ],
        device="cpu",
    ).flatten()


def reduce() -> None:
    rank = dist.get_rank()
    tensor = _reduction_input()
    before = tensor.tolist()

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    # 只有 dst 上的规约结果有定义，不读取其他 rank 的结果。
    result = tensor.tolist() if rank == 0 else "仅 rank=0 接收结果"
    _print_rank_lines(f"[reduce] rank={rank}: input={before}, output={result}")


def all_reduce() -> None:
    rank = dist.get_rank()
    tensor = _reduction_input()
    before = tensor.tolist()

    # 原地逐元素求和，不是把一个张量的所有元素加成一个标量。
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    _print_rank_lines(f"[all_reduce] rank={rank}: input={before}, output={tensor.tolist()}")


def reduce_scatter() -> None:
    rank = dist.get_rank()
    tensor = _reduction_input()
    received = torch.empty(2, dtype=tensor.dtype, device="cpu")

    # 每块两个元素；先逐元素求和，rank i 只接收规约后的第 i 块。
    dist.reduce_scatter_single(received, tensor, op=dist.ReduceOp.SUM)

    _print_rank_lines(f"[reduce_scatter] rank={rank}: input={tensor.tolist()}, output={received.tolist()}")


def gather() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.tensor([rank + 1, 10 * (rank + 1)], device="cpu")
    received = [torch.empty_like(tensor) for _ in range(world_size)] if rank == 0 else None

    dist.gather(tensor, gather_list=received, dst=0)

    # 列表按 rank 排列；gather 只收集，不求和。
    result = [item.tolist() for item in received] if received is not None else "仅 rank=0 接收结果"
    _print_rank_lines(f"[gather] rank={rank}: input={tensor.tolist()}, output={result}")


def all_gather() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.tensor([rank + 1, 10 * (rank + 1)], device="cpu")
    received = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.all_gather(received, tensor)

    result = [item.tolist() for item in received]
    _print_rank_lines(f"[all_gather] rank={rank}: input={tensor.tolist()}, output={result}")


def scatter() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.empty(2, dtype=torch.int64, device="cpu")
    chunks = None
    messages = []
    if rank == 0:
        chunks = [
            torch.tensor([index + 1, 10 * (index + 1)], device="cpu")
            for index in range(world_size)
        ]
        messages.append(f"[scatter] rank=0: chunks={[item.tolist() for item in chunks]}")

    # src 的第 i 个张量发给 rank=i；其他 rank 不提供 scatter_list。
    dist.scatter(tensor, scatter_list=chunks, src=0)

    messages.append(f"[scatter] rank={rank}: output={tensor.tolist()}")
    _print_rank_lines(*messages)


def all_to_all() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.arange(rank * world_size, (rank + 1) * world_size, device="cpu")
    received = torch.empty_like(tensor)

    # 按第 0 维均分，每个目标 rank 一个元素；接收结果按来源 rank 排列。
    dist.all_to_all_single(received, tensor)

    _print_rank_lines(f"[all_to_all] rank={rank}: input={tensor.tolist()}, output={received.tolist()}")


def main() -> None:
    examples = {
        "broadcast": broadcast,
        "reduce": reduce,
        "all_reduce": all_reduce,
        "reduce_scatter": reduce_scatter,
        "gather": gather,
        "all_gather": all_gather,
        "scatter": scatter,
        "all_to_all": all_to_all,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["all", *examples], default="all", help="默认依次运行全部算子")
    args = parser.parse_args()

    # torchrun 提供 RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT；无需手动分配 rank。
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=60))
    try:
        selected = examples.values() if args.op == "all" else [examples[args.op]]
        # 每个 rank 必须以相同顺序调用集合通信，不能只让 rank=0 调用。
        for example in selected:
            example()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
