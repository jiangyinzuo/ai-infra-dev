#!/usr/bin/env python3
"""复算 Lightning Indexer / Full MLA / DSA 的理论前向 FLOPs。

默认配置与 deepseek-v3-v32-glm5-2.md 中的对比表一致：单层、单序列，
完整因果 Prefill；Decode 的可见 KV 数包含当前 token。1 MAC = 2 FLOPs。
主表只计有效位置对上的 QK 点积与 Value 聚合，不是稠密 mask 参考实现
实际执行的指令数，也不是实测耗时或 FLOP/s。投影单独输出，其他归一化、
ReLU/Softmax、Hadamard、量化、Top-K、访存和通信不计入主表。

运行：
    python3 lightning_indexer_flops.py
    python3 lightning_indexer_flops.py --seq-len 32768 --top-k 2048
"""

import argparse


# DeepSeek-V3.2 的模型维度；修改模型配置时在这里调整。
HIDDEN_DIM = 7168
QUERY_LATENT_DIM = 1536
KV_LATENT_DIM = 512
MLA_HEADS = 128
CONTENT_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128


def causal_pairs(seq_len: int) -> int:
    """P = sum(t), t = 1..n，包含当前位置。"""
    return seq_len * (seq_len + 1) // 2


def sparse_pairs(seq_len: int, top_k: int) -> int:
    """S = sum(min(t, k))；n <= k 时退化为完整因果注意力。"""
    if seq_len <= top_k:
        return causal_pairs(seq_len)
    return seq_len * top_k - top_k * (top_k - 1) // 2


def verify_pair_counts() -> int:
    """用逐项枚举验证闭式计数，覆盖 k=1、n=k、n<k、n>k。"""
    cases = 0
    for seq_len in (1, 2, 7, 16, 31):
        for top_k in (1, 3, 16, 32):
            visible_pairs = [(t, j) for t in range(1, seq_len + 1)
                             for j in range(1, t + 1)]
            selected_pairs = [(t, j) for t in range(1, seq_len + 1)
                              for j in range(1, min(t, top_k) + 1)]
            assert causal_pairs(seq_len) == len(visible_pairs)
            assert sparse_pairs(seq_len, top_k) == len(selected_pairs)
            cases += 1
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=128 * 1024,
                        help="Prefill 长度 / Decode 可见 KV 数，默认 131072")
    parser.add_argument("--top-k", type=int, default=2048,
                        help="每个 Query 的选择上限，默认 2048")
    args = parser.parse_args()
    if args.seq_len < 1 or args.top_k < 1:
        parser.error("--seq-len 和 --top-k 必须为正整数")

    checked = verify_pair_counts()
    n, k = args.seq_len, args.top_k
    p = causal_pairs(n)
    s = sparse_pairs(n, k)
    m = min(n, k)

    # 每个有效位置对的 MAC 系数。Indexer 没有 Value 分支。
    index_mac = INDEX_HEADS * INDEX_HEAD_DIM
    expanded_mac = MLA_HEADS * (2 * CONTENT_HEAD_DIM + ROPE_HEAD_DIM)
    absorbed_mac = MLA_HEADS * (2 * KV_LATENT_DIM + ROPE_HEAD_DIM)

    full_expanded = (2 * p * expanded_mac, 2 * n * expanded_mac)
    full_absorbed = (2 * p * absorbed_mac, 2 * n * absorbed_mac)
    indexer = (2 * p * index_mac, 2 * n * index_mac)
    sparse_absorbed = (2 * s * absorbed_mac, 2 * m * absorbed_mac)
    dsa = (indexer[0] + sparse_absorbed[0],
           indexer[1] + sparse_absorbed[1])

    rows = (
        ("Full Attention: MLA 展开路径", full_expanded),
        ("Full Attention: MLA 吸收路径", full_absorbed),
        ("Lightning Indexer 全历史点积", indexer),
        ("Sparse MLA: 吸收路径", sparse_absorbed),
        ("DSA: Indexer + Sparse MLA 吸收路径", dsa),
    )
    print(f"枚举核验通过：{checked} 组；1 MAC = 2 FLOPs")
    print(f"n={n}, k={k}, m=min(n,k)={m}")
    print(f"主 MLA: heads={MLA_HEADS}, 内容维度={CONTENT_HEAD_DIM}, "
          f"RoPE={ROPE_HEAD_DIM}, KV 潜在维度={KV_LATENT_DIM}")
    print(f"Indexer: heads={INDEX_HEADS}, 每头维度={INDEX_HEAD_DIM}")
    print(f"P=n(n+1)/2={p}; S=sum(min(t,k))={s}")
    print("\n主表仅计点积与 Value 聚合；TFLOPs / GFLOPs 是运算量，不是吞吐率。")
    print("| 计算部分 | 完整因果 Prefill（TFLOPs） | 单 token Decode（GFLOPs） |")
    print("| --- | --- | --- |")
    for label, (prefill, decode) in rows:
        print(f"| {label} | {prefill / 1e12:.3f} | {decode / 1e9:.3f} |")

    # 额外三条线性投影：Q 上投影、共享 K 投影、动态头权重投影。
    # MLA 已经计算的 Query 下投影和 RMSNorm 不重复计入 indexer。
    projection_per_token = 2 * (
        QUERY_LATENT_DIM * INDEX_HEADS * INDEX_HEAD_DIM
        + HIDDEN_DIM * INDEX_HEAD_DIM
        + HIDDEN_DIM * INDEX_HEADS
    )
    print("\nIndexer 额外投影 C_I = "
          "2 * (d_c' * H_I * d_I + d * d_I + d * H_I)")
    print(f"C_I={projection_per_token} FLOPs/token; "
          f"完整 Prefill 投影={n * projection_per_token / 1e12:.6f} TFLOPs")
    print("完整因果 Prefill 的选中位置对占比 S/P："
          f"{s / p:.6%}")
    print(f"当前 Decode 的选中 KV 占比 m/n：{m / n:.6%}")
    print("Indexer 点积 / Full MLA 吸收路径主乘加："
          f"{index_mac / absorbed_mac:.6%}")
    print("DSA / 同路径 Full MLA 主乘加："
          f"Prefill={dsa[0] / full_absorbed[0]:.6%}, "
          f"Decode={dsa[1] / full_absorbed[1]:.6%}")


if __name__ == "__main__":
    main()
