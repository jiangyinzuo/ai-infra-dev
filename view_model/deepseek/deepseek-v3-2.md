
## Lighting Indexer算子在NPU上的性能问题

lighting indexer算子，NPU不亲和，耗时比较大。
使用PCP，切多份输入，弥补NPU上lighting indexer性能差的缺点。

### PCP

模型在 Prefill 阶段的大部分计算（如 RMSNorm, FFN, Embedding 等）都是
Token-Wise（逐 Token） 的，彼此独立，完全可以分开计算。只有 Attention 模块需要
Token 间的交互，而 PCP 通过特定的通信机制解决了这个问题。

## DeepSeek Sparse Attention

稀疏注意力，改变的是每个Query token在计算注意力时，所能访问的Key Value候选集合的大小。

记上下文token序列长度为$N$。Dense Attention可以看到全部历史KV，注意力矩阵是稠密的，计算复杂度$\mathcal{O}(N^2)$。
但在DeepSeek Sparse Attention中，被大幅缩减为Top-K个（$k=2048$）。模型先经过lightning indexer给所有历史token打分，然后只保留分数最高的 $k$ 个参与后续的核心注意力计算。 这意味着注意力矩阵变得稀疏，计算复杂度降至$\mathcal{O}(N\cdot k)$。

> 当上下文长度 $N$ = 128K 时，$k$ = 2048 仅占 $N$ 的 1.6%，计算量理论上可减少约 98%。

## DSA数学公式


| 符号 | 含义 |
| --- | --- |
| $t$、$s$ | 当前 Query 位置、被访问的 KV 位置；因果可见范围为 $1\le s\le t$，包含当前位置 |
| $\mathbf{h}_t\in\mathbb{R}^{d}$ | 当前层输入的 token 隐状态 |
| $i=1,\ldots,n_h$ | MLA 的注意力头编号 |
| $d_h$、$d_h^R$ | MLA 每头内容维度、解耦 RoPE 维度 |
| $\mathbf{c}_t^Q\in\mathbb{R}^{d_c'}$、$\mathbf{c}_s^{KV}\in\mathbb{R}^{d_c}$ | MLA 的 Query 压缩潜向量、KV 联合压缩潜向量 |
| 上标 $C$、$R$、$I$ | 分别表示内容、RoPE 、Indexer 相关变量；$I$ 不是 MLA 的头编号 |
| $j=1,\ldots,H^I$、$d^I$ | indexer 的头编号、每头维度，与 MLA 的 $i,n_h,d_h$ 区分 |
| $I_{t,s}$、$\mathcal{S}_t$、$k$ | 索引分数、被选位置集合、选择数量上限；V3.2 使用 $k=2048$ |

### lightning indexer

$$
\boxed{
I_{t,s}=\sum_{j=1}^{H^I}
w_{t,j}^I\cdot
\operatorname{ReLU}\!\left(
\mathbf{q}_{t,j}^I\cdot\mathbf{k}_s^I
\right)
}
$$

绿色表示 lightning indexer 自身的可学习参数。下列符号为[官方实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)中参数的简记：

| 权重 | 尺寸 | 官方参数 |
| --- | --- | --- |
| ${\color{green}W^{IQ}}$ | $H^Id^I\times d_c'=8192\times1536$ | `indexer.wq_b.weight` |
| ${\color{green}W^{IK}}$ | $d^I\times d=128\times7168$ | `indexer.wk.weight` |
| ${\color{green}W^{IW}}$ | $H^I\times d=64\times7168$ | `indexer.weights_proj.weight` |
| ${\color{green}\boldsymbol{\gamma}^I}$ | $d^I=128$ | `indexer.k_norm.weight`，Key 的 LayerNorm 缩放 |
| ${\color{green}\boldsymbol{\beta}^I}$ | $d^I=128$ | `indexer.k_norm.bias`，Key 的 LayerNorm 偏置 |

