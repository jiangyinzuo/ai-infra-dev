
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
| $\mathbf{h}_t\in\mathbb{R}^{d}$ | Attention / Indexer 的输入隐状态，已过当前层的 attention 前置 RMSNorm |
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

#### $\mathbf q^I$、$\mathbf k^I$ 的构造

来源：[报告 §2.1，式 (1)、Figure 2](https://arxiv.org/html/2512.02556v1#S2.SS1)；[官方 `MLA.forward` / `Indexer.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)；[HF `DeepseekV32Attention.forward` / `DeepseekV32Indexer.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py)。以下为实现的数学展开，忽略浮点舍入。

**投影与归一化**

$$
\begin{aligned}
\mathbf z_t^Q &= W^{DQ}\mathbf h_t,\\
\mathbf c_t^Q
&=\operatorname{RMSNorm}_{\boldsymbol\gamma^Q}(\mathbf z_t^Q)
=\boldsymbol\gamma^Q\odot
\frac{\mathbf z_t^Q}{\sqrt{\|\mathbf z_t^Q\|_2^2/d_c'+\varepsilon_Q}}
\in\mathbb R^{d_c'},\\
\widetilde{\mathbf q}_{t,j}^I
&=\left[\operatorname{reshape}_{H^I\times d^I}
\!\left({\color{green}W^{IQ}}\mathbf c_t^Q\right)\right]_{j,:},\\
\mathbf z_s^I&={\color{green}W^{IK}}\mathbf h_s,\\
\mu_s&=\frac{1}{d^I}\sum_{a=1}^{d^I}z_{s,a}^I,
\qquad
\sigma_s^2=\frac{1}{d^I}\sum_{a=1}^{d^I}(z_{s,a}^I-\mu_s)^2,\\
\widetilde{\mathbf k}_s^I
&=\operatorname{LayerNorm}_{{\color{green}\boldsymbol\gamma^I},{\color{green}\boldsymbol\beta^I}}(\mathbf z_s^I)
={\color{green}\boldsymbol\gamma^I}\odot
\frac{\mathbf z_s^I-\mu_s\mathbf 1}{\sqrt{\sigma_s^2+10^{-6}}}
+{\color{green}\boldsymbol\beta^I}
\in\mathbb R^{d^I}.
\end{aligned}
$$

| 符号 | 官方实现 | HF 实现 |
| --- | --- | --- |
| $W^{DQ}$、$\boldsymbol\gamma^Q$（MLA 参数，非 indexer 独有） | `wq_a.weight`、`q_norm.weight` | `q_a_proj.weight`、`q_a_layernorm.weight` |
| $\mathbf c_t^Q$（与 MLA 共享） | `qr` | `q_resid` |
| $\widetilde{\mathbf q}_{t,j}^I$ | `indexer.wq_b(qr)` 后按 head 拆分 | `indexer.wq_b(q_resid)` 后按 head 拆分 |
| $\widetilde{\mathbf k}_s^I$（所有 indexer heads 共享） | `indexer.k_norm(indexer.wk(x))` | `indexer.k_norm(indexer.wk(hidden_states))` |

**部分 RoPE：前 $r=d_h^R=64$ 维旋转，后 $d^I-r=64$ 维直通**

令 $m=r/2$，$C_p=\operatorname{diag}(\mathrm{cos}_p[:m])$、$S_p=\operatorname{diag}(\mathrm{sin}_p[:m])$；$\mathrm{cos}_p,\mathrm{sin}_p$ 取实现的位置编码。Half-split 配对 $(a,a+m)$：

$$
R_p^I=
\begin{bmatrix}C_p&-S_p\\S_p&C_p\end{bmatrix},
\qquad
P_p^I=\operatorname{diag}(R_p^I,\operatorname{Id}_{d^I-r}),
\qquad
\boxed{
\mathbf q_{t,j}^I=P_t^I\widetilde{\mathbf q}_{t,j}^I,
\quad
\mathbf k_s^I=P_s^I\widetilde{\mathbf k}_s^I
}.
$$

官方：`apply_rotary_emb(..., interleaved=False)`；HF：`apply_rotary_pos_emb(..., unsqueeze_dim=2)`。缓存保存位置变换后的 indexer Key（官方另做下述量化）。

**与评分式的缩放对齐**

$$
\begin{aligned}
a_{t,j}^I&=\left[{\color{green}W^{IW}}\mathbf h_t\right]_j,
\qquad
w_{t,j}^I:=\frac{a_{t,j}^I}{\sqrt{H^I d^I}},\\
I_{t,s}^{\mathrm{HF}}
&=\sum_j\frac{a_{t,j}^I}{\sqrt{H^I}}
\operatorname{ReLU}\!\left(
\frac{(\mathbf q_{t,j}^I)^\top\mathbf k_s^I}{\sqrt{d^I}}
\right)
=\sum_j w_{t,j}^I
\operatorname{ReLU}\!\left((\mathbf q_{t,j}^I)^\top\mathbf k_s^I\right).
\end{aligned}
$$

**官方额外的 Hadamard / FP8 路径**

$$
A=\frac{\operatorname{Hadamard}_{d^I}}{\sqrt{d^I}},
\qquad A^\top A=\operatorname{Id},
\qquad
(A\mathbf q)^\top(A\mathbf k)=\mathbf q^\top\mathbf k.
$$

$$
\widehat{\mathbf q}_{t,j}^I
=\mathcal Q_{\mathrm{FP8}}(A\mathbf q_{t,j}^I),
\qquad
\widehat{\mathbf k}_s^I
=\mathcal Q_{\mathrm{FP8}}(A\mathbf k_s^I),
\qquad
I_{t,s}^{\mathrm{official}}
\approx I_{t,s}^{\mathrm{HF}}.
$$

$\mathcal Q_{\mathrm{FP8}}$ 表示含 scale 的量化—反量化有效值；官方执行 `rotate_activation → act_quant → fp8_index`，HF 省略这三步，以模型 dtype 投影、FP32 点积评分；量化后不保证分数或 Top-K 完全相同。

