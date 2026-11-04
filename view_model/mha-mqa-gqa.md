---
title: MHA、MQA、GQA 通用计算流程
---

沿用 [LLaMA Attention](./llama2_7b/llama_attention.md) 中的 $X,n,d,H,d_k,d_v,Q,K,V,M,W_O$ 等符号。三种 attention 的计算流程相同，区别在于：**多少个 Query 头共享同一组 Key、Value 头**。

## 统一符号

先假设 Batch size $B=1$，讨论长度为 $n$ 的 causal self-attention；忽略投影偏置和 dropout。

| 符号 | 含义 |
|---|---|
| $X=[x_1,x_2,\dots,x_n]\in\mathbb{R}^{n\times d}$ | 输入序列的 hidden states |
| $n$ | 序列长度，即 token 个数 |
| $d$ | 模型隐藏维度（hidden size） |
| $H$ | Query 头数，沿用参考文档中 attention 头数的记号 |
| $H_{\mathrm{kv}}$ | Key 头数，也是 Value 头数 |
| $d_k$ | 每个 Query、Key 头的维度，两者必须相同才能做点积 |
| $d_v$ | 每个 Value 头的维度，也是每个 attention 头的输出维度 |
| $r=H/H_{\mathrm{kv}}$ | 每组 KV 被多少个 Query 头共享 |

通常采用：

$$
d_k=d_v=\frac{d}{H}.
$$

这里分母是 **Query 头数 $H$**，不是 $H_{\mathrm{kv}}$；减少 KV 头数不意味着增大每个头的维度。

假设 $1\le H_{\mathrm{kv}}\le H$ 且 $H_{\mathrm{kv}}$ 整除 $H$，按连续、等大小分组。对第 $h$ 个 Query 头（$h=1,\dots,H$），定义它使用的 KV 头编号：

$$
g(h)=\left\lfloor\frac{h-1}{r}\right\rfloor+1,
\qquad r=\frac{H}{H_{\mathrm{kv}}}.
$$

即第 $\ell$ 组 KV 头供 Query 头 $(\ell-1)r+1,\dots,\ell r$ 使用，其中 $\ell=1,\dots,H_{\mathrm{kv}}$。

## 统一计算流程

### 1. Q、K、V 投影并拆分为多个头

完整的投影矩阵为：

$$
\begin{aligned}
W_Q&=[W_Q^{(1)}\mid\cdots\mid W_Q^{(H)}]
\in\mathbb{R}^{d\times(Hd_k)},\\
W_K&=[W_K^{(1)}\mid\cdots\mid W_K^{(H_{\mathrm{kv}})}]
\in\mathbb{R}^{d\times(H_{\mathrm{kv}}d_k)},\\
W_V&=[W_V^{(1)}\mid\cdots\mid W_V^{(H_{\mathrm{kv}})}]
\in\mathbb{R}^{d\times(H_{\mathrm{kv}}d_v)}.
\end{aligned}
$$

先做矩阵乘法：

$$
\begin{aligned}
Q&=XW_Q\in\mathbb{R}^{n\times(Hd_k)},\\
K&=XW_K\in\mathbb{R}^{n\times(H_{\mathrm{kv}}d_k)},\\
V&=XW_V\in\mathbb{R}^{n\times(H_{\mathrm{kv}}d_v)}.
\end{aligned}
$$

再沿输出特征维拆分为各个头，等价于：

$$
\begin{aligned}
Q_h&=XW_Q^{(h)}\in\mathbb{R}^{n\times d_k},
&&h=1,\dots,H,\\
K_\ell&=XW_K^{(\ell)}\in\mathbb{R}^{n\times d_k},
&&\ell=1,\dots,H_{\mathrm{kv}},\\
V_\ell&=XW_V^{(\ell)}\in\mathbb{R}^{n\times d_v},
&&\ell=1,\dots,H_{\mathrm{kv}}.
\end{aligned}
$$

每个 Query 头都有独立的投影参数；同组 Query 头使用同一个 $K_\ell,V_\ell$，而不是把多个头的结果求平均。

### 2. 加入位置编码（使用 RoPE）

先按每个 token 的位置旋转 Q、K：

$$
\widetilde{Q}_h=\mathrm{RoPE}(Q_h),\qquad
\widetilde{K}_\ell=\mathrm{RoPE}(K_\ell).
$$

RoPE 不改变形状，也不作用于 $V_\ell$。

### 3. 每个 Query 头独立计算注意力分数

第 $h$ 个 Query 头使用第 $g(h)$ 个 Key 头：

$$
S_h=\frac{\widetilde{Q}_h\widetilde{K}_{g(h)}^\top}{\sqrt{d_k}}+M
\in\mathbb{R}^{n\times n}.
$$

缩放因子是 $\sqrt{d_k}$，不是 $\sqrt{d}$。其中 causal mask 为：

$$
M_{ij}=\begin{cases}
0, & j\le i,\\
-\infty, & j>i,
\end{cases}
\qquad M\in\mathbb{R}^{n\times n}.
$$

$S_h$ 的第 $i$ 行表示第 $i$ 个 Query token 对所有 Key token 的打分；所有头可以共享同一个 mask。

### 4. 沿 Key 维做 softmax，再对 Value 加权求和

#### 矩阵写法
$$
A_h=\mathrm{softmax}(S_h)\in\mathbb{R}^{n\times n},
\qquad
(A_h)_{ij}=\frac{\exp((S_h)_{ij})}{\sum_{t=1}^{n}\exp((S_h)_{it})}.
$$

softmax 沿每一行的 Key token 维归一化，不沿头维归一化；被 mask 的位置权重为 $0$。

$$
\text{head}_h=A_hV_{g(h)}\in\mathbb{R}^{n\times d_v}.
$$

对第 $i$ 个 token，等价地写成：

$$
(\text{head}_h)_{i,:}
=\sum_{j=1}^{n}(A_h)_{ij}(V_{g(h)})_{j,:}.
$$

一个Query头对应一个$Q_h$、$S_h$、$A_h$（Softmax输出） 、$\text{head}_h$。

#### 向量写法

固定第 $h$ 个 Query 头，取第 $i$ 个 token 作为 query。为简化记号，以下省略向量的头下标，将 RoPE 后矩阵的对应行记为：

$$
\begin{aligned}
q_i&:=(\widetilde{Q}_h)_{i,:}\in\mathbb{R}^{1\times d_k},\\
k_j&:=(\widetilde{K}_{g(h)})_{j,:}\in\mathbb{R}^{1\times d_k},
\qquad j=1,\dots,n.
\end{aligned}
$$

打分向量就是 $S_h$ 的第 $i$ 行，包含缩放点积与 causal mask：

$$
\begin{aligned}
s_i&:=(S_h)_{i,:}\\
&=\left[
\frac{q_i k_1^\top}{\sqrt{d_k}},\
\frac{q_i k_2^\top}{\sqrt{d_k}},\
\dots,\
\frac{q_i k_n^\top}{\sqrt{d_k}}
\right]+M_{i,:}
\in\mathbb{R}^{1\times n}.
\end{aligned}
$$

对这一行做 softmax，得到 $A_h$ 的对应行：

$$
\alpha_i:=(A_h)_{i,:}=\mathrm{softmax}(s_i)
\in\mathbb{R}^{1\times n}.
$$

$$
\alpha_{ij}:=(\alpha_i)_j=(A_h)_{ij},
\qquad \sum_{j=1}^{n}\alpha_{ij}=1,
\qquad \alpha_{ij}=0\quad(j>i).
$$

再用该行权重对 Value 加权求和：

$$
(\text{head}_h)_{i,:}=\alpha_i V_{g(h)}
\in\mathbb{R}^{1\times d_v}.
$$

### 5. 拼接所有 Query 头的输出，再做输出投影

$$
C=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_H)
\in\mathbb{R}^{n\times(Hd_v)}.
$$

$$
Y=CW_O\in\mathbb{R}^{n\times d},
\qquad W_O\in\mathbb{R}^{(Hd_v)\times d}.
$$

这里拼接的始终是 **$H$ 个 Query 头的输出**，不是 $H_{\mathrm{kv}}$ 个。$W_O$ 对不同头的输出做线性混合；残差连接、归一化不包含在上述 $Y$ 中。

## 三种 attention 的具体形式

以下都使用上述相同的投影、位置编码、mask、softmax 和输出投影流程，只改变 KV 头数与映射 $g(h)$。

以下尺寸均按**单层 attention** 统计：模型权重只包含 $W_Q,W_K,W_V,W_O$，不包含 MLP、Embedding 等其他模块，也不含偏置。为便于直接比较，统一采用 $d_k=d_v=d/H$。

KV Cache 恢复 batch 维 $B$，假设每条序列均已缓存 $n$ 个 token（包含历史 token 与当前已写入的 token），只统计 K、V 的逻辑尺寸，不含分页、对齐等实现开销。$N_{\mathrm{weights}}$ 表示权重参数总数，$N_{\mathrm{KV}}$ 表示 K、V 缓存元素总数；换算字节占用时，分别乘以对应数据类型的每元素字节数。

三者均按下式计算第 $h$ 个头，只需代入表中的 $g(h)$：

$$
\text{head}_h
=\mathrm{softmax}\left(
\frac{\widetilde{Q}_h\widetilde{K}_{g(h)}^\top}{\sqrt{d_k}}+M
\right)V_{g(h)},
\qquad h=1,\dots,H.
$$

| 对比项 | **MHA（多头注意力）** | **MQA（多查询注意力）** | **GQA（分组查询注意力）** |
|---|---|---|---|
| 英文全称 | Multi-Head Attention | Multi-Query Attention | Grouped-Query Attention |
| Query 头数 | $H$ | $H$ | $H$ |
| KV 头数 $H_{\mathrm{kv}}$ | $H$ | $1$ | $1<H_{\mathrm{kv}}<H$ |
| KV 共享方式 | 每个 Query 头独享一组 KV | 所有 Query 头共享一组 KV | 每组 Query 头共享一组 KV |
| 每组 Query 头数 $r$ | $1$ | $H$ | $H/H_{\mathrm{kv}}$ |
| Query 到 KV 的映射 $g(h)$ | $h$ | $1$ | $\left\lfloor\frac{h-1}{r}\right\rfloor+1$ |
| $\textcolor{green}{W_Q},\textcolor{green}{W_O}$ 的形状 | $d\times d$ | $d\times d$ | $d\times d$ |
| $\textcolor{green}{W_K},\textcolor{green}{W_V}$ 的形状 | $d\times d$ | $d\times\frac{d}{H}$ | $d\times\left(H_{\mathrm{kv}}\frac{d}{H}\right)$ |
| 权重参数总数 $N_{\mathrm{weights}}$ | $4d^2$ | $2d^2\left(1+\frac{1}{H}\right)$ | $2d^2\left(1+\frac{H_{\mathrm{kv}}}{H}\right)$ |
| 权重参数量 / MHA | $1$ | $\frac{1}{2}\left(1+\frac{1}{H}\right)$ | $\frac{1}{2}\left(1+\frac{1}{r}\right)$ |
| $\textcolor{blue}{K_{\mathrm{cache}}},\textcolor{blue}{V_{\mathrm{cache}}}$ 的形状 | $B\times H\times n\times\frac{d}{H}$ | $B\times1\times n\times\frac{d}{H}$ | $B\times H_{\mathrm{kv}}\times n\times\frac{d}{H}$ |
| KV Cache 元素总数 $N_{\mathrm{KV}}$ | $2Bnd$ | $\frac{2Bnd}{H}$ | $2Bnd\frac{H_{\mathrm{kv}}}{H}$ |
| KV Cache 元素数 / MHA | $1$ | $\frac{1}{H}$ | $\frac{H_{\mathrm{kv}}}{H}=\frac{1}{r}$ |

GQA中，令$H_{\mathrm{kv}}=H$ 就退化为 MHA，令 $H_{\mathrm{kv}}=1$ 就退化为 MQA。

MHA中，$H=1$就是单头注意力。

MQA、GQA 缩减的是 $W_K$,$W_V$,$K_{cache}$,$V_{cache}$；$W_Q,W_O$ 不变。

## 带 Batch 维的张量形状

加入 Batch 维 $B$ 后，将投影结果按头拆分、转置到 head-first 布局，逻辑形状如下：

| 张量 | 形状 |
|---|---|
| 输入 $X$ | $B\times n\times d$ |
| 按头排列后的 $Q$、$\widetilde{Q}$ | $B\times H\times n\times d_k$ |
| 按头排列后的 $K$、$\widetilde{K}$ | $B\times H_{\mathrm{kv}}\times n\times d_k$ |
| 按头排列后的 $V$ | $B\times H_{\mathrm{kv}}\times n\times d_v$ |
| 所有 Query 头的 $S$、$A$ | $B\times H\times n\times n$ |
| 拼接后的 $C$ | $B\times n\times(Hd_v)$ |
| 最终输出 $Y$ | $B\times n\times d$ |

同一 causal mask 可用 $1\times1\times n\times n$ 的逻辑形状广播到所有 batch、Query 头。按 $g(h)$ 访问共享的 K、V 即可，**数学上不要求把 KV 实际复制到 $H$ 个头**；$S$、$A$ 的形状也只是逻辑形状，实现中不一定完整存储。
