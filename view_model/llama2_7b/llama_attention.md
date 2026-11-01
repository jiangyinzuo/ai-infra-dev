# LLAMA Attention

## LLaMA Attention 公式

输入序列：$X=[x_1,x_2,\dots,x_n]\in\mathbb{R}^{n\times d}$

- **$n$**：序列长度，也就是 token 的个数；
- **$d$**：模型隐藏维度，也就是每个 token 向量的维度（hidden size）。

投影得到：$Q=XW_Q,\quad K=XW_K,\quad V=XW_V$

### 单头注意力

$$
\text{Attn}(Q,K,V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

其中 $M$ 是 causal mask：
$$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

### 多头注意力

对第 $h$ 个头（$h=1,\dots,H$）：

$$
Q_h=XW_Q^{(h)},\quad K_h=XW_K^{(h)},\quad V_h=XW_V^{(h)}
$$

每个头维度通常为：$d_k=\frac{d}{H}$

第 $h$ 个头的输出：
$$
\text{head}_h
=
\text{softmax}\left(
\frac{Q_hK_h^T}{\sqrt{d_k}}+M
\right)V_h
$$

其中 $M$ 是 causal mask。

多头拼接后：

$$
\text{MultiHead}(X)
=
\text{Concat}(\text{head}_1,\dots,\text{head}_H)W_O
$$

如果有 RoPE，则先对每个头的 \(Q_h,K_h\) 做位置旋转，再算 attention：
$$
\text{head}_h
=
\text{softmax}\left(
\frac{\text{RoPE}(Q_h)\text{RoPE}(K_h)^T}{\sqrt{d_k}}+M
\right)V_h
$$

## 没有RoPE的后果

\[
\text{score}_{ij}
=
\frac{q_i^T k_j}{\sqrt{d_k}}
=
\frac{(x_iW_Q)^T(x_jW_K)}{\sqrt{d_k}}
\]

这个分数**只依赖 \(x_i\) 和 \(x_j\) 的内容，完全不依赖 \(i\) 和 \(j\)**。位置 \(i,j\) 没有以任何形式进入公式。

---

## 2. 核心问题：Attention 是置换等变的

假设没有 causal mask，对输入做任意置换 \(P\)（打乱 token 顺序）：

\[
X'=PX
\]

那么：

\[
Q'=PXW_Q=PQ
\]

\[
K'=PK,\quad V'=PV
\]

代入注意力：

\[
\text{Attn}(Q',K',V')
=
\text{softmax}\left(\frac{PQK^TP^T}{\sqrt{d_k}}\right)PV
=
P\cdot\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
=
P\cdot\text{Attn}(Q,K,V)
\]

即：

\[
\boxed{
\text{Attn}(PX)=P\cdot\text{Attn}(X)
}
\]

这叫**置换等变**：打乱输入顺序，输出只是被同样打乱，内容完全不变。

**这意味着：模型对“顺序”完全无感。** “猫追狗”和“狗追猫”在模型眼里是同一个东西，只是排列不同。

---

## 3. Causal mask 能救多少？

LLaMA 是 decoder-only，有 causal mask，情况稍微好一点，但远远不够。

Causal mask 只做了这件事：

```text
位置 i 只能 attend 到 j ≤ i
```

它提供了“**谁在谁前面**”这个方向信息，但**没有提供“隔多远”**。

看 attention score：

\[
\text{score}_{ij}
=
\frac{q_i^Tk_j}{\sqrt{d_k}}
\]

即使有 mask，这个分数仍然只取决于内容。mask 只是把 \(j>i\) 的项置为 \(-\infty\)，对 \(j\le i\) 的项**没有任何位置相关的修正**。

具体后果：

### 后果一：无法表达“距离衰减”

在自然语言里，邻近 token 通常更相关。理想情况下：

\[
\text{score}_{ij}\ \text{应该随}\ |i-j|\ \text{增大而衰减}
\]

但没有位置编码时，\(\text{score}_{ij}\) 对 \(|i-j|\) 没有任何偏好。模型无法表达“位置 5 的 token 对位置 6 的关注应该强于对位置 100 的关注”这种先验。

### 后果二：无法区分“相同内容、不同距离”

假设 \(q_i\) 和两个 key \(k_j,k_{j'}\) 内容完全相同：

\[
k_j=k_{j'}=k
\]

那么：

\[
\text{score}_{ij}=\text{score}_{ij'}
\]

不管 \(j\) 离 \(i\) 是 1 步还是 100 步，attention 权重完全一样。模型没有任何机制区分“刚才说的词”和“很久以前说的词”。

### 后果三：语序信息丢失

考虑两个句子：

```text
句子 A：狗 咬 人
句子 B：人 咬 狗
```

Token 集合相同，只是顺序不同。在没有位置编码的 causal attention 下：

- 位置 3 的 token 在 A 中是“人”，在 B 中是“狗”；
- 它能看到的前面 token 集合分别是 {狗,咬,人} 和 {人,咬,狗}；
- 但 attention score 只比较内容，不比较位置。

模型很难从 attention 本身学到“主语在动词前、宾语在动词后”这种语序模式。它只能依赖 causal mask 提供的“可见范围”差异，这远远不足以表达完整的语序。

---

## 4. 一个具体数值例子

取极简设定：\(d_k=2\)，所有 token 的 \(q,k\) 都相同：

\[
q_i=(1,0),\quad k_j=(1,0)\quad \forall i,j
\]

则：

\[
q_i^Tk_j=1\quad \forall i,j
\]

经过 causal mask 后，位置 \(i\) 的 attention 权重是：

\[
\alpha_{ij}
=
\frac{\exp(1)}{\sum_{j'\le i}\exp(1)}
=
\frac{1}{i}
\quad (j\le i)
\]

也就是说，位置 \(i\) 对**所有可见位置均匀分配注意力**：

```text
位置 5 的 attention：
  j=1   j=2   j=3   j=4   j=5
  1/5   1/5   1/5   1/5   1/5
```

模型完全不知道“j=4 比 j=1 更近”。如果语言模型需要“最近说的词更重要”，它没有任何归纳偏置可以利用，只能完全从数据里硬学。

---

## 5. 那为什么不能“让模型自己学位置”？

理论上，模型可以通过 \(W_Q,W_K\) 学到一些位置相关的东西吗？**不能**，因为 \(Q=XW_Q\) 中 \(X\) 的每一行只包含该 token 的内容，不包含它的位置。同一个 token 出现在位置 1 和位置 100，它的 \(q\) 向量完全一样。

没有位置编码，模型**没有任何入口**把位置信息注入 Q/K。这不是“学得好不好的问题”，而是“信息根本不存在”。

---

## 6. RoPE 补上了什么？

RoPE 在计算 attention score 之前，对 Q/K 做位置相关的旋转：

\[
q_i'=R_i q_i,\quad k_j'=R_j k_j
\]

其中 \(R_i\) 由位置 \(i\) 决定。于是：

\[
\text{score}_{ij}
=
\frac{(R_iq_i)^T(R_jk_j)}{\sqrt{d_k}}
=
\frac{q_i^TR_i^TR_jk_j}{\sqrt{d_k}}
=
\frac{q_i^TR_{j-i}k_j}{\sqrt{d_k}}
\]

现在 score **显式依赖 \(j-i\)**：

- 距离近，旋转差小；
- 距离远，旋转差大；
- 模型可以通过学习 \(W_Q,W_K\) 来决定“哪些内容维度对距离敏感”。

对比一下：

| | 无位置编码 | 有 RoPE |
|---|---|---|
| score 公式 | \(q_i^Tk_j\) | \(q_i^TR_{j-i}k_j\) |
| 是否依赖位置 | 否 | 是，且只依赖相对位置 |
| 能否表达距离衰减 | 不能 | 可以 |
| 置换等变 | 是 | 否 |
| 语序信息 | 几乎没有 | 有 |

---

## 7. 一句话总结

LLaMA 的 Attention 公式

\[
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

中，\(QK^T\) 只比较内容，\(M\) 只提供“可见/不可见”。没有 RoPE，模型**没有任何机制知道 token 的顺序和距离**，attention 对置换等变，无法表达“邻近更重要”“语序不同语义不同”这些基本语言规律。RoPE 通过对 Q/K 做位置相关旋转，把相对位置 \(j-i\) 注入点积，才让 Attention 真正“看得见顺序”。

## RoPE

### 二维向量的旋转矩阵

假设有一个二维向量：$\mathbf{v} = (x, y)$，向量长度是$r = \sqrt{x^2+y^2}$。

现在把这个向量绕原点逆时针旋转 $\theta$ 角，得到新向量：$\mathbf{v}' = (x', y')$


旋转过程中有两个重要性质：
1. **长度不变**：$|\mathbf{v}'| = |\mathbf{v}| = r$。
2. **方向变了**：原来的角度加上 $\theta$，变成新的角度。

#### 旋转公式推导

原来的向量可以写成：$x = r\cos\alpha$, $y = r\sin\alpha$，
其中 $\alpha$ 是原向量和 x 轴的夹角。

旋转 $\theta$ 后，新角度是：$\alpha + \theta$。

所以，
$x' = r\cos(\alpha+\theta)$，$y' = r\sin(\alpha+\theta)$

用三角和角公式展开：

$$
x' = r(\cos\alpha\cos\theta - \sin\alpha\sin\theta)
y' = r(\sin\alpha\cos\theta + \cos\alpha\sin\theta)
$$

代入：$x = r\cos\alpha,\quad y = r\sin\alpha$

得到：

$$
x' = x\cos\theta - y\sin\theta
y' = x\sin\theta + y\cos\theta
$$

这就是二维旋转公式。

#### 写成矩阵形式

$$
\begin{pmatrix}
x' \\
y'
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
x \\
y
\end{pmatrix}
$$

### RoPE核心原理

LLaMA 中的 RoPE (Rotary Position Embedding) 通过一种**旋转**操作将位置信息融入注意力机制。它的核心思想是：对 Query (Q) 和 Key (K) 向量的维度进行成对旋转，旋转的角度由该 token 的**绝对位置**决定。这样一来，当计算两个 token 的 Q 和 K 的点积时，其结果会自然地只依赖于它们的**相对位置**。

RoPE 会把一个高维向量拆成很多二维子空间，例如：

```text
q = [ q0, q1, q2, q3, q4, q5, ... ]

可以看成：

(q0, q1)  → 一个二维平面
(q2, q3)  → 另一个二维平面
(q4, q5)  → 另一个二维平面
...
```

对每一对维度做旋转：

$$
\begin{pmatrix}
q'_{2i} \\
q'_{2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
q_{2i} \\
q_{2i+1}
\end{pmatrix}
$$

其中：

- $m$ 是 token 的位置；
- $\theta_i$ 是这一对维度对应的固定频率；
- $m\theta_i$ 就是旋转角度。
- $\theta_i = 10000^{-2i/d}$（LLaMA 3将10000调为500000，支持更长的上下文）

### LLaMA 中的具体实现：`rotate_half` 风格

在实际的 LLaMA 代码实现中，为了高效计算，通常不会显式地构造旋转矩阵，而是采用一种称为 **`rotate_half`** 的向量化操作。

假设一个头维度 `head_dim` 为 \(d\) 的向量 `x`，其整体计算可以简洁地表示为：
\[
\text{RoPE}(x) = x \odot \cos + \text{rotate\_half}(x) \odot \sin
\]
其中，\(\odot\) 表示逐元素乘法。关键在于 `rotate_half` 函数的操作：

1. **分割**：将向量 `x` 沿最后一维**平均分成两半**，得到前半部分 \(x_{\text{first}}\) 和后半部分 \(x_{\text{second}}\)。
   \[
   x = [x_{\text{first}} \mid x_{\text{second}}]
   \]
2. **构造旋转**：`rotate_half(x)` 的结果是将**后半部分取负，前半部分保持原样，然后交换位置**。
   \[
   \text{rotate\_half}(x) = [-x_{\text{second}} \mid x_{\text{first}}]
   \]
   这个操作等价于对向量的每一个二维子空间进行 90 度的旋转，再与预先计算好的 `cos` 和 `sin` 表相乘，就能组合出任意角度的旋转效果。

这种实现方式可以完全利用现有的矩阵乘法和逐元素运算来完成，无需编写特殊的内核，因此在 LLaMA、GPT-NeoX、Mistral 等模型中被广泛采用。

