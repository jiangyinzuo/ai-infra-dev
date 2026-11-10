// Compile HTML: typst compile --root .. --features html llama_attention.typ llama_attention.html
#import "../typst_template/web-template.typ": web-document, boxed
#show: web-document
#set document(title: "LLaMA Attention")

= LLaMA Attention

假设Batch size （$B$）、头数（$H$）均为1。
输入序列：$X = \[ x_1 , x_2 , dots.h , x_n \] in bb(R)^(n times d)$

- #strong[$n$];：序列长度，也就是 token 的个数；
- #strong[$d$];：模型隐藏维度，也就是每个 token 向量的维度（hidden size）。LLaMA2-7b的$d$=4096

== 投影
<投影>
LLaMA2-7b的$W_Q$, $W_K$, $W_V$都是$d times d$的矩阵。
投影得到：

$ Q = X W_Q , quad K = X W_K , quad V = X W_V\
Q in bb(R)^(n times d) , quad K in bb(R)^(n times d) , quad V in bb(R)^(n times d) $

=== 投影算子融合机会
<投影算子融合机会>
- 未融合时，假设$B$=1，$H$=1
  - prefill阶段，执行3次独立的矩阵x矩阵
  - decode阶段，执行3次独立的向量x矩阵
  - transformers库这类简单实现中，QKV投影矩阵是三个独立的nn.Linear

  ```python
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
  ```
- 融合时，由于$X$相同，可以把$W_Q , W_K , W_V$拼成一个矩阵，做一次大矩阵（prefill）/向量（decode）x矩阵。
  - vLLM中，QKV投影矩阵需要合并实现

  ```python
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
  ```

  ```
  W_QKV = [ W_Q | W_K | W_V ]   形状 4096 × 12288

                 c=0 ... 4095              c=4096 ... 8191           c=8192 ... 12287
              ┌──── W_Q，4096列 ────┐  ┌──── W_K，4096列 ────┐  ┌──── W_V，4096列 ────┐
  i=0           [ WQ[0,0] ... WQ[0,4095] | WK[0,0] ... WK[0,4095] | WV[0,0] ... WV[0,4095] ]
  i=1           [ WQ[1,0] ... WQ[1,4095] | WK[1,0] ... WK[1,4095] | WV[1,0] ... WV[1,4095] ]
  i=2           [ WQ[2,0] ... WQ[2,4095] | WK[2,0] ... WK[2,4095] | WV[2,0] ... WV[2,4095] ]
  ...
  i=4095        [ WQ[4095,0] ... WQ[4095,4095] | WK[4095,0] ... WK[4095,4095] | WV[4095,0] ... WV[4095,4095] ]
  ```

== 单头注意力
<单头注意力>
$ A = "softmax"_(upright(r o w)) #h(-0.1667em) (frac(Q K^T, sqrt(d)) + M) in bb(R)^(n times n) , #h(2em) upright(A t t n) \( Q , K , V \) = A V = O in bb(R)^(n times d) . $

其中 $A_(t j) = alpha_(t j)$；矩阵乘法 $A V$ 的第 $t$ 行就是当前 token 的加权和输出。

对当前 Query 位置 $t$，Softmax 得到各个可见 Value 的权重，再加权求和得到输出：

$ alpha_(t j) = frac(exp #h(-0.1667em) (q_t^T k_j \/ sqrt(d)), sum_(ell = 1)^t exp #h(-0.1667em) (q_t^T k_ell \/ sqrt(d))) in \[ 0 , 1 \] subset bb(R) , quad j = 1 , dots.h , t , #h(2em) o_t = sum_(j = 1)^t alpha_(t j) v_j in bb(R)^d . $

因此整段序列的输出为 $O = \[ o_1 ; o_2 ; dots.h ; o_n \] in bb(R)^(n times d)$。

下面先将 $A$ 视为一般系数矩阵，不施加因果遮罩或归一化约束，只看矩阵乘法 $A V = O$ 如何按行展开。

// Redraw: typst compile attention-row.typ attention-row.svg
#html.elem("style", ".attention-row-diagram img { height: auto; }")
#html.elem("div", attrs: (class: "model-diagram attention-row-diagram"), image("./attention-row.svg", width: 100%, alt: "矩阵乘法按行展开：完整展示 4×4 的 A、4×3 的 V 和 O 网格，高亮 A 与 O 的第 3 行；四个系数分别乘 V 的四个横向行向量，再按分量相加，不使用因果遮罩或归一化条件。图内说明：Attention 中 α₃ⱼ 是第 3 个 Query 对第 j 个 token 的注意力权重标量，vⱼ = xⱼWᵥ 是第 j 个 token 经 Value 投影后的内容行向量。"))

图中取 $n = 4$、$d = 3$，完整画出所有矩阵元素和向量分量；$t = 3$ 表示选取第 3 行。每个标量 $alpha_(3 j)$ 乘到行向量 $v_j$ 的所有分量，再将四个结果按相同分量相加，得到行向量 $o_3$。

$M$ 是 causal mask：
$ M_(i j) = cases(delim: "{", 0 "," & j lt.eq i, - oo "," & j > i) $

逻辑上，$M in bb(R)^(n times n)$；真实实现中，attention 是带 batch 和多头的：

$ Q , K in bb(R)^(B times H times n times d)\
Q K^T in bb(R)^(B times H times n times n) $

此时 mask 不需要真的存 $B times H times n times n$ 份，通常存成：
$ M in bb(R)^(1 times 1 times n times n) $

== $Q K^T$的含义
<qkt的含义>
$Q K^T in bb(R)^(n times n)$

- #strong[第 $i$ 行];：第 $i$ 个 query token 对所有 key token 的打分；
- #strong[第 $j$ 列];：所有 query token 对第 $j$ 个 key token 的打分；
- #strong[元素 $\( i , j \)$];：第 $i$ 个 query向量 与第 $j$ 个 key向量 的相似度分数：$\( Q K^T \)_(i j) = q_i^T k_j$

```text
              key j →
           k1   k2   k3  ...  kn
        ┌────┬────┬────┬────┬────┐
   q1   │s11 │s12 │s13 │ ...│s1n │  ← 第 1 个 query 对全部 key 的分数
        ├────┼────┼────┼────┼────┤
   q2   │s21 │s22 │s23 │ ...│s2n │
        ├────┼────┼────┼────┼────┤
   q3   │s31 │s32 │s33 │ ...│s3n │
        ├────┼────┼────┼────┼────┤
  ...   │ ...│ ...│ ...│ ...│ ...│
        ├────┼────┼────┼────┼────┤
   qn   │sn1 │sn2 │sn3 │ ...│snn │
        └────┴────┴────┴────┴────┘
         ↑
      第 j 列：所有 query 对第 j 个 key 的分数
```

其中$s_(i j) = q_i^T k_j$

== Causal Mask
<causal-mask>
LLaMA 的 causal mask 让 $j > i$ 的位置变成 $- oo$：

```
        k1   k2   k3   k4
   q1 │ s11  -∞   -∞   -∞ │
   q2 │ s21  s22  -∞   -∞ │
   q3 │ s31  s32  s33  -∞ │
   q4 │ s41  s42  s43  s44│
```

- 第 $i$ 行只有前 $i$ 列有效；
- 第 $i$ 个 query 只能看到 $j lt.eq i$ 的 key。

变成$- oo$后， 再经过softmax ，这些位置的权重精确为 0。

== Softmax归一化
<softmax归一化>
softmax 的输入被称为logits。
除以$sqrt(d)$是为了把点积的方差从$d$拉回1，防止 softmax 饱和、梯度消失和数值溢出。

归一化通用形式：
$alpha_i = frac(f \( z_i \), sum_j f \( z_j \))$

- $f \( z \) = z$：普通归一化（保持比例）
- $f \( z \) = e^z$：softmax归一化（放大差异）
- $f \( z \) = max \( z , 0 \)$：ReLU 归一化

示例：$z = \( 1 , 2 , 3 \)$

- 普通归一化：$\( 1 \/ 6 , 2 \/ 6 , 3 \/ 6 \) = \( 0.167 , 0.333 , 0.500 \)$，比例仍是 $1 : 2 : 3$；
- softmax：$\( 0.090 , 0.245 , 0.665 \)$，比例变成 $1 : 2.7 : 7.4$，差距被拉大。

== QKV和Attention的含义
<qkv和attention的含义>
#table(
    columns: (50%, 50%),
    align: (auto,auto,),
    table.header([符号], [角色],),
    table.hline(),
    [$X in bb(R)^(n times d)$], [输入序列$X$的hidden state],
    [$Q$], [当前 token 发出的#strong[查询];],
    [$K$], [每个 token 提供的#strong[索引/标签];],
    [$V$], [每个 token 携带的#strong[实际内容];],
    [$upright(A t t n) \( Q , K , V \) in bb(R)^(n times d)$], [#strong[行 $i$];：第 $i$ 个 token 经过注意力后的#strong[新表示];。#strong[列];：隐藏特征维，每个 token 一个 $d$ 维向量。],
)

- Attention的输出和输入$X$形状完全相同，可以做残差。
- 经过Attention后，行与行之间不再独立，每一行都把其它行的$V$按权重加起来了。

== 多头注意力（MHA）
<多头注意力mha>
#table(
    columns: (20%, 20%, 20%, 20%, 20%),
    align: (left,left,left,left,left,),
    table.header([模型版本], [注意力头数 (Attention Heads)], [KV 头数 (KV Heads)], [每个头维度 ($d_(upright("head"))$)], [备注],),
    table.hline(),
    [#strong[Llama 2 7B];], [#strong[32];], [32], [#strong[128];], [标准多头注意力 (MHA)],
    [#strong[Llama 2 13B];], [#strong[40];], [40], [#strong[128];], [标准多头注意力 (MHA)],
    [#strong[Llama 2 70B];], [#strong[64];], [#strong[8];], [#strong[128];], [分组查询注意力 (GQA)],
)

Transformers的Llama 2 7B实现中，多头在投影阶段是经过同一个`q_proj、k_proj、v_proj`矩阵乘法，
经过view + transpose改变shape实现。

```
=== prefill attention ===
hidden_states: shape=(1, 4, 4096), dtype=torch.float32
position_ids: shape=(1, 4), dtype=torch.int64
attention_mask: shape=(1, 1, 4, 4), dtype=torch.float32
attention_mask[0, 0]: [[0.0, -inf, -inf, -inf], [0.0, 0.0, -inf, -inf], [0.0, 0.0, 0.0, -inf], [0.0, 0.0, 0.0, 0.0]]
cache tokens before attention: 0

=== Python-level operator trace ===
[01] LlamaAttention.forward
  [02] q_proj: Linear, input=(shape=(1, 4, 4096), dtype=torch.float32), output=(shape=(1, 4, 4096), dtype=torch.float32)
  [03] q_proj: view + transpose: output=(shape=(1, 32, 4, 128))
  [04] k_proj: Linear, input=(shape=(1, 4, 4096), dtype=torch.float32), output=(shape=(1, 4, 4096), dtype=torch.float32)
  [05] k_proj: view + transpose: output=(shape=(1, 32, 4, 128))
  [06] v_proj: Linear, input=(shape=(1, 4, 4096), dtype=torch.float32), output=(shape=(1, 4, 4096), dtype=torch.float32)
  [07] v_proj: view + transpose: output=(shape=(1, 32, 4, 128))
```

在投影阶段，多头注意力本质和单头没区别

```
W_Q，数学形状 4096 × 4096
行 i = 输入维度；列 c = 输出维度

                c=0 ... 127                  c=128 ... 255            ...         c=3968 ... 4095
        ┌────────── 头1 ─────────────┐ ┌────────── 头2 ─────────────┐     ┌──────────── 头32 ─────────────┐
i=0     [ WQ[0,0]   ... WQ[0,127]    | WQ[0,128]   ... WQ[0,255]    | ... | WQ[0,3968]   ... WQ[0,4095]   ]
i=1     [ WQ[1,0]   ... WQ[1,127]    | WQ[1,128]   ... WQ[1,255]    | ... | WQ[1,3968]   ... WQ[1,4095]   ]
i=2     [ WQ[2,0]   ... WQ[2,127]    | WQ[2,128]   ... WQ[2,255]    | ... | WQ[2,3968]   ... WQ[2,4095]   ]
...
i=4095  [ WQ[4095,0]... WQ[4095,127] | WQ[4095,128]... WQ[4095,255] | ... | WQ[4095,3968]...WQ[4095,4095] ]
        └──── W_Q^(1) 4096×128 ──────┘ └────── W_Q^(2) 4096×128 ────┘     └────── W_Q^(32) 4096×128 ──────┘
```

对第 $h$ 个头（$h = 1 , dots.h , H$）：

$ W_Q^(\( h \)) in bb(R)^(d times d_k) , #h(2em) W_K^(\( h \)) in bb(R)^(d times d_k) , #h(2em) W_V^(\( h \)) in bb(R)^(d times d_v) $

$ Q_h = X W_Q^(\( h \)) in bb(R)^(n times d_k) ,\
K_h = X W_K^(\( h \)) in bb(R)^(n times d_k) ,\
V_h = X W_V^(\( h \)) in bb(R)^(n times d_v) . $

每个头维度通常为：$d_k = d_v = d / H$

第 $h$ 个头的输出：

$ upright("head")_h = upright("softmax") (frac(Q_h K_h^top, sqrt(d_k)) + M) V_h in bb(R)^(n times d_v) . $

其中 $M$ 是 causal mask。

多头拼接后：

$ upright("MultiHead") \( X \) = upright("Concat") \( upright("head")_1 , dots.h , upright("head")_H \) W_O in bb(R)^(n times d) $

#html.elem("div", attrs: (class: "model-diagram"), image("./mha.svg", alt: "MHA"))

输出投影$W_O$：让不同 head 的输出可以线性混合，形成新的特征。没有
$W_O$ ，各 head 的信息只是简单拼接，缺少跨 head 的线性融合。

如果有 RoPE，则先对每个头的 $Q_h , K_h$ 做位置旋转，再算 attention：
$ upright("head")_h = upright("softmax") (frac(upright("RoPE") \( Q_h \) upright("RoPE") \( K_h \)^T, sqrt(d_k)) + M) V_h $

== 多头相比单头，理论计算量相同
<多头相比单头理论计算量相同>
从算法复杂度的角度看，对于长度为 $n$、维度为 $d$ 的序列：

- #strong[单头注意力的时间复杂度];：$Theta \( n^2 dot.op d \)$。
- #strong[多头注意力的时间复杂度];：假设有 $H$ 个头，每个头维度为 $d_k = d \/ H$。每个头的计算量为 $Theta \( n^2 dot.op d_k \)$，总和为 $H dot.op Theta \( n^2 dot.op d_k \) = Theta \( n^2 dot.op H dot.op d_k \) = Theta \( n^2 dot.op d \)$。

两者在时间复杂度上是#strong[同一个量级];。

实际实现性能对比：待分析

== RoPE
<rope>
=== 没有RoPE的后果
<没有rope的后果>
$ upright("score")_(i j) = frac(q_i^T k_j, sqrt(d_k)) = frac(\( x_i W_Q \)^T \( x_j W_K \), sqrt(d_k)) $

这个分数#strong[只依赖 $x_i$ 和 $x_j$ 的内容，完全不依赖 $i$ 和 $j$];。位置 $i , j$ 没有以任何形式进入公式。

==== Attention 是置换等变的
<attention-是置换等变的>
假设没有 causal mask，对输入做任意置换 $P$（打乱 token 顺序）：

$ X' = P X $

那么：

$ Q' = P X W_Q = P Q $

$ K' = P K , quad V' = P V $

代入注意力：

$ upright("Attn") \( Q' , K' , V' \) = upright("softmax") (frac(P Q K^T P^T, sqrt(d_k))) P V = P dot.op upright("softmax") (frac(Q K^T, sqrt(d_k))) V = P dot.op upright("Attn") \( Q , K , V \) $

#link("./softmax_permutation_equivariance.md")[逐行softmax的置换等变性证明]

即：

$ #boxed[$ upright("Attn") \( P X \) = P dot.op upright("Attn") \( X \) $] $

这叫#strong[置换等变];：打乱输入顺序，输出只是被同样打乱，内容完全不变。

==== 单层 attention、无RoPE、有Causal mask，只取最后位置，结果相同
<单层-attention无rope有causal-mask只取最后位置结果相同>
```
        k1   k2   k3   k4
   q1 │ s11  -∞   -∞   -∞ │
   q2 │ s21  s22  -∞   -∞ │
   q3 │ s31  s32  s33  -∞ │
   q4 │ s41  s42  s43  s44│
```

=== 二维向量的旋转矩阵
<二维向量的旋转矩阵>
假设有一个二维向量：$upright(bold(v)) = \( x , y \)$，向量长度是$r = sqrt(x^2 + y^2)$。

现在把这个向量绕原点逆时针旋转 $theta$ 角，得到新向量：$upright(bold(v))' = \( x' , y' \)$

旋转过程中有两个重要性质：

+ #strong[长度不变];：$\| upright(bold(v))' \| = \| upright(bold(v)) \| = r$。
+ #strong[方向变了];：原来的角度加上 $theta$，变成新的角度。

==== 旋转公式推导
<旋转公式推导>
原来的向量可以写成：$x = r cos alpha$, $y = r sin alpha$，
其中 $alpha$ 是原向量和 x 轴的夹角。

旋转 $theta$ 后，新角度是：$alpha + theta$。

所以，
$x' = r cos \( alpha + theta \)$，$y' = r sin \( alpha + theta \)$

用三角和角公式展开：

$ x' = r \( cos alpha cos theta - sin alpha sin theta \) y' = r \( sin alpha cos theta + cos alpha sin theta \) $

代入：$x = r cos alpha , quad y = r sin alpha$

得到：

$ x' = x cos theta - y sin theta y' = x sin theta + y cos theta $

这就是二维旋转公式。

==== 写成矩阵形式
<写成矩阵形式>
$ vec(x', y') = mat(delim: "(", cos theta, - sin theta; sin theta, cos theta) vec(x, y) $

=== RoPE核心原理
<rope核心原理>
LLaMA 中的 RoPE (Rotary Position Embedding) 通过一种#strong[旋转];操作将位置信息融入注意力机制。它的核心思想是：对 Query (Q) 和 Key (K) 向量的维度进行成对旋转，旋转的角度由该 token 的#strong[绝对位置];决定。这样一来，当计算两个 token 的 Q 和 K 的点积时，其结果会自然地只依赖于它们的#strong[相对位置];。

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

$ vec(q'_(2 i), q'_(2 i + 1)) = mat(delim: "(", cos \( m theta_i \), - sin \( m theta_i \); sin \( m theta_i \), cos \( m theta_i \)) vec(q_(2 i), q_(2 i + 1)) $

其中：

- $m$ 是 token 的位置；
- $theta_i$ 是这一对维度对应的固定频率；
- $m theta_i$ 就是旋转角度。
- $theta_i = 10000^(- 2 i \/ d)$（LLaMA 3将10000调为500000，支持更长的上下文）

=== RoPE保持相对位置不变，旋转后向量的点积只取决于角度差$\( m - n \) theta$
<rope保持相对位置不变旋转后向量的点积只取决于角度差m-ntheta>
+ #strong[旋转不改变向量长度];：所以 RoPE 只改变方向，不改变 Q/K 的尺度； \
+ #strong[两个旋转后的向量做点积时，位置信息只以“旋转角之差”的形式出现];； \
+ 因为旋转角分别是 $m theta$ 和 $n theta$，所以“角度差”自然变成 $\( m - n \) theta$，也就是相对位置。

假设某个 Query 向量和 Key 向量在二维平面里分别是：

$ q = \( r_q cos alpha , med r_q sin alpha \) , quad\
k = \( r_k cos beta , med r_k sin beta \) $

其中：

- $r_q , r_k$ 是长度；
- $alpha , beta$ 是它们原来的方向角。

现在 Query 在位置 $m$，Key 在位置 $n$。RoPE 对它们分别旋转：

$ q' = R \( m theta \) q $

$ k' = R \( n theta \) k $

其中 $R \( dot.op \)$ 是二维旋转矩阵。

旋转后：

- $q'$ 的方向角变成 $alpha + m theta$；
- $k'$ 的方向角变成 $beta + n theta$。

两个向量的点积等于：

$ q' dot.op k' = \| q' \| \| k' \| cos \( upright("两个向量的夹角") \) $

而旋转保持长度不变，所以：

$ \| q' \| = r_q , quad \| k' \| = r_k $

两个旋转后向量的夹角是：

$ \( alpha + m theta \) - \( beta + n theta \) = \( alpha - beta \) + \( m - n \) theta $

因此：

$ q' dot.op k' = r_q r_k cos (\( alpha - beta \) + \( m - n \) theta) $

注意看这个结果：

- $alpha - beta$ 来自 Query 和 Key 的#strong[内容方向差];；
- $\( m - n \) theta$ 来自#strong[位置差];；
- 单独的 $m$ 和 $n$ 没有出现，只有 $m - n$。

所以对于固定的 $q , k$，点积只依赖相对位置 $m - n$。

==== 用矩阵解释
<用矩阵解释>
二维旋转矩阵是：

$ R \( theta \) = mat(delim: "(", cos theta, - sin theta; sin theta, cos theta) $

它有一个关键性质：

$ R \( a \)^T R \( b \) = R \( b - a \) $

也就是说，两个旋转矩阵的“相对作用”只取决于角度差。

RoPE 后：

$ q' = R \( m theta \) q $

$ k' = R \( n theta \) k $

点积为：

$ q'^T k' = q^T R \( m theta \)^T R \( n theta \) k $

利用上面的性质：

$ R \( m theta \)^T R \( n theta \) = R \( n theta - m theta \) = R \( \( n - m \) theta \) $

所以：

$ q'^T k' = q^T R \( \( n - m \) theta \) k $

右边只出现：

$ n - m $

也就是相对位置。如果令：

$ Delta = m - n $

那么点积就是 $Delta$ 的函数：

$ q'^T k' = F \( q , k , Delta \) $

更准确地说：

#quote(block: true)[
Query 和 Key 的内容 $q , k$ 仍然会影响点积；但#strong[位置部分];只通过相对位移 $Delta = m - n$ 进入点积。
]

==== “保持长度不变”的作用
<保持长度不变的作用>
旋转矩阵是正交矩阵，满足：

$ R \( theta \)^T R \( theta \) = I $

所以：

$ parallel R \( theta \) x parallel = parallel x parallel $

也就是：

```text
旋转只改变方向，不改变长度
```

这有几个意义：

+ #strong[不会因为位置编码把 Q/K 的数值放大或缩小] \
  RoPE 不是给向量加一个位置向量，而是把向量旋转。范数不变，注意力的尺度更稳定。

+ #strong[点积公式可以写成“长度 × 夹角余弦”] \
  因为长度不变，两个旋转后向量的点积变化完全来自夹角变化。

+ #strong[位置信息只影响方向，不干扰内容幅值] \
  模型仍然可以通过 Q/K 的长度表达内容强度，而位置只改变方向。

不过要注意： \
“保持长度不变”本身不是相对位置编码的根本原因；根本原因是：

$ R \( m theta \)^T R \( n theta \) = R \( \( n - m \) theta \) $

也就是两个旋转的#strong[相对旋转];只依赖角度差。长度保持让这个几何解释更干净、数值更稳定。

==== 高维 RoPE
<高维-rope>
LLaMA 的 Q/K 不是二维，而是比如 128 维。RoPE 把它拆成很多二维对：

$ \( q_0 , q_1 \) , med \( q_2 , q_3 \) , med \( q_4 , q_5 \) , dots.h $

每一对用一个固定频率 $theta_i$ 旋转：

$ theta_i = 10000^(- 2 i \/ d) $

对于位置 $m$，第 $i$ 对维度的旋转角是：

$ m theta_i $

对于位置 $n$，第 $i$ 对维度的旋转角是：

$ n theta_i $

于是整个高维 Q/K 的 RoPE 可以写成块对角旋转：

$ R_m = "diag" (R \( m theta_0 \) , R \( m theta_1 \) , dots.h , R \( m theta_(d \/ 2 - 1) \)) $

那么：

$ q' = R_m q , quad k' = R_n k $

点积为：

$ q'^T k' = q^T R_m^T R_n k $

由于每个二维块都满足：

$ R \( m theta_i \)^T R \( n theta_i \) = R \( \( n - m \) theta_i \) $

所以：

$ q'^T k' = sum_i q_i^T R \( \( n - m \) theta_i \) k_i $

这个结果仍然是：

$ F \( q , k , m - n \) $

也就是说：

#quote(block: true)[
高维 RoPE 中，QK 点积的位置部分由多个频率的“相对旋转”叠加而成，但整体仍然只依赖相对位置 $m - n$。
]

它不是只依赖一个角度，而是依赖一组相对角度：

$ \( n - m \) theta_0 , med \( n - m \) theta_1 , dots.h $

但这些角度全都由同一个相对位移 $m - n$ 决定。

==== “相对位置编码”的含义
<相对位置编码的含义>
假设有两个 token：

- Query 在位置 $m$；
- Key 在位置 $n$。

注意力分数大致是：

$ upright("score") \( m , n \) = q_m^T k_n $

经过 RoPE 后：

$ upright("score") \( m , n \) = F \( q_m , k_n , m - n \) $

这意味着：

- 模型不需要直接知道“这是第 5 个 token、那是第 9 个 token”；
- 它通过 QK 点积看到的是“这两个 token 相距 4 个位置”这种相对关系；
- 如果把所有 token 的位置同时平移一个常数 $c$，即：

$ m arrow.r m + c , quad n arrow.r n + c $

那么：

$ \( m + c \) - \( n + c \) = m - n $

相对位置不变，因此 QK 点积不变。

这就是“相对位置编码”的核心：

```text
绝对位置：m, n
        ↓ RoPE 分别旋转
旋转角：mθ, nθ
        ↓ 点积只看角度差
相对位置：m-n
```

==== 一个容易误解的点
<一个容易误解的点>
“点积只取决于旋转角度之差”这句话要加一个前提：

#quote(block: true)[
对于固定的原始 $q , k$ 向量，位置部分只取决于旋转角之差。
]

完整的注意力分数当然还依赖 Q/K 的内容：

$ upright("score") = F \( upright("内容")_q , upright("内容")_k , upright("相对位置") \) $

不是“只依赖相对位置”。更准确的说法是：

#quote(block: true)[
RoPE 让位置信息以相对位移的形式进入 QK 点积，而不是以绝对位置 $m$ 或 $n$ 的形式进入。
]

==== 总结
<总结>
二维旋转满足：

$ R \( m theta \)^T R \( n theta \) = R \( \( n - m \) theta \) $

所以 RoPE 后：

$ \( R \( m theta \) q \)^T \( R \( n theta \) k \) = q^T R \( \( n - m \) theta \) k $

右边只出现 $n - m$，不单独出现 $m$ 或 $n$。同时旋转保持长度：

$ parallel R \( theta \) x parallel = parallel x parallel $

因此 RoPE 只改变 Q/K 的方向，不改变尺度，并让 QK 点积中的位置部分自然变成相对位置 $m - n$ 的函数。高维 RoPE 只是把这件事在很多二维子空间上并行做了一遍。

=== RoPE 为什么要用多频率旋转编码位置
<rope-为什么要用多频率旋转编码位置>
RoPE 不是只用一个角度编码位置，而是把 Q/K 向量拆成多个二维子空间，每个子空间用不同频率 $theta_i$ 旋转。这样做的本质是：用一组不同频率的旋转共同编码相对位置，形成多尺度、多分辨率的位置表示。

==== 1. 每个 $theta_i$ 是一个二维子空间的“角速度”
<每个-theta_i-是一个二维子空间的角速度>
LLaMA 把 Q/K 向量拆成若干二维对：

$ \( q_0 , q_1 \) , med \( q_2 , q_3 \) , med \( q_4 , q_5 \) , dots.h $

第 $i$ 对维度使用频率 $theta_i$。位置为 $m$ 的 token，其第 $i$ 对的旋转角为：

$ upright("angle")_i \( m \) = m theta_i $

其中：

$ theta_i = 10000^(- 2 i \/ d) $

因此不同维度对的“角速度”不同：

- $i = 0$ 时，$theta_0 = 1$，旋转最快；
- $i$ 越大，$theta_i$ 越小，旋转越慢；
- 最后一对维度频率最低，周期最长。

如果所有维度都用同一个 $theta$，所有二维对会同步旋转，位置编码就退化成“只有一个指针的钟”。

==== 2. 多频率使 QK 点积成为相对位置的多频率函数
<多频率使-qk-点积成为相对位置的多频率函数>
经过 RoPE 后，Query 在位置 $m$，Key 在位置 $n$。令相对位置差为 $Delta = m - n$，则 QK 点积中第 $i$ 个二维子空间的贡献可写成：

$ q_i^T R \( \( n - m \) theta_i \) k_i $

其位置相关部分由 $cos \( Delta theta_i \)$、$sin \( Delta theta_i \)$ 组成。整个 QK 点积的位置部分是多个频率项之和：

$ sum_i [cos \( Delta theta_i \) \( q_(2 i) k_(2 i) + q_(2 i + 1) k_(2 i + 1) \) + sin \( Delta theta_i \) \( dots.h.c \)] $

也就是说，相对位置 $Delta$ 不是只通过一个角度进入模型，而是通过一组不同频率的三角函数进入模型：

$ Delta theta_0 , med Delta theta_1 , med Delta theta_2 , dots.h $

这比单一频率的表达能力强得多。

==== 3. 如果所有维度同频会怎样
<如果所有维度同频会怎样>
假设所有 $theta_i = theta$，则整个 Q/K 向量相当于被同一个旋转矩阵旋转：

$ q' = R \( m theta \) q , quad k' = R \( n theta \) k $

点积只通过一个角度差进入：

$ q'^T k' = q^T R \( \( n - m \) theta \) k $

于是位置信息只表现为：

$ Delta theta med mod med 2 pi $

这带来两个问题。

===== 问题一：周期性强，容易混淆
<问题一周期性强容易混淆>
旋转是周期性的：

$ R \( Delta theta \) = R \( Delta theta + 2 pi \) $

所以 $Delta$ 和 $Delta + frac(2 pi, theta)$ 会产生完全相同的旋转。模型无法区分相差一个周期的位置差。

例如 $theta = 1$ 时，位置差 $0$ 和 $2 pi approx 6.28$ 在旋转上无法区分。

===== 问题二：无法同时兼顾近距和远距分辨率
<问题二无法同时兼顾近距和远距分辨率>
- 频率太高：相邻位置区分明显，但很快周期重复，远距离混淆；
- 频率太低：远距离不混淆，但相邻位置几乎转不动，近距离分辨率差。

单一频率无法同时做好“局部精细”和“长程区分”。

==== 4. 多频率组合相当于多尺度计数
<多频率组合相当于多尺度计数>
可以用时钟类比：

- 秒针转得快，负责精细时间；
- 分针转得慢，负责中等尺度；
- 时针转得更慢，负责大尺度。

只看秒针，不知道过了多少小时；只看时针，分不清几秒。三者组合，才能在大范围内唯一确定时间。

RoPE 的多频率也是同样道理：

- 高频 $theta_i$ 大：波长短，擅长区分邻近位置；
- 低频 $theta_i$ 小：波长长，擅长区分远距离；
- 多个频率组合：在很大范围内形成较唯一的相对位置模式。

单频时，混淆条件是：

$ Delta theta = 2 pi k $

多频时，要同时混淆需要：

$ Delta theta_i = 2 pi k_i , quad forall i $

这通常对应大得多的周期，因此可区分范围大大增加。

==== 5. 为什么用几何级数 $10000^(- 2 i \/ d)$
<为什么用几何级数-10000-2id>
$theta_i$ 对应的波长为：

$ lambda_i = frac(2 pi, theta_i) = 2 pi dot.op 10000^(2 i \/ d) $

因此：

- $i = 0$：$lambda_0 approx 2 pi approx 6.28$，高频、短波长；
- $i$ 增大：波长按几何级数增长；
- 最大波长约为：

$ 2 pi dot.op 10000 approx 6.28 times 10^4 $

这样频率在对数尺度上均匀分布，覆盖从“几个 token”到“几万个 token”的多个尺度。这种设计来自原始 Transformer 的正弦位置编码，LLaMA 沿用并用于旋转。底数 $10000$ 是经验超参；LLaMA 3 改成 $500000$，是为了让最低频周期更长，从而支持更长上下文。

==== 6. 同频问题的具体例子
<同频问题的具体例子>
取一个极简例子：

- `head_dim = 4`，共有 2 对二维子空间：$\( q_0 , q_1 \)$ 和 $\( q_2 , q_3 \)$；
- 取 $q = k = \( 1 , 1 , 1 , 1 \)$；
- 先假设所有维度对都用同一个频率 $theta = 1$。

对任意一对维度，RoPE 后点积的位置部分为：

$ \( q_(2 i) k_(2 i) + q_(2 i + 1) k_(2 i + 1) \) cos \( Delta theta_i \) + \( q_(2 i) k_(2 i + 1) - q_(2 i + 1) k_(2 i) \) sin \( Delta theta_i \) $

代入 $q = \( 1 , 1 \)$、$k = \( 1 , 1 \)$，有：

$ q_0 k_0 + q_1 k_1 = 2 , quad q_0 k_1 - q_1 k_0 = 0 $

所以每一对维度的贡献简化为：

$ 2 cos \( Delta theta_i \) $

===== 情况 A：所有维度同频 $theta = 1$
<情况-a所有维度同频-theta1>
两对维度完全相同，总 QK 位置分数为：

$ S_A \( Delta \) = 2 cos \( Delta \) + 2 cos \( Delta \) = 4 cos \( Delta \) $

这是一个周期为 $2 pi$ 的函数。因此：

$ S_A \( 0 \) = S_A \( 2 pi \) = S_A \( 4 pi \) = dots.h = 4 $

模型完全无法区分“位置差 0”“位置差 6.28”“位置差 12.57”。同样：

$ S_A \( 1 \) = S_A \( 1 + 2 pi \) = S_A \( 1 + 4 pi \) $

也无法区分。

也就是说，单一频率的位置编码以 $2 pi \/ theta$ 为周期重复。$theta = 1$ 时，周期只有约 $6.28$ 个 token，超过 6 个 token 的相对位置就开始“撞车”。

由于token的位置是整数，周期等于6.28不会严格撞车，但有近似重复、分辨率低的问题。
如果要严格撞车，周期需要是整数，但RoPE 实际使用：$theta_i = 10000^(- 2 i \/ d)$，
对应周期：$frac(2 pi, theta_i) = 2 pi dot.op 10000^(2 i \/ d)$。
这些周期通常不是整数，所以对整数位置差而言，#strong[严格周期撞车并不常见];。但单频的问题仍然存在，只是表现形式变成：

- 高频维度周期短，局部模式很快近似重复；
- 低频维度转得慢，近距离分辨率差；
- 单一频率无法同时提供多尺度信息；
- 多频组合把“严格或近似混淆”推到更远的距离。

===== 情况 B：用多频率 $theta_0 = 1 , med theta_1 = 0.01$
<情况-b用多频率-theta_01-theta_10.01>
总分数变为：

$ S_B \( Delta \) = 2 cos \( Delta times 1 \) + 2 cos \( Delta times 0.01 \) $

计算几个位置差：

#table(
    columns: (25%, 25%, 25%, 25%),
    align: (auto,auto,auto,auto,),
    table.header([$Delta$], [$2 cos \( Delta \)$], [$2 cos \( 0.01 Delta \)$], [$S_B \( Delta \)$],),
    table.hline(),
    [$0$], [$2$], [$2$], [$4.000$],
    [$2 pi approx 6.283$], [$2$], [$approx 1.996$], [$approx 3.996$],
    [$4 pi approx 12.566$], [$2$], [$approx 1.984$], [$approx 3.984$],
    [$1$], [$approx 1.081$], [$approx 1.9999$], [$approx 3.081$],
    [$1 + 2 pi approx 7.283$], [$approx 1.081$], [$approx 1.9947$], [$approx 3.076$],
)

现在：

- $Delta = 0$：$S_B = 4.000$；
- $Delta = 2 pi$：$S_B = 3.996$，不再相等；
- $Delta = 4 pi$：$S_B = 3.984$，继续不同。

高频那一对在 $Delta = 2 pi$ 处已经“转回原点”，但低频那一对只转了 $0.0628$ 弧度，还远没转回来。两对组合后，整体分数不再周期性重复。

多频率把这个“单周期”打散成一组不同周期的叠加：

$ Delta theta_0 , med Delta theta_1 , med dots.h , med Delta theta_(d \/ 2 - 1) $

要让所有频率同时回到原点，需要 $Delta$ 同时满足：

$ Delta theta_i equiv 0 med \( mod med 2 pi \) , quad forall i $

这要求 $Delta$ 是所有 $2 pi \/ theta_i$ 的公倍数。由于 $theta_i$ 按几何级数分布，这个公倍数极大，在实际上下文长度内基本不会发生。这就是多频率能大幅扩展可区分位置范围的原因。

==== 7. 总结
<总结-1>
不同维度使用不同 $theta_i$，是为了让 RoPE 用多个频率的旋转共同编码相对位置：

$ Delta theta_0 , Delta theta_1 , dots.h $

高频负责近距精细分辨，低频负责长程区分，组合起来形成多尺度位置表示。如果所有维度同频，位置编码会退化成单一正弦波，周期只有 $2 pi \/ theta$，周期性强、分辨率单一、表达能力弱。RoPE 用几何级数分布的频率 $theta_i = 10000^(- 2 i \/ d)$，把“周期撞车”推到极远的位置差上，从而实现长程可区分的位置编码。

=== LLaMA 中的具体实现：`rotate_half`
<llama-中的具体实现rotate_half>
在实际的 LLaMA 代码实现中，为了高效计算，通常不会显式地构造旋转矩阵，而是采用一种称为 #strong[`rotate_half`] 的向量化操作。
`rotate_half`和原始RoPE计算结果#strong[完全等价];。

假设一个头维度 `head_dim` 为 $d$ 的向量 `x`，其整体计算可以简洁地表示为：
$ upright("RoPE") \( x \) = x ⊙ cos + upright("rotate_half") \( x \) ⊙ sin $
其中，$⊙$ 表示逐元素乘法。关键在于 `rotate_half` 函数的操作：

+ #strong[分割];：将向量 `x` 沿最后一维#strong[平均分成两半];，得到前半部分 $x_(upright("first"))$ 和后半部分 $x_(upright("second"))$。
  $ x = \[ x_(upright("first")) divides x_(upright("second")) \] $
+ #strong[构造旋转];：`rotate_half(x)` 的结果是将#strong[后半部分取负，前半部分保持原样，然后交换位置];。
  $ upright("rotate_half") \( x \) = \[ - x_(upright("second")) divides x_(upright("first")) \] $
  这个操作等价于对向量的每一个二维子空间进行 90 度的旋转，再与预先计算好的 `cos` 和 `sin` 表相乘，就能组合出任意角度的旋转效果。

这种实现方式可以完全利用现有的矩阵乘法和逐元素运算来完成，无需编写特殊的内核，因此在 LLaMA、GPT-NeoX、Mistral 等模型中被广泛采用。
