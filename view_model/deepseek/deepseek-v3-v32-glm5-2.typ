// Compile HTML: typst compile --features html deepseek-v3-v32-glm5-2.typ deepseek-v3-v32-glm5-2.html
#import "web-template.typ": web-document, vbold, boxed, horizontalrule, Red, Green, Blue, TextRed
#import "deepseek-symbols.typ": symbols
#import "diagrams/deepseek-mla.typ": mla-diagram
#import "diagrams/deepseek-absorption.typ": absorption-diagram
#import "diagrams/deepseek-dsa.typ": dsa-diagram
#show: web-document.with(symbols: symbols)

== DeepSeek MLA
<deepseek-mla>
#strong[MLA（Multi-head Latent Attention，多头潜在注意力）] 是 DeepSeek-V2 首次提出、并在 DeepSeek-V3 中继续作为核心架构的注意力机制。它的根本动机是#strong[压缩推理阶段占用显存最大的 KV Cache];，同时保持与标准多头注意力（MHA）相当的模型表达能力。

与 MHA 为每个头分别缓存完整的 Key 和 Value 不同，MLA 将所有头的内容 Key/Value 联合低秩压缩为共享潜向量 $vbold(c)_t^(K V)$，并将单独投影的共享 RoPE Key $vbold(k)_t^R$ 一并存入 KV Cache；计算注意力时再从 $vbold(c)_t^(K V)$ 上投影恢复各头的内容 Key 和 Value。RoPE 位置部分与内容部分分开处理。

=== 结构图

#html.elem("div", attrs: (class: "model-diagram"), html.frame(mla-diagram()))

框图将下投影与实际实现的 RMSNorm 合并为 `MatMulRMSNorm`，三个输入分别是 hidden state、下投影权重、RMSNorm 缩放参数，输出沿用报告的 $vbold(c)_t^Q$、$vbold(c)_j^(K V)$ 记号。图中的输出实际已归一化；DSA 正文仍用 $macron(vbold(c))_t^Q$ 明确区分归一化前后。报告公式省略 RMSNorm；独立 RoPE Key 分支不经过 KV RMSNorm。融合框是逻辑计算表示，不声明官方或 HF 已使用单个融合 kernel；实际顺序与另一种 RMSNorm → MatMul 融合的数学变换见悬浮提示。输入 $vbold(h)$ 已过 MLA 模块外的 attention 前置 RMSNorm。

实现对照：#link("https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py")[官方 `MLA.forward`、`Block.forward`];；#link("https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py")[Transformers `DeepseekV3Attention`];。


=== 公式
<公式>
设第 $t$ 个 token 的输入为 $vbold(h)_t in RR^d$，注意力头数为 $n_h$，每个头的维度为 $d_h$。 以下以蓝色方框标出推理阶段需要写入 KV Cache 的向量。

==== 1. KV 的低秩联合“压缩”与“解压”
<kv-的低秩联合压缩与解压>
MLA 不分别缓存各头的 $K$ 和 $V$，而是先将输入 $vbold(h)_t$ #TextRed[下投影]为一个低维潜在向量 $vbold(c)_t^(K V)$ #TextRed[（压缩）]：

$
  Blue( #boxed[$ vbold(c)_t^(K V) $] ) = W^(D K V) vbold(h)_t in RR^(d_c)
$

其中 $W^(D K V) in RR^(d_c times d)$，$d_c$ 是远小于 $d_h n_h$ 的潜在维度。

计算注意力时，再用两个#TextRed[上投影]矩阵将 $vbold(c)_t^(K V)$ 解压为所有头的#strong[内容键];和#strong[值];：

$
  [vbold(k)_(t \, 1)^C ; vbold(k)_(t \, 2)^C ; dots.h ; vbold(k)_(t \, n_h)^C] = vbold(k)_t^C = W^(U K) vbold(c)_t^(K V) in RR^(n_h d_h)
$

RoPE Key 单独由输入投影并旋转得到，且在所有头间共享：

$
  Blue( #boxed[$ vbold(k)_t^R $] ) = "RoPE" \( W^(K R) vbold(h)_t \) in RR^(d_h^R)
$

将内容 Key 的第 $i$ 个头分量与共享 RoPE Key 拼接，得到完整 Key：

$
  vbold(k)_(t \, i) = \[ vbold(k)_(t \, i)^C ; vbold(k)_t^R \] in RR^(d_h + d_h^R) \, #h(2em) vbold(k)_(t \, i)^C in RR^(d_h) \, quad i = 1 \, dots.h \, n_h
$

$
  [vbold(v)_(t \, 1)^C ; vbold(v)_(t \, 2)^C ; dots.h ; vbold(v)_(t \, n_h)^C] = vbold(v)_t^C = W^(U V) vbold(c)_t^(K V) in RR^(n_h d_h)
$

其中 $W^(U K) \, W^(U V) in RR^(n_h d_h times d_c)$，$W^(K R) in RR^(d_h^R times d)$。推理时 KV Cache 只存储蓝框标出的 $vbold(c)_t^(K V)$ 和 $vbold(k)_t^R$，不存储解压后的 $vbold(k)_t^C$ 与 $vbold(v)_t^C$。

#quote(block: true)[
  上投影：升维、解压 \
  下投影：降维、压缩
]

==== 2. 查询的低秩压缩与解耦 RoPE
<查询的低秩压缩与解耦-rope>
MLA 对 Query 也做了低秩压缩，并严格将#strong[内容部分];与#strong[RoPE 位置部分];分开。

#strong[内容查询];由压缩后的潜在向量生成：

$
  vbold(c)_t^Q = W^(D Q) vbold(h)_t in RR^(d_(c'))
$ $
  [vbold(q)_(t \, 1)^C ; vbold(q)_(t \, 2)^C ; dots.h ; vbold(q)_(t \, n_h)^C] = vbold(q)_t^C = W^(U Q) vbold(c)_t^Q in RR^(n_h d_h)
$

#strong[RoPE 查询];则单独投影并施加位置旋转，不经过低秩压缩的干扰：

$
  [vbold(q)_(t \, 1)^R ; vbold(q)_(t \, 2)^R ; dots.h ; vbold(q)_(t \, n_h)^R] = vbold(q)_t^R = "RoPE" \( W^(Q R) vbold(c)_t^Q \) in RR^(n_h d_h^R)
$

最终第 $t$ 个 token 在第 $i$ 个头上的完整查询，由内容部分和位置部分#strong[拼接];而成：

$
  vbold(q)_(t \, i) = \[ vbold(q)_(t \, i)^C ; vbold(q)_(t \, i)^R \] in RR^(d_h + d_h^R) \, #h(2em) i = 1 \, dots.h \, n_h
$

$W^(D Q) in RR^(d_(c') times d)$，$W^(U Q) in RR^(n_h d_h times d_(c'))$，$W^(Q R) in RR^(n_h d_h^R times d_(c'))$。技术报告指出 $d_(c') lt.double n_h d_h$，Query 低秩压缩可降低激活内存；KV Cache 的缓存宽度则由 $d_c + d_h^R$ 决定，即缓存 $vbold(c)_t^(K V)$ 和共享的 $vbold(k)_t^R$。

==== 3. 注意力计算
<注意力计算>
注意力计算中，报告式（3）---（4）的 Key 在历史位置 $j$ 上写作 $vbold(k)_(j \, i)$；对位置 $t$ 的因果注意力，只使用 $j = 1 \, dots.h \, t$： $
  vbold(k)_(j \, i) = \[ vbold(k)_(j \, i)^C ; vbold(k)_j^R \] in RR^(d_h + d_h^R) \, #h(2em) vbold(k)_(j \, i)^C in RR^(d_h) \, quad vbold(k)_j^R in RR^(d_h^R) \, quad i = 1 \, dots.h \, n_h
$

第 $i$ 个头的注意力输出为：

$
  vbold(o)_(t \, i) = sum_(j = 1)^t "Softmax"_j (frac(vbold(q)_(t \, i)^T vbold(k)_(j \, i), sqrt(d_h + d_h^R))) vbold(v)_(j \, i)^C in RR^(d_h) \, #h(2em) vbold(v)_(j \, i)^C in RR^(d_h)
$

所有头的输出拼接后，经输出投影矩阵 $W^O$ 得到最终结果：

$
  vbold(u)_t = W^O \[ vbold(o)_(t \, 1) ; vbold(o)_(t \, 2) ; dots.h ; vbold(o)_(t \, n_h) \] in RR^d
$

这对应技术报告中的式（1）---（11）：KV 内容低秩压缩与解压、按头 Query 分解、解耦 RoPE Key、因果注意力及输出投影。需要特别注意，Value 只取内容部分 $vbold(v)_(j \, i)^C$，不带 RoPE 位置信息。

MLA 将每个 token 的 KV Cache 从标准多头注意力下的 $n_h \( 2 d_h + d_h^R \)$ 个标量，压缩为 $d_c + d_h^R$ 个标量：缓存共享潜向量 $vbold(c)_t^(K V)$ 和共享 RoPE Key $vbold(k)_t^R$，而不是完整的各头内容 Key/Value。DeepSeek-V3 技术报告将 MLA 确认为高效推理的重要架构设计。

== Llama MHA vs. DeekSeek MLA
<llama-mha-vs.-deekseek-mla>
#table(
  columns: (25%, 25%, 25%, 25%),
  align: (left,left,left,left,),
  table.header([符号], [含义], [Llama 2-7B \
  实际值], [DeepSeek-V3 \
  实际值],),
  table.hline(),

  [$d$],
  [模型隐藏维度。MHA并不要求$d = n_h \* d_h$],
  [$4096$],
  [$7168$],

  [$n_h$],
  [注意力头数],
  [$32$],
  [$128$],

  [$d_h$],
  [每个头的内容/Value 维度；DeepSeek-V3 的完整 Q/K 头维度为 $d_h + d_h^R$],
  [$128$],
  [$128$],

  [$d_c$],
  [DeepSeek-V3 中 KV 联合压缩后的潜在维度],
  [---],
  [$512$],

  [$d_(c')$],
  [DeepSeek-V3 中 Query 压缩后的潜在维度，满足 $d_(c') lt.double d_h n_h$],
  [---],
  [$1536$],

  [$d_h^R$],
  [每个头的解耦 RoPE 子维度；Key 的 RoPE 分量在所有头间共享],
  [---],
  [$64$],

  [$H in RR^(n times d)$],
  [输入隐藏矩阵，第 $t$ 行为 $vbold(h)_t$],
  [$H$$in RR^(n times 4096)$],
  [$H$$in RR^(n times 7168)$],

  [$vbold(h)_t in RR^d$],
  [第 $t$ 个 token 的隐藏向量],
  [$vbold(h)_t$$in RR^4096$],
  [$vbold(h)_t$$in RR^7168$],

  [$M in RR^(n times n)$],
  [causal mask，$M_(t \, j) = 0$ 当 $j lt.eq t$，否则 $M_(t \, j) = - oo$],
  [$n times n$],
  [$n times n$],

  [$W_Q^(\( i \)) \, W_K^(\( i \)) \, W_V^(\( i \)) in RR^(d_h times d)$],
  [Llama 2-7B 第 $i$ 个头的 Q/K/V 投影矩阵],
  [$128 times 4096$],
  [---],

  [$W^O in RR^(d times n_h d_h)$],
  [输出投影矩阵],
  [$4096 times 4096$],
  [$7168 times 16384$],

  [$W^(D Q) in RR^(d_(c') times d)$],
  [DeepSeek-V3 Query 下投影],
  [---],
  [$1536 times 7168$],

  [$W^(U Q) in RR^(n_h d_h times d_(c'))$],
  [DeepSeek-V3 Query 上投影],
  [---],
  [$16384 times 1536$],

  [$W^(Q R) in RR^(n_h d_h^R times d_(c'))$],
  [DeepSeek-V3 Query 的 RoPE 投影],
  [---],
  [$8192 times 1536$],

  [$W^(D K V) in RR^(d_c times d)$],
  [DeepSeek-V3 KV 联合下投影],
  [---],
  [$512 times 7168$],

  [$W^(U K) \, W^(U V) in RR^(n_h d_h times d_c)$],
  [DeepSeek-V3 K/V 上投影],
  [---],
  [$16384 times 512$],

  [$W^(K R) in RR^(d_h^R times d)$],
  [DeepSeek-V3 Key 的 RoPE 投影],
  [---],
  [$64 times 7168$],

  [$C^Q \, C^(K V)$],
  [DeepSeek-V3 压缩潜在矩阵],
  [---],
  [
    $C^Q in RR^(n times 1536)$ \
    $C^(K V) in RR^(n times 512)$
  ],

  [$Q_i \, K_i \, V_i \, O_i$],
  [第 $i$ 个头的 Q/K/V/O 矩阵],
  [$Q_i \, K_i \, V_i \, O_i in RR^(n times 128)$],
  [
    $Q_i \, K_i in RR^(n times 192)$ \
    $V_i \, O_i in RR^(n times 128)$
  ],

  [$Q_i^C \, K_i^C \, V_i^C$],
  [DeepSeek-V3 第 $i$ 个头的内容部分矩阵],
  [---],
  [$Q_i^C \, K_i^C \, V_i^C in RR^(n times 128)$],

  [$Q^R \, K^R$],
  [DeepSeek-V3 按头拼接的 Query RoPE 部分与所有头共享的 Key RoPE 部分],
  [---],
  [
    $Q^R in RR^(n times 8192)$ \
    $K^R in RR^(n times 64)$
  ],

  [$S_i \, A_i$],
  [第 $i$ 个头的注意力打分矩阵和注意力权重矩阵],
  [$S_i \, A_i in RR^(n times n)$],
  [$S_i \, A_i in RR^(n times n)$],

  [$"RoPE" \( dot.op \)$],
  [旋转位置编码],
  [
    作用于 Q/K 的 \
    $128$ 维头向量
  ],
  [作用于每个头 $64$ 维的 Query RoPE 分量，以及所有头共享的 $64$ 维 Key RoPE 向量],

  [$\[ X \, Y \]$],
  [最后一维拼接],
  [
    多头拼接后为 \
    $n times 4096$
  ],
  [
    单头 Q/K 拼接后为 \
    $n times 192$； \
    内容或 Value 多头拼接后为 \
    $n times 16384$
  ],

  [$"Softmax"$],
  [矩阵公式中表示对矩阵每一行沿列维 $j$ 做归一化；即固定 query 位置 $t$，对历史 key 位置 $j$ 归一化，且已含 causal mask],
  [
    作用于 \
    $S_i in RR^(n times n)$
  ],
  [
    作用于 \
    $S_i in RR^(n times n)$
  ],
)

#horizontalrule

#table(
  columns: (20%, 20%, 20%, 20%, 20%),
  align: (left,left,left,left,left,),
  table.header([模型], [组件], [向量公式], [矩阵公式], [备注],),
  table.hline(),
  [#strong[Llama 2-7B];],
  [#strong[Q];],
  [$vbold(q)_(t \, i) = "RoPE" (W_Q^(\( i \)) vbold(h)_t)$],
  [
    $Q_i = "RoPE" (H \( W_Q^(\( i \)) \)^T) in RR^(n times d_h)$ \
    $Q = \[ Q_1 \, dots.h \, Q_(n_h) \] in RR^(n times n_h d_h)$
  ],
  [
    $vbold(q)_(t \, i)$ 中，$t = 1 \, dots.h \, n$ 表示 Query token 位置，$i = 1 \, dots.h \, n_h$ 表示注意力头；$Q_i$ 是第 $i$ 个头的矩阵，$Q$ 沿头维拼接
  ],

  [],
  [#strong[K];],
  [$vbold(k)_(j \, i) = "RoPE" (W_K^(\( i \)) vbold(h)_j)$],
  [
    $K_i = "RoPE" (H \( W_K^(\( i \)) \)^T) in RR^(n times d_h)$ \
    $K = \[ K_1 \, dots.h \, K_(n_h) \] in RR^(n times n_h d_h)$
  ],
  [
    $vbold(k)_(j \, i)$ 中，$j = 1 \, dots.h \, n$ 是 Key 所属 token 的位置，$i = 1 \, dots.h \, n_h$ 是注意力头；query 位置为 $t$ 时，因果注意力只使用 $j lt.eq t$ 的 Key；每个头独立生成并全量缓存
  ],

  [],
  [#strong[V];],
  [$vbold(v)_(j \, i) = W_V^(\( i \)) vbold(h)_j$],
  [
    $V_i = H \( W_V^(\( i \)) \)^T in RR^(n times d_h)$ \
    $V = \[ V_1 \, dots.h \, V_(n_h) \] in RR^(n times n_h d_h)$
  ],
  [
    $vbold(v)_(j \, i)$ 中，$j = 1 \, dots.h \, n$ 是 Value 所属 token 的位置，$i = 1 \, dots.h \, n_h$ 是注意力头；V 不施加 RoPE
  ],

  [],
  [#strong[O];],
  [
    $A_(t \, j \, i) = "Softmax"_j (frac(vbold(q)_(t \, i)^T vbold(k)_(j \, i), sqrt(d_h)) + M_(t \, j))$ \
    $vbold(o)_(t \, i) = sum_(j = 1)^n A_(t \, j \, i) vbold(v)_(j \, i)$ \
    $vbold(O)_t = W^O \[ vbold(o)_(t \, 1) ; dots.h ; vbold(o)_(t \, n_h) \]$
  ],
  [
    $S_i = frac(Q_i K_i^T, sqrt(d_h)) + M$ \
    $A_i = "Softmax" \( S_i \) in RR^(n times n)$ \
    $O_i = A_i V_i in RR^(n times d_h)$ \
    $O_(upright("cat")) = \[ O_1 \, dots.h \, O_(n_h) \] in RR^(n times n_h d_h)$ \
    $O = O_(upright("cat")) \( W^O \)^T in RR^(n times d)$
  ],
  [$"Softmax" \( S_i \)$ 表示对 $S_i$ 每一行沿 $j$ 做归一化，已含 causal mask],

  [],
  [#strong[KV Cache];],
  [$"KVCache"_j = { vbold(k)_(j \, i) \, vbold(v)_(j \, i) }_(i = 1)^(n_h)$],
  [
    $K_(upright(c a c h e)) = K in RR^(n times n_h d_h) = RR^(n times 4096)$ \
    $V_(upright(c a c h e)) = V in RR^(n times n_h d_h) = RR^(n times 4096)$
  ],
  [实际保存完整 K/V；每层每个 token 存储 $2 n_h d_h = 8192$ 个标量],

  [#strong[DeepSeek-V3];],
  [#strong[Q];],
  [
    $vbold(c)_t^Q = W^(D Q) vbold(h)_t$ \
    $\[ vbold(q)_(t \, 1)^C ; dots.h ; vbold(q)_(t \, n_h)^C \] = vbold(q)_t^C = W^(U Q) vbold(c)_t^Q$ \
    $\[ vbold(q)_(t \, 1)^R ; dots.h ; vbold(q)_(t \, n_h)^R \] = vbold(q)_t^R = "RoPE" \( W^(Q R) vbold(c)_t^Q \)$ \
    $vbold(q)_(t \, i) = \[ vbold(q)_(t \, i)^C ; vbold(q)_(t \, i)^R \]$
  ],
  [
    $C^Q = H \( W^(D Q) \)^T in RR^(n times d_(c'))$ \
    $Q^C = C^Q \( W^(U Q) \)^T in RR^(n times n_h d_h)$ \
    按头拆分：$Q_i^C in RR^(n times d_h)$ \
    $Q^R = "RoPE" (C^Q \( W^(Q R) \)^T) in RR^(n times n_h d_h^R)$ \
    按头拆分：$Q_i^R in RR^(n times d_h^R)$ \
    $Q_i = \[ Q_i^C \, Q_i^R \] in RR^(n times \( d_h + d_h^R \))$ \
    $Q = \[ Q_1 \, dots.h \, Q_(n_h) \] in RR^(n times n_h \( d_h + d_h^R \))$
  ],
  [
    $vbold(q)_(t \, i)^C \, vbold(q)_(t \, i)^R \, vbold(q)_(t \, i)$ 中，$t = 1 \, dots.h \, n$ 表示 Query token 位置，$i = 1 \, dots.h \, n_h$ 表示注意力头；$vbold(q)_t^C \, vbold(q)_t^R$ 分别是按头拼接的内容查询和 RoPE 查询，$Q_i^R$ 是第 $i$ 个头的 RoPE 矩阵
  ],

  [],
  [#strong[K];],
  [
    $Blue( #boxed[$ vbold(c)_t^(K V) $] ) = W^(D K V) vbold(h)_t$ \
    $vbold(k)_(t \, i)^C = W_i^(U K) vbold(c)_t^(K V)$ \
    $Blue( #boxed[$ vbold(k)_t^R $] ) = "RoPE" \( W^(K R) vbold(h)_t \)$ \
    $vbold(k)_(t \, i) = \[ vbold(k)_(t \, i)^C ; vbold(k)_t^R \]$
  ],
  [
    $C^(K V) = H \( W^(D K V) \)^T in RR^(n times d_c)$ \
    $K^C = C^(K V) \( W^(U K) \)^T in RR^(n times n_h d_h)$ \
    按头拆分：$K_i^C in RR^(n times d_h)$ \
    $K^R = "RoPE" (H \( W^(K R) \)^T) in RR^(n times d_h^R)$ \
    $K_i = \[ K_i^C \, K^R \] in RR^(n times \( d_h + d_h^R \))$ \
    $K = \[ K_1 \, dots.h \, K_(n_h) \] in RR^(n times n_h \( d_h + d_h^R \))$
  ],
  [
    生成当前 token 的 Key 时，$t$ 是 token 位置，$i$ 是头索引；报告式（2）---（4）用 $t$ 表示正在生成的 Key，式（10）在注意力中以 $j lt.eq t$ 遍历历史 Key，故同一向量写作 $vbold(k)_(j \, i)$（RoPE 分量为 $vbold(k)_j^R$）。$t$ 与 $j$ 的用法有意区分，并非笔误
  ],

  [],
  [#strong[V];],
  [$[vbold(v)_(t \, 1)^C ; dots.h ; vbold(v)_(t \, n_h)^C] = vbold(v)_t^C = W^(U V) vbold(c)_t^(K V)$],
  [
    $V^C = C^(K V) \( W^(U V) \)^T in RR^(n times n_h d_h)$ \
    按头拆分：$V_i^C in RR^(n times d_h)$ \
    $V = \[ V_1^C \, dots.h \, V_(n_h) \] in RR^(n times n_h d_h)$
  ],
  [
    $vbold(v)_t^C$ 是 token $t$ 的多头拼接 Value；注意力中的 $vbold(v)_(j \, i)^C$ 以 $j$ 标记 Value 所属位置、以 $i$ 标记注意力头，由 $vbold(c)_j^(K V)$ 解压得到
  ],

  [],
  [#strong[O];],
  [
    $vbold(o)_(t \, i) = sum_(j = 1)^t "Softmax"_j (frac(vbold(q)_(t \, i)^T vbold(k)_(j \, i), sqrt(d_h + d_h^R))) vbold(v)_(j \, i)^C in RR^(d_h)$ \
    $vbold(O)_t = W^O \[ vbold(o)_(t \, 1) ; dots.h ; vbold(o)_(t \, n_h) \]$
  ],
  [
    $S_i = frac(Q_i K_i^T, sqrt(d_h + d_h^R)) + M$ \
    $A_i = "Softmax" \( S_i \) in RR^(n times n)$ \
    $O_i = A_i V_i^C in RR^(n times d_h)$ \
    $O_(upright("cat")) = \[ O_1 \, dots.h \, O_(n_h) \] in RR^(n times n_h d_h)$ \
    $O = O_(upright("cat")) \( W^O \)^T in RR^(n times d)$
  ],
  [$"Softmax" \( S_i \)$ 表示对 $S_i$ 每一行沿 $j$ 归一化，causal mask 已包含于 $M$；V 只取内容部分],

  [],
  [#strong[KV Cache];],
  [$"KVCache"_j = { Blue( #boxed[$ vbold(c)_j^(K V) $] ) \, Blue( #boxed[$ vbold(k)_j^R $] ) }$],
  [
    $C_(upright(c a c h e))^(K V) = Blue( #boxed[$ C^(K V) $] ) in RR^(n times d_c) = RR^(n times 512)$ \
    $K_(upright(c a c h e))^R = Blue( #boxed[$ K^R $] ) in RR^(n times d_h^R) = RR^(n times 64)$
  ],
  [
    实际只保存 $C^(K V)$ 和 $K^R$，不保存解压后的 $K^C \/ V^C$；每层每个 token 存储 $d_c + d_h^R = 512 + 64 = 576$ 个标量。按同配置 MHA（每头缓存 $d_h + d_h^R$ 维 K 和 $d_h$ 维 V），$r_(upright(c a c h e)) = frac(d_c + d_h^R, n_h \( 2 d_h + d_h^R \)) = 576 / 40960 approx 1.41 %$，即缓存减少约 $98.59 %$
  ],
)

=== MLA给大模型推理带来的变化
<mla给大模型推理带来的变化>
- #emph[KVCache管理];。潜向量 $C^(K V)$是所有头共享的表示，$K^R$ 为所有头共享的 RoPE 键，#TextRed[MLA 的 KV Cache 不再具有“头”这一维度，内存管理上必须为 MLA 设计完全不同的缓存块布局。] \[\[kvcache\]\]
- #emph[矩阵吸收（matrix absorption，出自DeepSeek-V2）];。
- #emph[张量并行];。$C^(K V)$和$K^R$被所有头共享，TP需要全量复制KVCache。
- #emph[算子融合];。FlashMLA，压缩---解压---注意力三段式流程。
- #emph[量化与精度约束];。MLA 在低秩潜空间工作，对量化误差更为敏感。推理引擎在为 MLA 设计 FP8 等低精度解码 kernel 时，通常只对潜向量缓存做量化，而保持查询和 RoPE 分量在较高精度。

==== MLA 矩阵吸收
<mla-矩阵吸收>

每组左图为吸收前，右图为吸收后，均复用完整 MLA 图。实色突出本组比较路径；其余模块（包括 RoPE）以原色半透明显示，坐标、尺寸、形状及布局与未吸收图完全一致。吸收后只替换相关权重、激活和连线，已消去的上投影不再绘制。

#html.elem("div", attrs: (class: "model-diagram diagram-comparison"), {
  html.elem("figure", {
    html.elem("figcaption", [内容分数路径 · 吸收前])
    html.frame(absorption-diagram(path: "qk", stage: "before"))
  })
  html.elem("figure", {
    html.elem("figcaption", [内容分数路径 · 吸收后])
    html.frame(absorption-diagram(path: "qk", stage: "after"))
  })
})

#html.elem("div", attrs: (class: "model-diagram diagram-comparison"), {
  html.elem("figure", {
    html.elem("figcaption", [Value / 输出路径 · 吸收前])
    html.frame(absorption-diagram(path: "value", stage: "before"))
  })
  html.elem("figure", {
    html.elem("figcaption", [Value / 输出路径 · 吸收后])
    html.frame(absorption-diagram(path: "value", stage: "after"))
  })
})

图示按本节代数关系预合并线性权重；官方实现也可保留权重分解，在运行时依次执行等价乘法。源潜向量已经过 RMSNorm；归一化依赖激活，不参与静态矩阵吸收。

先约定记号。忽略 batch 和层号，固定一个 query 位置 $t$，历史 key 位置 $j lt.eq t$。DeepSeek-V3 中：

按头分块：

$
  W^(U Q) & = mat(delim: "[", W_1^(U Q); dots.v; W_(n_h)^(U Q)) in RR^(n_h d_h times d_(c')) \,\
  W^(U K) & = mat(delim: "[", W_1^(U K); dots.v; W_(n_h)^(U K)) in RR^(n_h d_h times d_c) \,\
  W^(U V) & = mat(delim: "[", W_1^(U V); dots.v; W_(n_h)^(U V)) in RR^(n_h d_h times d_c) \,\
  W^O & in RR^(d times n_h d_h) .
$

其中 $W_i^(U Q) in RR^(d_h times d_(c'))$，$W_i^(U K) \, W_i^(U V) in RR^(d_h times d_c)$。

===== 第一步：把注意力分数拆成“内容 + RoPE”
<第一步把注意力分数拆成内容-rope>
原始 MLA 的注意力分数（第 $i$ 个头）：

$
  S_(t \, j \, i) = frac(vbold(q)_(t \, i)^T vbold(k)_(j \, i), sqrt(d_h + d_h^R)) = frac(\( vbold(q)_(t \, i)^C \)^T vbold(k)_(j \, i)^C, sqrt(d_h + d_h^R)) + frac(\( vbold(q)_t^R \)^T vbold(k)_j^R, sqrt(d_h + d_h^R))
$

其中：

$
  vbold(q)_(t \, i)^C = W_i^(U Q) vbold(c)_t^Q \, #h(2em) vbold(k)_(j \, i)^C = W_i^(U K) vbold(c)_j^(K V)
$

$
  vbold(q)_t^R = "RoPE" \( W^(Q R) vbold(c)_t^Q \) \, #h(2em) vbold(k)_j^R = "RoPE" \( W^(K R) vbold(h)_j \)
$

===== 第二步：内容分数的吸收------消掉显式 $vbold(q)^C$ 和 $vbold(k)^C$
<第二步内容分数的吸收消掉显式-mathbfqc-和-mathbfkc>
内容分数：

$
  \( vbold(q)_(t \, i)^C \)^T vbold(k)_(j \, i)^C = \( W_i^(U Q) vbold(c)_t^Q \)^T \( W_i^(U K) vbold(c)_j^(K V) \) = \( vbold(c)_t^Q \)^T underbrace(\( W_i^(U Q) \)^T W_i^(U K), W_i^(Q K)) vbold(c)_j^(K V)
$

定义#strong[吸收后的内容打分矩阵];：

$
  #boxed[$ W_i^(Q K) = \( W_i^(U Q) \)^T W_i^(U K) in RR^(d_(c') times d_c) $]
$

于是：

$
  \( vbold(q)_(t \, i)^C \)^T vbold(k)_(j \, i)^C = \( vbold(c)_t^Q \)^T W_i^(Q K) vbold(c)_j^(K V)
$

#strong[关键变化];：推理时不再需要显式构造 $vbold(k)_(j \, i)^C = W_i^(U K) vbold(c)_j^(K V)$。所有 $n_h$ 个 query 头#strong[直接与同一个共享潜向量 $vbold(c)_j^(K V)$ 交互];，只是各自使用不同的预计算矩阵 $W_i^(Q K)$。

===== 第三步：V 侧的吸收------消掉显式 $vbold(v)^C$
<第三步v-侧的吸收消掉显式-mathbfvc>
第 $i$ 个头对位置 $j$ 的注意力权重 $A_(t \, j \, i) in RR$ 是标量；$vbold(c)_j^(K V), tilde(vbold(o))_(t \, i) in RR^(d_c)$ 是压缩空间向量，$W_i^(U V) in RR^(d_h times d_c)$ 是上投影矩阵，$vbold(v)_(j \, i)^C, vbold(o)_(t \, i) in RR^(d_h)$。本模型维度为 $n_h=128,d_h=128,d_c=512,d=7168$。

$
vbold(o)_(t \, i) &= sum_(j=1)^t underbrace(A_(t \, j \, i), in RR) thin underbrace(vbold(v)_(j \, i)^C, in RR^(d_h)) \
&= sum_(j=1)^t A_(t \, j \, i) thin underbrace(W_i^(U V), in RR^(d_h times d_c)) thin underbrace(vbold(c)_j^(K V), in RR^(d_c)) \
&= sum_(j=1)^t W_i^(U V) (A_(t \, j \, i) thin vbold(c)_j^(K V)) \
&= W_i^(U V) underbrace(sum_(j=1)^t A_(t \, j \, i) thin vbold(c)_j^(K V), in RR^(d_c)) in RR^(d_h).
$

$A_(t \, j \, i)$ 是标量，所以可与矩阵 $W_i^(U V)$ 交换相乘顺序；固定头 $i$ 后，同一个 $W_i^(U V)$ 用于所有位置 $j$，因此可由线性分配律提出求和号。维度链为 $(d_h times d_c)(d_c times 1)=d_h times 1$；这不是交换两个矩阵的顺序。

#strong[压缩空间中的中间输出];

$
#boxed[$ tilde(vbold(o))_(t \, i) = sum_(j=1)^t A_(t \, j \, i) thin underbrace(vbold(c)_j^(K V), in RR^(d_c)) in RR^(d_c) $]
$

则：

$
vbold(o)_(t \, i) = underbrace(W_i^(U V), in RR^(d_h times d_c)) underbrace(tilde(vbold(o))_(t \, i), in RR^(d_c)) in RR^(d_h).
$

把所有头拼起来：

$
underbrace(mat(delim: "[", vbold(o)_(t \, 1);dots.v;vbold(o)_(t \, n_h)), in RR^(n_h d_h)) = underbrace(W_(upright("blk"))^(U V), in RR^(n_h d_h times n_h d_c)) underbrace(mat(delim: "[", tilde(vbold(o))_(t \, 1);dots.v;tilde(vbold(o))_(t \, n_h)), in RR^(n_h d_c)).
$

其中：

$
W_(upright("blk"))^(U V) = upright("diag") (W_1^(U V),dots.c,W_(n_h)^(U V)) in RR^(n_h d_h times n_h d_c), quad W_i^(U V) in RR^(d_h times d_c).
$

$upright("diag")$ 是分块对角矩阵。

再乘输出投影 $W^O$：

$
vbold(u)_t &= underbrace(W^O, in RR^(d times n_h d_h)) underbrace(mat(delim: "[", vbold(o)_(t \, 1);dots.v;vbold(o)_(t \, n_h)), in RR^(n_h d_h)) \
&= W^O W_(upright("blk"))^(U V) underbrace(mat(delim: "[", tilde(vbold(o))_(t \, 1);dots.v;tilde(vbold(o))_(t \, n_h)), in RR^(n_h d_c)) in RR^d.
$

定义#strong[吸收后的输出投影];：

$
#boxed[$ W^(O \, U V) = W^O W_(upright("blk"))^(U V) in RR^(d times n_h d_c) $]
$

于是：

$
vbold(u)_t = W^(O \, U V) mat(delim: "[", tilde(vbold(o))_(t \, 1);dots.v;tilde(vbold(o))_(t \, n_h)) in RR^d.
$

#strong[关键变化];：推理时不再需要显式构造 $vbold(v)_(j \, i)^C = W_i^(U V) vbold(c)_j^(K V)$。注意力加权直接在压缩潜空间完成，得到 $tilde(vbold(o))_(t \, i)$，最后用预计算的 $W^(O \, U V)$ 一次性投影回模型维度。

===== 第四步：NoPE 情形下的完全吸收（实际都是RoPE）
<第四步nope-情形下的完全吸收>
若无 RoPE（即去掉 $vbold(q)_t^R \, vbold(k)_j^R$），注意力权重为：

$
  A_(t \, j \, i) = "Softmax"_j (frac(\( vbold(c)_t^Q \)^T W_i^(Q K) vbold(c)_j^(K V), sqrt(d_h)))
$

完整输出：

$
  vbold(u)_t = W^(O \, U V) #scale(x: 180%, y: 180%)[\[] tilde(vbold(o))_(t \, 1) ; dots.h ; tilde(vbold(o))_(t \, n_h) #scale(x: 180%, y: 180%)[\]] \, #h(2em) tilde(vbold(o))_(t \, i) = sum_(j lt.eq t) A_(t \, j \, i) thin vbold(c)_j^(K V)
$

此时推理的计算图变为：

$
  vbold(c)_t^Q #h(0em) arrow.r #h(0em) #scale(x: 120%, y: 120%)[{] \( vbold(c)_t^Q \)^T W_i^(Q K) #scale(x: 120%, y: 120%)[}]_(i = 1)^(n_h) #h(0em) arrow.r #h(0em) A_(t \, j \, i) #h(0em) arrow.r #h(0em) tilde(vbold(o))_(t \, i) #h(0em) arrow.r #h(0em) vbold(u)_t
$

#strong[KV Cache 中只存 $vbold(c)_j^(K V)$];，所有 $n_h$ 个 query 头共享这一个表示。这在缓存结构和注意力形式上与 MQA 完全同构：

#table(
  columns: (33.33%, 33.33%, 33.33%),
  align: (left,left,left,),
  table.header([], [MQA], [MLA（NoPE 吸收后）],),
  table.hline(),
  [共享 KV 缓存],
  [一个 $d_h$ 维 K/V 头],
  [一个 $d_c$ 维潜向量 $vbold(c)^(K V)$],

  [Query 头数],
  [$n_h$],
  [$n_h$],

  [每头如何与共享 KV 交互],
  [直接点积],
  [通过预计算 $W_i^(Q K)$ 变换后点积],

  [输出如何回到模型维度],
  [$W^O$],
  [预计算 $W^(O \, U V)$],
)

所以“变身 MQA”的准确含义是：#strong[吸收后，KV 侧退化为一个所有头共享的压缩表示，Query 侧仍保留 $n_h$ 个头的独立打分能力];。它不是严格意义上的 MQA（MQA 的共享 KV 是显式高维向量，MLA 的是低秩潜向量），但在#strong[缓存布局和访存模式];上表现为 MQA 风格。

===== 第五步：RoPE 为什么不能被吸收
<第五步rope-为什么不能被吸收>
有 RoPE 时，位置分数为：

$
  \( vbold(q)_t^R \)^T vbold(k)_j^R = #scale(x: 120%, y: 120%)[\(] "RoPE" \( W^(Q R) vbold(c)_t^Q \) #scale(x: 120%, y: 120%)[\)]^T #scale(x: 120%, y: 120%)[\(] "RoPE" \( W^(K R) vbold(h)_j \) #scale(x: 120%, y: 120%)[\)]
$

RoPE 是对向量施加#strong[位置相关的旋转];：

$
  "RoPE" \( vbold(x) \, p \) = R_p thin vbold(x)
$

其中 $R_p$ 是随位置 $p$ 变化的旋转矩阵。因此：

$
  \( vbold(q)_t^R \)^T vbold(k)_j^R = \( R_t W^(Q R) vbold(c)_t^Q \)^T \( R_j W^(K R) vbold(h)_j \) = \( vbold(c)_t^Q \)^T \( W^(Q R) \)^T R_t^T R_j W^(K R) vbold(h)_j
$

中间夹着 $R_t^T R_j$，它#strong[依赖 $t$ 和 $j$ 的相对位置];，不能预先合并成一个与位置无关的矩阵。所以 $W^(Q R)$ 和 $W^(K R)$ 无法像 $W^(U Q) \, W^(U K)$ 那样被吸收掉。这就是 MLA 必须把 RoPE 分量#strong[解耦出来单独计算];的根本原因。

吸收后的完整推理公式为：

$
  #boxed[$ A_(t \, j \, i) = "Softmax"_j (frac(\( vbold(c)_t^Q \)^T W_i^(Q K) vbold(c)_j^(K V) + \( vbold(q)_t^R \)^T vbold(k)_j^R, sqrt(d_h + d_h^R))) $]
$

$
  #boxed[$ vbold(u)_t = W^(O \, U V) #scale(x: 180%, y: 180%)[\[] tilde(vbold(o))_(t \, 1) ; dots.h ; tilde(vbold(o))_(t \, n_h) #scale(x: 180%, y: 180%)[\]] \, #h(2em) tilde(vbold(o))_(t \, i) = sum_(j lt.eq t) A_(t \, j \, i) thin vbold(c)_j^(K V) $]
$

===== 推理前预计算什么，推理时缓存什么
<推理前预计算什么推理时缓存什么>
#strong[推理前];（离线融合，不增加运行时开销）：

$
  W_i^(Q K) = \( W_i^(U Q) \)^T W_i^(U K) \, #h(2em) W^(O \, U V) = W^O "diag" \( W_1^(U V) \, dots.h \, W_(n_h)^(U V) \)
$

#strong[推理时 KV Cache 存储];：

$
  vbold(c)_j^(K V) in RR^(d_c) quad upright("（所有头共享）") \, #h(2em) vbold(k)_j^R in RR^(d_h^R) quad upright("（所有头共享）")
$

#strong[不再存储];：$vbold(k)_(j \, i)^C$、$vbold(v)_(j \, i)^C$ 这些按头展开的显式高维向量。

===== 总结
<总结>
$
  #boxed[$ & upright("内容分数：") \( vbold(q)_(t \, i)^C \)^T vbold(k)_(j \, i)^C = \( vbold(c)_t^Q \)^T underbrace(\( W_i^(U Q) \)^T W_i^(U K), W_i^(Q K)) vbold(c)_j^(K V)\
   & upright("V 侧：") vbold(o)_(t \, i) = W_i^(U V) tilde(vbold(o))_(t \, i) \, quad tilde(vbold(o))_(t \, i) = sum_j A_(t \, j \, i) vbold(c)_j^(K V)\
   & upright("输出：") vbold(u)_t = W^(O \, U V) \[ tilde(vbold(o))_(t \, 1) ; dots.h ; tilde(vbold(o))_(t \, n_h) \] $]
$

矩阵吸收的本质是#strong[利用线性映射的结合律，把上投影矩阵提前融合进相邻矩阵];，使推理时 KV 侧只剩一个所有头共享的压缩潜向量 $vbold(c)_j^(K V)$------这正是“MLA 在推理时变身 MQA 风格”的数学来源。RoPE 因位置相关旋转不可吸收，必须作为独立的共享分量保留。

== DeepSeek MoE
<deepseek-moe>
V3 拥有 #strong[671B 总参数];，但每个 token 仅激活 #strong[37B];，激活率约 5.5%。这种解耦使得模型能存储海量知识，同时将推理计算成本控制在一个中等规模密集模型的水平。

传统 MoE（如 GShard）面临两个核心痛点：#strong[知识混杂];（少数大专家被迫学习差异极大的知识）与#strong[知识冗余];（不同专家重复存储通用常识）。DeepSeekMoE 通过两项基础设计来应对：

#strong[\1. 细粒度专家分割] 在保持总参数不变的前提下，将专家的 FFN 隐藏维度切得更细，从而用更多数量的小专家来替代少数大专家。V3 的每个 MoE 层包含 #strong[256 个路由专家];，每个专家的中间隐藏维度为 2048。这让专家能更灵活地组合，提升专业化程度。

#strong[\2. 共享专家隔离] 设置少量#strong[始终激活];的共享专家，专门负责捕获跨上下文的通用知识；路由专家则得以更专注于各自的特异性模式，从而减少路由专家之间的知识冗余。

在 V3 的 61 层中，#strong[除前三层保留稠密 FFN 外，其余 58 层均采用上述 MoE 结构];。

=== 辅助损失自由的全局负载均衡
<辅助损失自由的全局负载均衡>
#strong[问题根源];：在专家并行场景下，负载不均会导致“路由坍塌”------少数专家被过度使用，其余专家因梯度稀少而无法有效训练。传统方案是引入#strong[辅助损失];来惩罚不均衡，但这相当于在语言建模的主目标之外“掺入干扰”，以牺牲模型性能为代价换取负载均衡。

#strong[V3 的解法];：取消以全局负载均衡为目标的辅助损失，并为每个路由专家维护一个#strong[由负载动态调整的偏置项 $b_i$];。路由决策时，模型用 #strong[sigmoid 门控];为每个 token 打分，再将偏置项加到分数上，选取分数最高的 Top-K 个专家。关键在于，#strong[偏置项仅用于路由选择，不参与最终的门控权重计算];。专家负载偏高时调低其偏置，负载偏低时调高其偏置。

这样，负载均衡通过一个#strong[工程层面的动态调节机制];来实现，主训练目标不再受到辅助损失的污染，从而在负载均衡与模型性能之间取得了更优的平衡。作为补充，V3 仍保留了一个#strong[极小的序列级辅助损失];，仅用于防止单个序列内部出现极端失衡，而非全局负载调控。

=== 训练时路由约束
<训练时路由约束>
训练时，每个token被发到最多4个节点。

=== 推理阶段的 MoE 公式
<推理阶段的-moe-公式>
以下公式描述单个 MoE 层对 token 的前向计算。设 $ell$ 为层编号；DeepSeek-V3 前 3 层使用稠密 FFN，后 58 层使用 MoE。对任一 MoE 层和 token 位置 $t$，令 $vbold(u)_t in RR^d$ 为 FFN 输入。为简洁起见，省略层编号；专家参数和路由偏置均属于当前层。

==== 1. 计算路由分数并选择专家
<计算路由分数并选择专家>
对 $N_r$ 个路由专家分别计算 token 与专家的亲和分数：

$
  s_(i \, t) = "Sigmoid" \( vbold(u)_t^T vbold(e)_i \) \, #h(2em) i = 1 \, dots.h \, N_r
$

其中 $vbold(e)_i$ 是第 $i$ 个路由专家的中心向量（centroid vector）。路由时加入偏置 $b_i$ 以选择 Top-K 专家，但偏置不参与最终门控权重：

$
  g'_(i \, t) = cases(delim: "{", s_(i \, t) \, & s_(i \, t) + b_i in "TopK" ({ s_(j \, t) + b_j divides 1 lt.eq j lt.eq N_r } \, K_r) \,, 0 \, & upright("otherwise"))
$

推理时使用训练完成后固定的路由偏置；训练期间根据专家负载调整偏置的过程不属于这次前向计算。

===== hf transformers实现
<hf-transformers实现>
专家路由打分，其实是一个Linear。token数量大于0时，是矩阵乘法。

```python
    def __init__(self, config: DeepseekV3Config):
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
```

```
batchsize=1, num_token=4，专家数量256，hidden_state=7168时，
linear: input_shapes=[(4, 7168), (256, 7168)], output_shapes=[(4, 256)]
sigmoid: input_shapes=[(4, 256)], output_shapes=[(4, 256)]
```

==== 2. 由原始亲和分数计算门控权重
<由原始亲和分数计算门控权重>
被选中的专家使用未加偏置的亲和分数作为权重，并在选中的专家之间归一化：

$
  g_(i \, t) = frac(g'_(i \, t), sum_(j = 1)^(N_r) g'_(j \, t))
$

因此，偏置 $b_i$ 只影响 Top-K 专家的选择；门控权重 $g_(i \, t)$ 仍由未加偏置的亲和分数 $s_(i \, t)$ 归一化得到。

==== 3. 合并共享专家与路由专家输出
<合并共享专家与路由专家输出>
MoE 层输出为残差输入、始终激活的共享专家输出，以及 Top-K 路由专家的加权输出之和：

$
  vbold(h)'_t = vbold(u)_t + sum_(i = 1)^(N_s) "FFN"_i^(\( s \)) \( vbold(u)_t \) + sum_(i = 1)^(N_r) g_(i \, t) "FFN"_i^(\( r \)) \( vbold(u)_t \)
$

其中 $"FFN"_i^(\( s \))$ 和 $"FFN"_i^(\( r \))$ 分别表示共享专家和路由专家；未满足 Top-K 条件的路由专家具有 $g_(i \, t) = 0$，因此推理时只需计算被选中的 $K_r$ 个路由专家。共享专家不经过 Top-K 路由，始终参与计算。

==== 4. DeepSeek-V3 配置代入
<deepseek-v3-配置代入>
DeepSeek-V3 每个 MoE 层使用 $N_s = 1$ 个共享专家、$N_r = 256$ 个路由专家，并取 $K_r = 8$。因此每个 token 在该层激活 1 个共享专家和 8 个路由专家，共 9 个专家；每个路由专家的中间维度为 $2048$。模型共 61 层，其中 58 层使用该 MoE 结构。推理时不计算训练用的序列级负载均衡辅助损失；技术报告说明推理阶段不会丢弃 token。

以上路由与合并公式对应 DeepSeek-V3 技术报告 §2.1.2 的式（12）---（16）：#link("https://arxiv.org/html/2412.19437")[DeepSeek-V3 Technical Report];。

=== 单个专家内部：SwiGLU FFN
<单个专家内部swiglu-ffn>
#strong[DeepSeek-V3 的单个专家与 Llama 2-7B 的单层 FFN 都采用无偏置的 SwiGLU，区别主要是输入/输出维度和中间维度。] 这里比较的是一个专家与一个稠密 FFN，不是整个 MoE 层与整个 Llama 模型。

沿用前文的 FFN 输入 $vbold(u)_t in RR^d$，记中间维度为 $d_(upright(f f))$，输出为 $vbold(y)_t in RR^d$。按两者官方实现中的 `w1`、`w3`、`w2` 命名，单个 FFN 的完整推理公式为：

$
  #boxed[$ vbold(y)_t = "FFN" \( vbold(u)_t \) = W_2 ["SiLU" \( W_1 vbold(u)_t \) ⊙ \( W_3 vbold(u)_t \)] $]
$

其中 $"SiLU" \( x \) = x "Sigmoid" \( x \)$ 按元素作用，$⊙$ 表示逐元素乘法。$W_1$ 是门控分支投影，$W_3$ 是另一条线性分支投影，$W_2$ 将中间结果投影回模型隐藏维度；两条输入分支是并行计算，不是依次串联。

#table(
  columns: (25%, 25%, 25%, 25%),
  align: (left,left,left,left,),
  table.header([符号 / 组件], [逐 token 推理公式或含义], [Llama 2-7B \
  单层 FFN], [DeepSeek-V3 \
  单个路由专家],),
  table.hline(),
  [$d$],
  [输入和输出隐藏维度],
  [$4096$],
  [$7168$],

  [$d_(upright(f f))$],
  [每条分支的中间维度；不是注意力头维度 $d_h$],
  [$11008$],
  [$2048$],

  [$vbold(u)_t in RR^d$],
  [当前 token 的 FFN 输入],
  [$RR^4096$],
  [$RR^7168$],

  [$W_1 in RR^(d_(upright(f f)) times d)$],
  [
    门控投影后施加激活： \
    $vbold(a)_t = "SiLU" \( W_1 vbold(u)_t \)$
  ],
  [
    $W_1 in RR^(11008 times 4096)$ \
    $vbold(a)_t in RR^11008$
  ],
  [
    $W_1 in RR^(2048 times 7168)$ \
    $vbold(a)_t in RR^2048$
  ],

  [$W_3 in RR^(d_(upright(f f)) times d)$],
  [
    线性分支，不施加激活： \
    $vbold(v)_t^(upright(f f)) = W_3 vbold(u)_t$
  ],
  [
    $W_3 in RR^(11008 times 4096)$ \
    $vbold(v)_t^(upright(f f)) in RR^11008$
  ],
  [
    $W_3 in RR^(2048 times 7168)$ \
    $vbold(v)_t^(upright(f f)) in RR^2048$
  ],

  [$vbold(z)_t in RR^(d_(upright(f f)))$],
  [
    SwiGLU 逐元素门控： \
    $vbold(z)_t = vbold(a)_t ⊙ vbold(v)_t^(upright(f f))$
  ],
  [$RR^11008$],
  [$RR^2048$],

  [$W_2 in RR^(d times d_(upright(f f)))$],
  [
    输出投影： \
    $vbold(y)_t = W_2 vbold(z)_t$
  ],
  [
    $W_2 in RR^(4096 times 11008)$ \
    $vbold(y)_t in RR^4096$
  ],
  [
    $W_2 in RR^(7168 times 2048)$ \
    $vbold(y)_t in RR^7168$
  ],

  [线性层偏置],
  [三个投影均不带 bias；与前文路由偏置 $b_i$ 无关],
  [无],
  [无],

  [$P_(upright(F F N)) = 3 d thin d_(upright(f f))$],
  [仅计三个投影矩阵的权重参数；不计路由器、归一化及量化元数据],
  [
    $3 times 4096 times 11008$ \
    $= 135 \, 266 \, 304$ \
    约 $135.27$ M
  ],
  [
    $3 times 7168 times 2048$ \
    $= 44 \, 040 \, 192$ \
    约 $44.04$ M
  ],

  [何时执行],
  [指当前层中该 FFN / 专家是否参与某个 token 的计算],
  [每个 token 都执行该 FFN],
  [仅在该专家被 Top-8 路由选中时执行],
)

#strong[与上文 MoE 公式的衔接];：对第 $i$ 个路由专家，将上式 $W_1 \, W_2 \, W_3$ 换成它独立的 $W_(1 \, i)^(\( r \)) \, W_(2 \, i)^(\( r \)) \, W_(3 \, i)^(\( r \))$，即可得到 $"FFN"_i^(\( r \)) \( vbold(u)_t \)$。

依据：#link("https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py")[DeepSeek-V3 官方实现 `Expert` / `MLP`];、#link("https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json")[V3 671B 配置];、#link("https://github.com/meta-llama/llama/blob/main/llama/model.py")[Llama 官方实现 `FeedForward`];。Llama 2-7B 的中间维度按官方实现由 $8 / 3 d$ 向上对齐至 $256$ 的整数倍，得到 $11008$。

== DeepSeek V3 Dense FFN和MoE FFN对比
<deepseek-v3-dense-ffn和moe-ffn对比>
#table(
  columns: (14.29%, 14.29%, 14.29%, 14.29%, 14.29%, 14.29%, 14.29%),
  align: (left,left,left,left,left,left,left,),
  table.header([结构], [层数 / 配置], [中间维度 / \
  路由维度], [单个 FFN 参数量 \
  （估算）], [Router 每层参数量 \
  （含偏置，估算）], [所有 FFN 总参数量 \
  （不含 Router）], [所有 FFN + Router \
  总参数量（估算）],),
  table.hline(),
  [
    #strong[Dense FFN] \
    （前 3 层）
  ],
  [3 层，每层 1 个密集 FFN],
  [
    $d_(upright("dense")) = 18432$ \
    $d = 7168$
  ],
  [
    $3 times 7168 times 18432$ \
    $= 396.361728 upright("M")$
  ],
  [0],
  [
    $3 times 396.361728 upright("M")$ \
    $= 1.189085184 upright("B")$
  ],
  [$1.189085184 upright("B")$],

  [
    #strong[MoE 专家 FFN] \
    （后 58 层）
  ],
  [
    58 层，每层 \
    256 路由专家 + 1 共享专家 \
    \= 257 个专家
  ],
  [
    $d_(upright("moe")) = 2048$ \
    路由：$N_r = 256$ \
    $d = 7168$
  ],
  [
    $3 times 7168 times 2048$ \
    $= 44.040192 upright("M")$
  ],
  [
    $256 times 7168 + 256$ \
    $= 1 \, 835 \, 264$ \
    $approx 1.835264 upright("M")$ \
    其中偏置 $b_i$ \
    共 256 个标量
  ],
  [
    每层： \
    $257 times 44.040192 upright("M")$ \
    $= 11.318329344 upright("B")$ \
    58 层： \
    $58 times 11.318329344 upright("B")$ \
    $= 656.463101952 upright("B")$
  ],
  [
    每层： \
    $11.318329344 upright("B")$ \
    $+ 1.835264 upright("M")$ \
    $= 11.320164608 upright("B")$ \
    58 层： \
    $58 times 11.320164608 upright("B")$ \
    $= 656.569547264 upright("B")$
  ],

  [
    #strong[全模型所有] \
    #strong[FFN + Router 合计];
  ],
  [3 层密集 + 58 层 MoE],
  [---],
  [---],
  [
    总 Router： \
    $58 times 1 \, 835 \, 264$ \
    $= 106 \, 445 \, 312$ \
    $approx 0.106445312 upright("B")$
  ],
  [
    $1.189085184 upright("B")$ \
    $+ 656.463101952 upright("B")$ \
    $= 657.652187136 upright("B")$
  ],
  [
    $1.189085184 upright("B")$ \
    $+ 656.569547264 upright("B")$ \
    $= upright(bold(657.758632448 upright("B")))$
  ],
)

=== MoE算子实现
<moe算子实现>
- router算子实现：打分 (Sigmoid/Softmax) → 加选择偏置 → (可选) 分组海选 → Top-K 选择 → 提取原始权重 → 归一化 → 缩放
- MoE Group GEMM
  - persistent kernel
  - 只有token数量（M轴）在变，专家形状相同且固定（N、K轴固定），“M-grouped”调度
- SwiGLU 融合实现
- 通信算子：Dispatch和Combine

==== Persistent Kernel
<persistent-kernel>
+ 减少Kernel反复启动开销。
+ 减轻wave quantization现象。波量化是指当总工作量（如输出 tile 数）不能被 GPU 的 SM 数量整除时，会导致最后一“波”计算中只有部分 SM 在工作，其余闲置，造成利用率的骤降。在任务数量动态变化的 MoE 中，这个问题会被进一步放大。Persistent Kernel 让固定的 CTA 集合动态地从全局队列中“领取”下一个可用 tile，而不是静态地预先分配任务。这意味着先完成的 CTA 可以立即接手新任务，从而自然地在各个 SM 之间实现了动态负载均衡，有效“填平”了波量化造成的利用率低谷。
+ 跨 Tile 的流水线化。传统模式下，CTA 在完成一个 tile 后便会退出，下一个 tile 需要重新经历完整的“启动-加载数据-计算-写回”流程，这被称为“序言”（prologue）和“尾声”（epilogue）开销。 Persistent CTA 因为持续存活，可以在处理完当前 tile 后，立即开始加载下一个 tile 所需的数据，从而将不同 tile 的“尾声”与“序言”重叠起来。这种跨 tile 的流水线化隐藏了数据搬运和初始化延迟，使得计算单元能够更持续地保持忙碌。
+ 提升缓存复用与数据局部性。在 MoE 的 Grouped GEMM 中，不同专家可能共享相同的激活或权重分片。Persistent CTA 在处理连续或相关的 tile 时，有机会在片上缓存（如 L2 cache）中复用数据。

- 共享权重包括：
  - 共享专家。始终被激活的专家。
  - 专家绑定。跨层复用专家参数。

#block[
  #set enum(numbering: "1.", start: 5)
  + Persistent Kernel 的“常驻”特性使其成为构建 Mega-kernel。
]

代价：实现复杂度更高（需要手动管理任务队列和同步），并且可能因常驻而带来更高的寄存器压力或与并发内核的资源争用问题。

== 打印模型权重
<打印模型权重>
```
(view_model) server@jyz1:~/ai-infra-dev/view_model$ HF_ENDPOINT=https://hf-mirror.com python deepseekv3-2/deepseek_v31_class.py
model_id: deepseek-ai/DeepSeek-V3.1
class: <class 'transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3ForCausalLM'>
source: /home/server/ai-infra-dev/view_model/.venv/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py

=== print(model) ===
DeepseekV3ForCausalLM(
  (model): DeepseekV3Model(
    (embed_tokens): Embedding(129280, 7168)
    (layers): ModuleList(
      (0-2): 3 x DeepseekV3DecoderLayer(
        (self_attn): DeepseekV3Attention(
          (q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
          (q_a_layernorm): DeepseekV3RMSNorm((1536,), eps=1e-06)
          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
          (kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
          (kv_a_layernorm): DeepseekV3RMSNorm((512,), eps=1e-06)
          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
          (o_proj): Linear(in_features=16384, out_features=7168, bias=False)
        )
        (mlp): DeepseekV3MLP(
          (gate_proj): Linear(in_features=7168, out_features=18432, bias=False)
          (up_proj): Linear(in_features=7168, out_features=18432, bias=False)
          (down_proj): Linear(in_features=18432, out_features=7168, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): DeepseekV3RMSNorm((7168,), eps=1e-06)
        (post_attention_layernorm): DeepseekV3RMSNorm((7168,), eps=1e-06)
      )
      (3-60): 58 x DeepseekV3DecoderLayer(
        (self_attn): DeepseekV3Attention(
          (q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
          (q_a_layernorm): DeepseekV3RMSNorm((1536,), eps=1e-06)
          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
          (kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
          (kv_a_layernorm): DeepseekV3RMSNorm((512,), eps=1e-06)
          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
          (o_proj): Linear(in_features=16384, out_features=7168, bias=False)
        )
        (mlp): DeepseekV3MoE(
          (experts): DeepseekV3Experts(
            (act_fn): SiLUActivation()
          )
          (gate): DeepseekV3TopkRouter()
          (shared_experts): DeepseekV3MLP(
            (gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
            (up_proj): Linear(in_features=7168, out_features=2048, bias=False)
            (down_proj): Linear(in_features=2048, out_features=7168, bias=False)
            (act_fn): SiLUActivation()
          )
        )
        (input_layernorm): DeepseekV3RMSNorm((7168,), eps=1e-06)
        (post_attention_layernorm): DeepseekV3RMSNorm((7168,), eps=1e-06)
      )
    )
    (norm): DeepseekV3RMSNorm((7168,), eps=1e-06)
    (rotary_emb): DeepseekV3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=7168, out_features=129280, bias=False)
)

=== Meta model state ===
decoder layers: 61
parameter numel: 671,026,404,352
parameter dtypes: ['torch.bfloat16']
all parameters meta: True
all buffers meta: True
仅根据官方配置构建模块；没有真实权重数值，不能直接推理。
当前 Transformers 实现不构建 MTP 层，并合并专家投影；这里的 BF16 参数不代表 checkpoint 的 FP8 存储布局和量化 scale。
```

== DeepSeek Sparse Attention
<deepseek-sparse-attention>

#html.elem("div", attrs: (class: "model-diagram"), html.frame(dsa-diagram()))

固定缩放折叠后：MLA 执行 QK 内容投影与 Value / 输出投影的矩阵吸收；Indexer 仅把固定标量缩放重参数化到头权重投影。半透明模块保持原图位置与尺寸。LayerNorm、部分 RoPE、ReLU 不参与静态合并。

#html.elem("div", attrs: (class: "model-diagram"), html.frame(dsa-diagram(absorbed: true)))

稀疏注意力，改变的是每个Query token在计算注意力时，所能访问的Key Value候选集合的大小。

Dense Attention 可以看到全部历史 KV，注意力矩阵是稠密的，计算复杂度为 $cal(O) \( n^2 \)$。
但在 DeepSeek Sparse Attention 中，被大幅缩减为 Top-K 个（$k = 2048$）。模型先经过 lightning indexer 给所有历史 token 打分，然后只保留分数最高的 $k$ 个参与后续的核心注意力计算。核心注意力的计算复杂度降至 $cal(O) \( n dot.op k \)$；indexer 本身仍为 $cal(O) \( n^2 \)$。

#quote(block: true)[
  当上下文长度 $n$ = 128K 时，$k$ = 2048 仅占 $n$ 的 1.6%，核心注意力计算量理论上可减少约 98%。
]

=== Lightning Indexer
<lightning-indexer>
```
DeepseekV32Indexer(
  (wq_b): Linear(in_features=1536, out_features=8192, bias=False)
  (wk): Linear(in_features=7168, out_features=128, bias=False)
  (k_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True, bias=True)
  (weights_proj): Linear(in_features=7168, out_features=64, bias=False)
)
wq_b.weight: shape=(8192, 1536), dtype=torch.bfloat16, numel=12,582,912
wk.weight: shape=(128, 7168), dtype=torch.bfloat16, numel=917,504
k_norm.weight: shape=(128,), dtype=torch.bfloat16, numel=128
k_norm.bias: shape=(128,), dtype=torch.bfloat16, numel=128
weights_proj.weight: shape=(64, 7168), dtype=torch.bfloat16, numel=458,752
parameter numel: 13,959,424
```

#table(
  columns: (50%, 50%),
  align: (auto,auto,),
  table.header([符号], [含义],),
  table.hline(),
  [$t$、$j$],
  [当前 Query 位置、被访问的 KV 位置；因果可见范围为 $1 lt.eq j lt.eq t lt.eq n$，包含当前位置],

  [$Red( vbold(h)_t ) in RR^d = RR^7168$],
  [Attention / Indexer 的输入隐状态，已过当前层的 attention 前置 RMSNorm],

  [$i$、$n_h$],
  [头编号、MLA 注意力头数；MLA 中 $i = 1 \, dots.h \, n_h$],

  [$d_h$、$d_h^R$],
  [MLA 每头内容维度、解耦 RoPE 维度],

  [$Red( vbold(c)_t^Q ) in RR^(d_(c')) = RR^1536$、$vbold(c)_j^(K V) in RR^(d_c) = RR^512$],
  [
    沿用上文的 Query 压缩潜向量、KV 联合压缩潜向量；$Red( macron(vbold(c))_t^Q ) in RR^(d_(c')) = RR^1536$ 显式表示 Query RMSNorm 后的共享表示
  ],

  [上标 $C$、$R$、$I$],
  [分别表示内容、RoPE 、Indexer 相关变量],

  [$n_h^I = 64$、$d_h^I = 128$],
  [indexer 头数、每头维度；indexer 中 $i = 1 \, dots.h \, n_h^I$，区别于 MLA 的 $n_h \, d_h$],

  [$vbold(q)_(t \, i)^I \, vbold(k)_j^I in RR^(d_h^I) = RR^128$],
  [Indexer 每头 Query 与所有 indexer 头共享的 Key，均已完成部分 RoPE],

  [$I_(t \, j)$、$cal(S)_t$、$k$],
  [索引分数、被选 KV 位置集合、选择数量上限；V3.2 使用 $k = 2048$],
)

$
  #boxed[$ I_(t \, j) = sum_(i = 1)^(n_h^I) w_(t \, i)^I dot.op "ReLU" #h(-0.1667em) (\( vbold(q)_(t \, i)^I \)^T vbold(k)_j^I) $]
$

红色表示 MLA 与 indexer 共用的权重和激活；绿色表示 lightning indexer 自身的可学习参数。下列符号为#link("https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py")[官方实现];中参数的简记：

#table(
  columns: (33.33%, 33.33%, 33.33%),
  align: (auto,auto,auto,),
  table.header([权重], [尺寸], [官方参数],),
  table.hline(),
  [$Green( W^(I Q) ) in RR^(n_h^I d_h^I times d_(c'))$],
  [$RR^(8192 times 1536)$],
  [`indexer.wq_b.weight`],

  [$Green( W^(I K) ) in RR^(d_h^I times d)$],
  [$RR^(128 times 7168)$],
  [`indexer.wk.weight`],

  [$Green( W^(I W) ) in RR^(n_h^I times d)$],
  [$RR^(64 times 7168)$],
  [`indexer.weights_proj.weight`],

  [$Green( bold(gamma)^I ) in RR^(d_h^I)$],
  [$RR^128$],
  [`indexer.k_norm.weight`，Key 的 LayerNorm 缩放],

  [$Green( bold(beta)^I ) in RR^(d_h^I)$],
  [$RR^128$],
  [`indexer.k_norm.bias`，Key 的 LayerNorm 偏置],
)

==== $vbold(h)_t -> vbold(c)_t^Q -> vbold(q)^I$、$vbold(h)_j -> vbold(k)^I$ 的构造
<mathbf-qimathbf-ki-的构造>
来源：#link("https://arxiv.org/html/2512.02556v1#S2.SS1")[报告 §2.1，式 (1)、Figure 2];；#link("https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py")[官方 `MLA.forward` / `Indexer.forward`];；#link("https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py")[HF `DeepseekV32Attention.forward` / `DeepseekV32Indexer.forward`];。以下为实现的数学展开，忽略浮点舍入。

===== 投影与归一化

上文按报告省略压缩后的归一化；此处以 $Red( macron(vbold(c))_t^Q )$ 区分实现中 MLA 与 indexer 共用的 RMSNorm 结果。

共享权重的尺寸为 $Red( W^(D Q) ) in RR^(d_(c') times d) = RR^(1536 times 7168)$、$Red( bold(gamma)^Q ) in RR^(d_(c')) = RR^1536$。

$
  Red( vbold(c)_t^Q ) & = Red( W^(D Q) ) Red( vbold(h)_t ) in RR^(d_(c')) = RR^1536 \,\
  Red( macron(vbold(c))_t^Q ) & = "RMSNorm"_(Red( bold(gamma)^Q )) \( Red( vbold(c)_t^Q ) \) := Red( bold(gamma)^Q ) ⊙ frac(Red( vbold(c)_t^Q ), sqrt(parallel Red( vbold(c)_t^Q ) parallel_2^2 \/ d_(c') + epsilon_Q)) in RR^(d_(c')) = RR^1536 \,\
  tilde(vbold(q))_(t \, i)^I & = ["reshape"_(n_h^I times d_h^I) #h(-0.1667em) (Green( W^(I Q) ) Red( macron(vbold(c))_t^Q ))]_(i \, :) in RR^(d_h^I) = RR^128 \,\
  vbold(z)_j^I & = Green( W^(I K) ) Red( vbold(h)_j ) in RR^(d_h^I) = RR^128 \,\
  mu_j & = 1 / d_h^I sum_(a = 1)^(d_h^I) z_(j \, a)^I \, #h(2em) sigma_j^2 = 1 / d_h^I sum_(a = 1)^(d_h^I) \( z_(j \, a)^I - mu_j \)^2 \,\
  tilde(vbold(k))_j^I & = "LayerNorm"_(Green( bold(gamma)^I ) \, Green( bold(beta)^I )) \( vbold(z)_j^I \) = Green( bold(gamma)^I ) ⊙ frac(vbold(z)_j^I - mu_j vbold(1), sqrt(sigma_j^2 + 10^(- 6))) + Green( bold(beta)^I ) in RR^(d_h^I) = RR^128 .
$

#table(
  columns: (33.33%, 33.33%, 33.33%),
  align: (auto,auto,auto,),
  table.header([符号], [官方实现], [HF 实现],),
  table.hline(),
  [$Red( W^(D Q) )$、$Red( bold(gamma)^Q )$（MLA 参数，非 indexer 独有）],
  [`wq_a.weight`、`q_norm.weight`],
  [`q_a_proj.weight`、`q_a_layernorm.weight`],

  [$Red( macron(vbold(c))_t^Q )$（与 MLA 共享）],
  [`qr`],
  [`q_resid`],

  [$tilde(vbold(q))_(t \, i)^I$],
  [`indexer.wq_b(qr)` 后按 head 拆分],
  [`indexer.wq_b(q_resid)` 后按 head 拆分],

  [$tilde(vbold(k))_j^I$（所有 indexer heads 共享）],
  [`indexer.k_norm(indexer.wk(x))`],
  [`indexer.k_norm(indexer.wk(hidden_states))`],
)

#strong[部分 RoPE：前 $d_h^R = 64$ 维旋转，后 $d_h^I - d_h^R = 64$ 维直通]

令 $m = d_h^R \/ 2 = 32$，$D_(cos \, p)^I = "diag" \( upright(c o s)_p \[ : m \] \) in RR^(m times m) = RR^(32 times 32)$、$D_(sin \, p)^I = "diag" \( upright(s i n)_p \[ : m \] \) in RR^(m times m) = RR^(32 times 32)$；$upright(c o s)_p \, upright(s i n)_p$ 取实现的位置编码。Half-split 配对 $\( a \, a + m \)$，$a = 1 \, dots.h \, m$：

$
  R_p^I & = mat(delim: "[", D_(cos \, p)^I, - D_(sin \, p)^I; D_(sin \, p)^I, D_(cos \, p)^I) in RR^(d_h^R times d_h^R) = RR^(64 times 64) \,\
  P_p^I & = "diag" \( R_p^I \, "Id"_(d_h^I - d_h^R) \) in RR^(d_h^I times d_h^I) = RR^(128 times 128) \,\
  vbold(q)_(t \, i)^I & = P_t^I tilde(vbold(q))_(t \, i)^I in RR^(d_h^I) = RR^128 \,\
  vbold(k)_j^I & = P_j^I tilde(vbold(k))_j^I in RR^(d_h^I) = RR^128 .
$

官方：`apply_rotary_emb(..., interleaved=False)`；HF：`apply_rotary_pos_emb(..., unsqueeze_dim=2)`。缓存保存位置变换后的 indexer Key（官方另做下述量化）。

#strong[与评分式的缩放对齐]

$
  a_(t \, i)^I & = [Green( W^(I W) ) Red( vbold(h)_t )]_i \, #h(2em) w_(t \, i)^I := a_(t \, i)^I / sqrt(n_h^I d_h^I) \,\
  I_(t \, j)^(upright(H F)) & = sum_(i = 1)^(n_h^I) a_(t \, i)^I / sqrt(n_h^I) "ReLU" #h(-0.1667em) (frac(\( vbold(q)_(t \, i)^I \)^T vbold(k)_j^I, sqrt(d_h^I))) = sum_(i = 1)^(n_h^I) w_(t \, i)^I "ReLU" #h(-0.1667em) (\( vbold(q)_(t \, i)^I \)^T vbold(k)_j^I) .
$

#strong[官方额外的 Hadamard / FP8 路径]

$
  cal(H)^I & = "Hadamard"_(d_h^I) / sqrt(d_h^I) in RR^(d_h^I times d_h^I) = RR^(128 times 128) \,\
  \( cal(H)^I \)^T cal(H)^I & = "Id"_(d_h^I) \,\
  \( cal(H)^I vbold(q)_(t \, i)^I \)^T \( cal(H)^I vbold(k)_j^I \) & = \( vbold(q)_(t \, i)^I \)^T vbold(k)_j^I .
$

$
  hat(vbold(q))_(t \, i)^I & := cal(Q)_(upright(F P 8)) \( cal(H)^I vbold(q)_(t \, i)^I \) in RR^(d_h^I) = RR^128 \,\
  hat(vbold(k))_j^I & := cal(Q)_(upright(F P 8)) \( cal(H)^I vbold(k)_j^I \) in RR^(d_h^I) = RR^128 \,\
  I_(t \, j)^(upright(o f f i c i a l)) & approx I_(t \, j)^(upright(H F)) .
$

$cal(Q)_(upright(F P 8))$ 表示含 scale 的量化---反量化有效值；官方执行 `rotate_activation → act_quant → fp8_index`，HF 省略这三步，以模型 dtype 投影、FP32 点积评分；量化后不保证分数或 Top-K 完全相同。

==== Lightning Indexer 与 Full Attention 的计算复杂度对比
<lightning-indexer-与-full-attention-的计算复杂度对比>
统计单层、单序列的推理前向，1 MAC = 2 FLOPs。Full 与 Sparse 均采用 #strong[MLA 吸收路径];，注意力头数为 $n_h = 128$。

Indexer 为 $n_h^I = 64$、$d_h^I = 128$。下方主项与数值只计位置对上的点积和 Value 聚合，其他开销见步骤表。

令 $n$ 为可见 token 数（包含当前 token），$m = min \( n \, k \)$。完整因果 Prefill 的位置对数为 $P$，稀疏主注意力的位置对数为 $S$：

$
  P = frac(n \( n + 1 \), 2) \, #h(2em) S = sum_(t = 1)^n min \( t \, k \) = n m - frac(m \( m - 1 \), 2) .
$

===== 主项与计算节省
<主项与计算节省>
下表固定 Query 位置 $t$，查询矩阵的行表示各个头。沿用上文矩阵吸收中的 $W_i^(Q K) = \( W_i^(U Q) \)^T W_i^(U K)$，将吸收后的内容查询记为 $Q_t^(Q K)$，RoPE 查询仍记为 $Q_t^R$。实现中使用共享 RMSNorm 后的 $Red( macron(vbold(c))_t^Q )$：

$
  Q_t^(Q K) = mat(delim: "[", \( Red( macron(vbold(c))_t^Q ) \)^T W_1^(Q K); dots.v; \( Red( macron(vbold(c))_t^Q ) \)^T W_(n_h)^(Q K)) in RR^(n_h times d_c) = RR^(128 times 512) \, #h(2em) Q_t^R = mat(delim: "[", \( vbold(q)_(t \, 1)^R \)^T; dots.v; \( vbold(q)_(t \, n_h)^R \)^T) in RR^(n_h times d_h^R) = RR^(128 times 64) .
$

$C_(lt.eq t)^(K V) in RR^(t times d_c) = RR^(t times 512)$、$K_(lt.eq t)^R in RR^(t times d_h^R) = RR^(t times 64)$ 分别将历史潜向量 $vbold(c)_j^(K V)$、RoPE Key $vbold(k)_j^R$ 按行堆叠。因此，表中的矩阵点积与上文的吸收公式一致：

$
  \[ S_t \]_(i \, j) = \( Red( macron(vbold(c))_t^Q ) \)^T W_i^(Q K) vbold(c)_j^(K V) + \( vbold(q)_(t \, i)^R \)^T vbold(k)_j^R .
$

令 $A_t = "Softmax"_j \( S_t \/ sqrt(d_h + d_h^R) \)$、$m_t = \| cal(S)_t \| = min \( t \, k \)$。Sparse MLA 只取 $cal(S)_t$ 对应的列计算 Softmax，并从 $C^(K V)$、$K^R$ 中取对应的行。

Indexer 的大写 $Q_t^I$、$K_(lt.eq t)^I$ 是上文位置变换后的小写向量 $vbold(q)_(t \, i)^I$、$vbold(k)_j^I$ 的矩阵写法。

$
  Q_t^I = mat(delim: "[", \( vbold(q)_(t \, 1)^I \)^T; dots.v; \( vbold(q)_(t \, n_h^I)^I \)^T) in RR^(n_h^I times d_h^I) = RR^(64 times 128) \, #h(2em) K_(lt.eq t)^I = mat(delim: "[", \( vbold(k)_1^I \)^T; dots.v; \( vbold(k)_t^I \)^T) in RR^(t times d_h^I) = RR^(t times 128) .
$

因此 $\[ Q_t^I \( K_(lt.eq t)^I \)^T \]_(i \, j) = \( vbold(q)_(t \, i)^I \)^T vbold(k)_j^I$，正是原索引评分式中第 $i$ 个头对位置 $j$ 的点积。

$d_K = d_c + d_h^R = 512 + 64 = 576$ 表示 #strong[MLA 矩阵吸收后，每个头进行 QK 点积的有效维度];，由 $512$ 维 KV 潜向量与 $64$ 维 RoPE 分量拼接而成；不是未吸收时的 $d_h + d_h^R = 128 + 64 = 192$。吸收后的 Value 聚合维度为 $d_V = d_c = 512$。

下表每次乘加计 $2$ FLOPs。每个位置 $t$ 只有 $1$ 条 Query，可见 KV 数为 $t$（稀疏时为 $m_t$）；Prefill 将 $t=1,dots.h,n$ 的开销相加，Decode 只计算当前 $t=n$ 一步。

#table(
  columns: (25%, 25%, 25%, 25%),
  align: (auto,auto,auto,auto,),
  table.header([计算部分], [主项对应的计算公式（含激活尺寸）], [完整因果 Prefill 主项（FLOPs）], [单 token Decode 主项（FLOPs）],),
  table.hline(),
  [#strong[Indexer：QK 点积];],
  [
    $B_t = Q_t^I \( K_(lt.eq t)^I \)^T$ \
    $Q_t^I in RR^(n_h^I times d_h^I) = RR^(64 times 128)$ \
    $K_(lt.eq t)^I in RR^(t times d_h^I) = RR^(t times 128)$ \
    $B_t in RR^(n_h^I times t) = RR^(64 times t)$
  ],
  [
    $sum_(t=1)^n 2 n_h^I (1 times t) d_h^I$ \
    $= 2 n_h^I d_h^I sum_(t=1)^n t$ \
    $= 2 P n_h^I d_h^I$ \
    $= 2 P times 64 times 128$ \
    $= 16384 P tilde.eq #boxed[$ 8192 thin n^2 $]$
  ],
  [
    $2 n_h^I (1 times n) d_h^I$ \
    $= 2 n n_h^I d_h^I$ \
    $= 2 n times 64 times 128$ \
    $= #boxed[$ 16384 thin n $]$
  ],

  [#strong[Full MLA：QK 点积];],
  [
    $S_t = Q_t^(Q K) \( C_(lt.eq t)^(K V) \)^T$ \
    $quad + Q_t^R \( K_(lt.eq t)^R \)^T$ \
    $Q_t^(Q K) in RR^(n_h times d_c) = RR^(128 times 512)$ \
    $Q_t^R in RR^(n_h times d_h^R) = RR^(128 times 64)$ \
    $C_(lt.eq t)^(K V) in RR^(t times d_c) = RR^(t times 512)$ \
    $K_(lt.eq t)^R in RR^(t times d_h^R) = RR^(t times 64)$ \
    $S_t in RR^(n_h times t) = RR^(128 times t)$
  ],
  [
    $sum_(t=1)^n 2 n_h (1 times t) d_K$ \
    $= 2 n_h d_K sum_(t=1)^n t$ \
    $= 2 P n_h d_K$ \
    $= 2 P times 128 times \( 512 + 64 \)$ \
    $= 147456 P$ \
    $tilde.eq #boxed[$ 73728 thin n^2 $]$
  ],
  [
    $2 n_h (1 times n) d_K$ \
    $= 2 n n_h d_K$ \
    $= 2 n times 128 times \( 512 + 64 \)$ \
    $= #boxed[$ 147456 thin n $]$
  ],

  [#strong[Full MLA：Value 聚合];],
  [
    $tilde(O)_t = A_t C_(lt.eq t)^(K V)$ \
    $A_t in RR^(n_h times t) = RR^(128 times t)$ \
    $C_(lt.eq t)^(K V) in RR^(t times d_V) = RR^(t times 512)$ \
    $tilde(O)_t in RR^(n_h times d_V) = RR^(128 times 512)$
  ],
  [
    $sum_(t=1)^n 2 n_h (1 times t) d_V$ \
    $= 2 n_h d_V sum_(t=1)^n t$ \
    $= 2 P n_h d_V$ \
    $= 2 P times 128 times 512$ \
    $= 131072 P$ \
    $tilde.eq #boxed[$ 65536 thin n^2 $]$
  ],
  [
    $2 n_h (1 times n) d_V$ \
    $= 2 n n_h d_V$ \
    $= 2 n times 128 times 512$ \
    $= #boxed[$ 131072 thin n $]$
  ],

  [#strong[Full MLA 合计];],
  [Full QK 点积 + Full Value 聚合],
  [
    $sum_(t=1)^n 2 n_h (1 times t) (d_K + d_V)$ \
    $= 2 n_h (d_K + d_V) P$ \
    $= (147456 + 131072) P$ \
    $= 278528 P$ \
    $tilde.eq #boxed[$ 139264 thin n^2 $]$
  ],
  [
    $2 n_h (1 times n) (d_K + d_V)$ \
    $= (147456 + 131072) n$ \
    $= #boxed[$ 278528 thin n $]$
  ],

  [#strong[Sparse MLA：QK 点积];],
  [
    $S_t = Q_t^(Q K) \( C_(cal(S)_t)^(K V) \)^T$ \
    $quad + Q_t^R \( K_(cal(S)_t)^R \)^T$ \
    $Q_t^(Q K) in RR^(n_h times d_c) = RR^(128 times 512)$ \
    $Q_t^R in RR^(n_h times d_h^R) = RR^(128 times 64)$ \
    $C_(cal(S)_t)^(K V) in RR^(m_t times d_c) = RR^(m_t times 512)$ \
    $K_(cal(S)_t)^R in RR^(m_t times d_h^R) = RR^(m_t times 64)$ \
    $S_t in RR^(n_h times m_t) = RR^(128 times m_t)$
  ],
  [
    $sum_(t=1)^n 2 n_h m_t d_K$ \
    $= 2 n_h d_K sum_(t=1)^n m_t$ \
    $= 2 S n_h d_K$ \
    $= 2 S times 128 times \( 512 + 64 \)$ \
    $= 147456 S$ \
    $tilde.eq #boxed[$ 147456 thin n k $]$
  ],
  [
    $2 n_h (1 times m) d_K$ \
    $= 2 m n_h d_K$ \
    $= 2 m times 128 times \( 512 + 64 \)$ \
    $= 147456 m$ \
    $=^(m = k) #boxed[$ 147456 thin k $]$
  ],

  [#strong[Sparse MLA：Value 聚合];],
  [
    $tilde(O)_t = A_t C_(cal(S)_t)^(K V)$ \
    $A_t in RR^(n_h times m_t) = RR^(128 times m_t)$ \
    $C_(cal(S)_t)^(K V) in RR^(m_t times d_V) = RR^(m_t times 512)$ \
    $tilde(O)_t in RR^(n_h times d_V) = RR^(128 times 512)$
  ],
  [
    $sum_(t=1)^n 2 n_h m_t d_V$ \
    $= 2 n_h d_V sum_(t=1)^n m_t$ \
    $= 2 S n_h d_V$ \
    $= 2 S times 128 times 512$ \
    $= 131072 S$ \
    $tilde.eq #boxed[$ 131072 thin n k $]$
  ],
  [
    $2 n_h (1 times m) d_V$ \
    $= 2 m n_h d_V$ \
    $= 2 m times 128 times 512$ \
    $= 131072 m$ \
    $=^(m = k) #boxed[$ 131072 thin k $]$
  ],

  [#strong[DSA 合计];],
  [
    Indexer QK 点积 \
    \+ Sparse QK 点积 \
    \+ Sparse Value 聚合
  ],
  [
    $sum_(t=1)^n 2 (n_h^I t d_h^I + n_h m_t (d_K + d_V))$ \
    $= 2 P n_h^I d_h^I + 2 S n_h (d_K + d_V)$ \
    $= 16384 P + 147456 S + 131072 S$ \
    $= 16384 P + 278528 S$ \
    $tilde.eq #boxed[$ 8192 thin n^2 + 278528 thin n k $]$
  ],
  [
    $2 (n_h^I n d_h^I + n_h m (d_K + d_V))$ \
    $= 16384 n + 147456 m + 131072 m$ \
    $= 16384 n + 278528 m$ \
    $=^(m = k) #boxed[$ 16384 thin n + 278528 thin k $]$
  ],
)

表中近似取 $n gt.double k$，即 $2 P tilde.eq n^2$、$S tilde.eq n k$、$m = k$。精确计算使用 $P \, S \, m$。Indexer 与 Full MLA 扫描相同的位置对，公共因子 $2 P$（Decode 为 $2 n$）约去后，成本比例为：

$
  F_(upright(I n d e x e r)) / F_(upright(F u l l)) = frac(n_h^I d_h^I, n_h \( d_K + d_V \)) = frac(64 times 128, 128 times \( 576 + 512 \)) = 8192 / 139264 = 1 / 17 .
$

（$n_h$是注意力头数。）

再加上 Sparse MLA 的位置对占比，得到 DSA 的相对主项成本：

$
  #boxed[$ r_(upright(p r e f i l l)) = 1 / 17 + S / P \, #h(2em) r_(upright(d e c o d e)) = 1 / 17 + m / n \, #h(2em) upright("主项净节省比例") = 1 - r $]
$

#strong[$n = 128 times 1024 = 131072$、$k = 2048$ 时];，将 $P = frac(131072 times 131073, 2)$、$S = 131072 times 2048 - frac(2048 times 2047, 2)$、$m = 2048$ 代入下表；百分比按未舍入值计算：

#table(
  columns: (20%, 20%, 20%, 20%, 20%),
  align: (auto,auto,auto,auto,auto,),
  table.header([计算部分], [Prefill（TFLOPs）], [占 Full 主项], [Decode（GFLOPs）], [占 Full 主项],),
  table.hline(),
  [Full MLA],
  [
    $frac(2 P times 139264, 10^12)$ \
    $approx 2392.556$
  ],
  [基准：$100 %$],
  [
    $frac(2 n times 139264, 10^9)$ \
    $approx 36.507$
  ],
  [基准：$100 %$],

  [Lightning Indexer],
  [
    $frac(2 P times 8192, 10^12)$ \
    $approx 140.739$
  ],
  [
    $1 / 17 times 100 %$ \
    $approx 5.88 %$
  ],
  [
    $frac(2 n times 8192, 10^9)$ \
    $approx 2.147$
  ],
  [
    $1 / 17 times 100 %$ \
    $approx 5.88 %$
  ],

  [Sparse MLA],
  [
    $frac(2 S times 139264, 10^12)$ \
    $approx 74.183$
  ],
  [
    $S / P times 100 %$ \
    $approx 3.10 %$
  ],
  [
    $frac(2 m times 139264, 10^9)$ \
    $approx 0.570$
  ],
  [
    $m / n times 100 %$ \
    $approx 1.56 %$
  ],

  [#strong[DSA 合计];],
  [
    $frac(2 P times 8192 + 2 S times 139264, 10^12)$ \
    $approx upright(bold(214.922))$
  ],
  [
    $(1 / 17 + S / P) times 100 %$ \
    $approx upright(bold(8.98 %))$
  ],
  [
    $frac(2 n times 8192 + 2 m times 139264, 10^9)$ \
    $approx upright(bold(2.718))$
  ],
  [
    $(1 / 17 + m / n) times 100 %$ \
    $approx upright(bold(7.44 %))$
  ],

  [#strong[净节省：Full − DSA];],
  [
    $frac(2 P times 139264 - 2 P times 8192 - 2 S times 139264, 10^12)$ \
    $approx upright(bold(2177.634))$
  ],
  [
    $\( 1 - r_(upright(p r e f i l l)) \) times 100 %$ \
    $approx upright(bold(91.02 %))$
  ],
  [
    $frac(2 n times 139264 - 2 n times 8192 - 2 m times 139264, 10^9)$ \
    $approx upright(bold(33.789))$
  ],
  [
    $\( 1 - r_(upright(d e c o d e)) \) times 100 %$ \
    $approx upright(bold(92.56 %))$
  ],
)

#strong[DSA 用低成本的全历史评分，换掉高成本的全历史 MLA；Indexer 仍有二次项，但没有 Value 聚合。]

完整 Prefill 要用 $S \/ P$，不能套用最后一个 Query 的 $k \/ n$；若 $n lt.eq k$，则 $S = P$，没有稀疏收益，反而多出 indexer 开销。

===== Indexer 关键步骤的时间复杂度
<indexer-关键步骤的时间复杂度>
Query 下投影和 RMSNorm 复用 MLA 的结果，不重复计费。以下 Q/K 预处理与缓存写入只作用于#strong[新 token];；Key 为所有 indexer 头共享。本配置每头一个 FP8 scale；Hadamard / FP8 是官方路径，HF 路径省略这两步。

#table(
  columns: (33.33%, 33.33%, 33.33%),
  align: (auto,auto,auto,),
  table.header([步骤], [完整 Prefill], [单 token Decode],),
  table.hline(),
  [Query 上投影 `wq_b`],
  [$cal(O) \( n d_(c') n_h^I d_h^I \)$],
  [$cal(O) \( d_(c') n_h^I d_h^I \)$],

  [Key 投影 `wk`],
  [$cal(O) \( n d thin d_h^I \)$],
  [$cal(O) \( d thin d_h^I \)$],

  [动态头权重投影 `weights_proj`],
  [$cal(O) \( n d thin n_h^I \)$],
  [$cal(O) \( d thin n_h^I \)$],

  [Key LayerNorm],
  [$cal(O) \( n d_h^I \)$],
  [$cal(O) \( d_h^I \)$],

  [Q/K 部分 RoPE],
  [$cal(O) \( n \( n_h^I + 1 \) d_h^R \)$],
  [$cal(O) \( \( n_h^I + 1 \) d_h^R \)$],

  [Q/K 快速 Hadamard],
  [$cal(O) \( n \( n_h^I + 1 \) d_h^I log d_h^I \)$],
  [$cal(O) \( \( n_h^I + 1 \) d_h^I log d_h^I \)$],

  [Q/K FP8 量化（含分块 scale）],
  [$cal(O) \( n \( n_h^I + 1 \) d_h^I \)$],
  [$cal(O) \( \( n_h^I + 1 \) d_h^I \)$],

  [合并头权重、Query scale 与固定缩放],
  [$cal(O) \( n n_h^I \)$],
  [$cal(O) \( n_h^I \)$],

  [写入 Key Cache 及 scale],
  [$cal(O) \( n d_h^I \)$],
  [$cal(O) \( d_h^I \)$],

  [#strong[全历史 $Q K^T$ 点积];],
  [$#boxed[$ cal(O) \( P n_h^I d_h^I \) $]$],
  [$#boxed[$ cal(O) \( n n_h^I d_h^I \) $]$],

  [ReLU、乘头权重、跨头求和],
  [$cal(O) \( P n_h^I \)$],
  [$cal(O) \( n n_h^I \)$],

  [显式 causal mask（若单独处理）],
  [$cal(O) \( n^2 \)$],
  [$cal(O) \( n \)$；可见范围已裁好时省略],

  [Top-K 无序选择（线性选择算法）],
  [期望 $cal(O) \( P \)$],
  [期望 $cal(O) \( n \)$],

  [所选项排序（若要求有序输出）],
  [$cal(O) \( S log \( 1 + m \) \)$],
  [$cal(O) \( m log \( 1 + m \) \)$],

  [输出 / 复制位置索引],
  [$cal(O) \( S \)$],
  [$cal(O) \( m \)$],

  [新 token 的 dtype 转换、拼接与布局复制],
  [$cal(O) \( n \[ d + \( n_h^I + 1 \) d_h^I \] \)$],
  [$cal(O) \( d + \( n_h^I + 1 \) d_h^I \)$],
)

Top-K 若采用堆式选择，Prefill / Decode 分别为 $cal(O) \( P log \( 1 + k \) \)$ / $cal(O) \( n log \( 1 + m \) \)$；`torch.topk` 的具体策略取决于后端。缓存、复制行表示数据搬运量；仅建立视图的 `view` / `split` 为 $cal(O) \( 1 \)$。

#strong[固定模型维度和 $k$，DSA 的完整 Prefill 仍为二次增长，单 token Decode 仍随 $n$ 线性增长；只有稀疏主注意力部分降为 $cal(O) \( n k \)$ / $cal(O) \( k \)$。]

#strong[适用边界];：上述净节省不含投影、归一化、Top-K、量化、访存和通信，不是整模型收益或实测加速比。省算需要真正跳过未选中 KV 的稀疏 kernel；官方参考实现先算完整分数再加 mask、使用稠密 Value 聚合，不能获得表中的稀疏收益。FlashAttention 减少中间矩阵存储，不改变 Full Attention 的二次计算量。

依据：#link("https://arxiv.org/html/2512.02556v1#S2.SS3")[报告 §2.1、§2.3];、#link("https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py")[官方 `Indexer.forward` / `MLA.forward`];。复算：#link("lightning_indexer_flops.py");，运行 `python3 lightning_indexer_flops.py`，可用 `--seq-len`、`--top-k` 调整参数。

==== Lighting Indexer算子在NPU上的性能问题
<lighting-indexer算子在npu上的性能问题>
lighting indexer算子，NPU不亲和，耗时比较大。 使用PCP，切多份输入，弥补NPU上lighting indexer性能差的缺点。

===== PCP
<pcp>
模型在 Prefill 阶段的大部分计算（如 RMSNorm, FFN, Embedding 等）都是 Token-Wise（逐 Token） 的，彼此独立，完全可以分开计算。只有 Attention 模块需要 Token 间的交互，而 PCP 通过特定的通信机制解决了这个问题。

== GLM 5.2/5.3 IndexShare
<glm-5.25.3-indexshare>
https:\/\/arxiv.org/abs/2603.12201

在 DSA（DeepSeek Sparse Attention）机制中，每一层都有一个轻量的“索引器 (Indexer)”，它负责从长达百万 token 的历史中，为本层的每个 token 快速筛选出最相关的 Top-k 个 token，后续的注意力计算只在这些被选中的 token 上进行，以此实现稀疏化。

注意到相邻层索引器选出的 Top-k token 集合高度重叠（重叠度高达 70% 到 100%）。这意味着下一层的索引器常常在“重新发现”上一层刚刚选过的位置，造成了大量的重复计算。

IndexShare 利用这种重叠性，让一组连续层共享同一次索引选择的结果。具体采用 “1 层 Full + 3 层 Shared” 的四层分组模式： - Full 层（组长）：在每 4 层中，有 1 层会实际运行索引器，完成一次完整的 Top-k token 选择。 - Shared 层（组员）：紧随其后的 3 层则完全不拥有自己的索引器（不建参数、不做投影、不跑索引选择），直接复用其组长（Full 层）选出的那组 Top-k token 位置来进行自己的稀疏注意力计算。

索引计算量被压缩到原来的 1/4。
