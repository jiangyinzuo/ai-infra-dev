// HTML export drops align elements, so preserve their formula bodies in a flex row.
#show align: it => context {
  if target() == "html" {
    html.elem("div", attrs: (style: "display: flex; overflow: auto hidden;"), it.body)
  } else {
    it
  }
}

== Online Softmax

Online Softmax 是 Flash Attention 的核心数学机制。其基本思想是将完整的 Softmax 计算拆分为多个 Block，通过维护少量中间统计量，实现分块、增量式计算，而无需物化完整的 Attention Score Matrix 和 Probability Matrix。

=== 初始计算

设当前 Query Block 包含 $B_q$ 个 Query Row，Query/Key 的 Head Dimension 为 $d$，Value 的 Head Dimension 为 $d_v$。令第 $j$ 个 Key/Value Block 包含 $B_(k,j)$ 个 Token（$j = 1, ..., T$），则

#align(left)[$ Q in RR^(B_q times d) $]

#align(left)[$ K_j in RR^(B_(k,j) times d) quad V_j in RR^(B_(k,j) times d_v) $]

其中 $K_j$ 与 $V_j$ 沿序列维度对齐，即二者具有相同的行数 $B_(k,j)$。将所有 KV Block 沿序列维度拼接，记

#align(left)[$ K = [K_1; K_2; ...; K_T] in RR^(N_k times d) $]

#align(left)[$ V = [V_1; V_2; ...; V_T] in RR^(N_k times d_v) $]

其中

#align(left)[$ N_k = sum_(j=1)^T B_(k,j). $]

因此，第 $j$ 个 KV Block 产生的局部 Attention Score 为

#align(left)[$ S_j = (Q K_j^T) / sqrt(d) in RR^(B_q times B_(k,j)). $]

其中 $K_j^T in RR^(d times B_(k,j))$，所以矩阵乘法的维度为

#align(left)[$ (B_q times d) (d times B_(k,j)). $]

完整的 Score Matrix 按列拼接为

#align(left)[$ S = [S_1, S_2, ..., S_T] in RR^(B_q times N_k), $]

但 Online Softmax 不需要将其整体物化。

Online Softmax 为每个 Query Row 维护三个统计量：

- $m$：当前已处理 Score 的最大值。
- $l$：以当前最大值为基准的指数和。
- $u$：以当前最大值为基准的加权 Value 累积结果。

初始化：

#align(left)[$ m_0 = -infinity $]

#align(left)[$ l_0 = 0 $]

#align(left)[$ u_0 = 0 $]

其中 $m_0$ 和 $l_0$ 的形状均为 $B_q times 1$，$u_0$ 的形状为 $B_q times d_v$。上述统计量均按 Query Row 独立维护。

=== 递推计算

对于第 $j$ 个 KV Block，首先计算局部 Score：

#align(left)[$ S_j = (Q K_j^T) / sqrt(d) $]

计算当前 Block 每个 Query Row 的局部最大值：

#align(left)[$ m_j^"local" = "rowmax"(S_j) $]

更新全局最大值：

#align(left)[$ m_j = max(m_(j-1), m_j^"local") $]

由于最大值发生变化，需要将历史统计量重新缩放到新的最大值基准下。定义两个缩放系数：

#align(left)[$ alpha_j = exp(m_(j-1) - m_j) $]

#align(left)[$ beta_j = exp(m_j^"local" - m_j) $]

更新指数和：

#align(left)[$ l_j = alpha_j l_(j-1) + beta_j "rowsum"(exp(S_j - m_j^"local")) $]

更新加权 Value 累积结果：

#align(left)[$ u_j = alpha_j u_(j-1) + beta_j exp(S_j - m_j^"local") V_j $]

其中 $alpha_j$ 和 $beta_j$ 均按 Query Row 广播，指数运算为逐元素运算。

=== 最终输出

当所有 $T$ 个 KV Block 处理完毕后，得到最终统计量：

#align(left)[$ m_T, l_T, u_T $]

最终 Attention Output 为：

#align(left)[$ O = u_T / l_T $]

其中除法按照 Query Row 在 Value Dimension 上进行广播。

同时，可以计算每个 Query Row 对应的 Log-Sum-Exp（LSE）：

#align(left)[$ "LSE" = m_T + log(l_T) $]

LSE 表示原始 Attention Score 的指数和的对数：

#align(left)[$ "LSE"_i = log(sum_j exp(S_(i,j))) $]

因此，Softmax Probability 可以通过以下公式恢复：

#align(left)[$ P_(i,j) = exp(S_(i,j) - "LSE"_i) $]

=== 数学意义

Online Softmax 的关键在于：每次处理新的 KV Block 时，通过更新全局最大值 $m$，并利用缩放系数 $alpha$ 和 $beta$ 对历史统计量进行重标定，从而保证增量计算结果与完整 Softmax 在数学上等价。

整个计算过程中，只需要维护 $m$、$l$ 和 $u$ 三个中间统计量，无需保存完整的 $N times N$ Attention Score Matrix 或 Probability Matrix。

此外，$(m,l,u)$ 还可以通过相同的重标定公式进行合并，因此支持不同 KV Partition 上的并行计算与分布式 Reduce。这也是 Online Softmax 能够应用于 Context Parallelism、Ring Attention 和 Flash-Decoding 等技术的重要数学基础。

*一个实现细节：* 上述初始化中的 $m_0 = -infinity$ 是数学上的初始状态。实际 GPU/NPU Kernel 中通常会对首个有效 Block 做特殊处理，避免直接计算 $exp(-infinity - (-infinity))$ 产生 NaN。
