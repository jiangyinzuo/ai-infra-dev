## DeepSeek MLA

**MLA（Multi-head Latent Attention，多头潜在注意力）** 是 DeepSeek-V2 首次提出、并在 DeepSeek-V3 中继续作为核心架构的注意力机制。它的根本动机是**压缩推理阶段占用显存最大的 KV Cache**，同时保持与标准多头注意力（MHA）相当的模型表达能力。

与 MHA 为每个头分别缓存完整的 Key 和 Value 不同，MLA 将所有头的内容 Key/Value 联合低秩压缩为共享潜向量 \(\mathbf{c}_t^{KV}\)，并将单独投影的共享 RoPE Key \(\mathbf{k}_t^R\) 一并存入 KV Cache；计算注意力时再从 \(\mathbf{c}_t^{KV}\) 上投影恢复各头的内容 Key 和 Value。RoPE 位置部分与内容部分分开处理。

---

### 数学公式（依据 DeepSeek-V3 技术报告）

设第 \(t\) 个 token 的输入为 \(\mathbf{h}_t \in \mathbb{R}^{d}\)，注意力头数为 \(n_h\)，每个头的维度为 \(d_h\)。
以下以蓝色方框标出推理阶段需要写入 KV Cache 的向量。

#### 1. KV 的低秩联合“压缩”与“解压”

MLA 不分别缓存各头的 \(K\) 和 \(V\)，而是先将输入 \(\mathbf{h}_t\) <span style="color: red;">下投影</span>为一个低维潜在向量 \(\mathbf{c}_t^{KV}\) <span style="color: red;">（压缩）</span>：

\[
{\color{blue}\boxed{\mathbf{c}_t^{KV}}} = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}
\]

其中 \(W^{DKV} \in \mathbb{R}^{d_c \times d}\)，\(d_c\) 是远小于 \(d_h n_h\) 的潜在维度。

计算注意力时，再用两个<span style="color: red;">上投影</span>矩阵将 \(\mathbf{c}_t^{KV}\) 解压为所有头的**内容键**和**值**：

\[
\left[\mathbf{k}_{t,1}^C;\mathbf{k}_{t,2}^C;\dots;\mathbf{k}_{t,n_h}^C\right]=\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV}\in\mathbb{R}^{n_h d_h}
\]

RoPE Key 单独由输入投影并旋转得到，且在所有头间共享：

\[
{\color{blue}\boxed{\mathbf{k}_t^R}}=\operatorname{RoPE}(W^{KR}\mathbf{h}_t)\in\mathbb{R}^{d_h^R}
\]

将内容 Key 的第 \(i\) 个头分量与共享 RoPE Key 拼接，得到完整 Key：

\[
\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^C;\mathbf{k}_t^R]\in\mathbb{R}^{d_h+d_h^R},\qquad \mathbf{k}_{t,i}^C\in\mathbb{R}^{d_h},\quad i=1,\dots,n_h
\]


\[
\left[\mathbf{v}_{t,1}^C;\mathbf{v}_{t,2}^C;\dots;\mathbf{v}_{t,n_h}^C\right]=\mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}\in\mathbb{R}^{n_h d_h}
\]

其中 \(W^{UK},W^{UV}\in\mathbb{R}^{n_h d_h\times d_c}\)，\(W^{KR}\in\mathbb{R}^{d_h^R\times d}\)。推理时 KV Cache 只存储蓝框标出的 \(\mathbf{c}_t^{KV}\) 和 \(\mathbf{k}_t^R\)，不存储解压后的 \(\mathbf{k}_t^C\) 与 \(\mathbf{v}_t^C\)。

> 上投影：升维、解压 \
> 下投影：降维、压缩

#### 2. 查询的低秩压缩与解耦 RoPE

MLA 对 Query 也做了低秩压缩，并严格将**内容部分**与**RoPE 位置部分**分开。

**内容查询**由压缩后的潜在向量生成：

\[
\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \in \mathbb{R}^{d_c'}
\]
\[
\left[\mathbf{q}_{t,1}^C;\mathbf{q}_{t,2}^C;\dots;\mathbf{q}_{t,n_h}^C\right]=\mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q\in\mathbb{R}^{n_h d_h}
\]

**RoPE 查询**则单独投影并施加位置旋转，不经过低秩压缩的干扰：

\[
\left[\mathbf{q}_{t,1}^R;\mathbf{q}_{t,2}^R;\dots;\mathbf{q}_{t,n_h}^R\right]=\mathbf{q}_t^R=\operatorname{RoPE}(W^{QR}\mathbf{c}_t^Q)\in\mathbb{R}^{n_h d_h^R}
\]

最终第 \(t\) 个 token 在第 \(i\) 个头上的完整查询，由内容部分和位置部分**拼接**而成：

\[
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R] \in \mathbb{R}^{d_h+d_h^R}, \qquad i=1,\dots,n_h
\]

\(W^{DQ}\in\mathbb{R}^{d_c'\times d}\)，\(W^{UQ}\in\mathbb{R}^{n_h d_h\times d_c'}\)，\(W^{QR}\in\mathbb{R}^{n_h d_h^R\times d_c'}\)。技术报告指出 \(d_c'\ll n_h d_h\)，Query 低秩压缩可降低激活内存；KV Cache 的缓存宽度则由 \(d_c+d_h^R\) 决定，即缓存 \(\mathbf{c}_t^{KV}\) 和共享的 \(\mathbf{k}_t^R\)。

#### 3. 注意力计算
注意力计算中，报告式（3）—（4）的 Key 在历史位置 \(j\) 上写作 \(\mathbf{k}_{j,i}\)；对位置 \(t\) 的因果注意力，只使用 \(j=1,\dots,t\)：
\[
\mathbf{k}_{j,i}=[\mathbf{k}_{j,i}^C;\mathbf{k}_j^R]\in\mathbb{R}^{d_h+d_h^R},\qquad \mathbf{k}_{j,i}^C\in\mathbb{R}^{d_h},\quad \mathbf{k}_j^R\in\mathbb{R}^{d_h^R},\quad i=1,\dots,n_h
\]

第 \(i\) 个头的注意力输出为：

\[
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \operatorname{Softmax}_j \left( \frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}} \right) \mathbf{v}_{j,i}^C \in \mathbb{R}^{d_h}, \qquad \mathbf{v}_{j,i}^C\in\mathbb{R}^{d_h}
\]

所有头的输出拼接后，经输出投影矩阵 \(W^O\) 得到最终结果：

\[
\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}] \in \mathbb{R}^{d}
\]

这对应技术报告中的式（1）—（11）：KV 内容低秩压缩与解压、按头 Query 分解、解耦 RoPE Key、因果注意力及输出投影。需要特别注意，Value 只取内容部分 \(\mathbf{v}_{j,i}^C\)，不带 RoPE 位置信息。

MLA 将每个 token 的 KV Cache 从标准多头注意力下的 \(n_h(2d_h+d_h^R)\) 个标量，压缩为 \(d_c+d_h^R\) 个标量：缓存共享潜向量 \(\mathbf{c}_t^{KV}\) 和共享 RoPE Key \(\mathbf{k}_t^R\)，而不是完整的各头内容 Key/Value。DeepSeek-V3 技术报告将 MLA 确认为高效推理的重要架构设计。


## Llama MHA vs. DeekSeek MLA


| 符号 | 含义 | Llama 2-7B<br>实际值 | DeepSeek-V3<br>实际值 |
| :--- | :--- | :--- | :--- |
| \(n\) | 序列长度，即 token 数 | — | — |
| \(d\) | 模型隐藏维度。MHA并不要求$d=n_h*d_h$ | \(4096\) | \(7168\) |
| \(n_h\) | 注意力头数 | \(32\) | \(128\) |
| \(d_h\) | 每个头的内容/Value 维度；DeepSeek-V3 的完整 Q/K 头维度为 \(d_h+d_h^R\) | \(128\) | \(128\) |
| \(d_c\) | DeepSeek-V3 中 KV 联合压缩后的潜在维度 | — | \(512\) |
| \(d_c'\) | DeepSeek-V3 中 Query 压缩后的潜在维度，满足 \(d_c'\ll d_h n_h\) | — | \(1536\) |
| \(d_h^R\) | 每个头的解耦 RoPE 子维度；Key 的 RoPE 分量在所有头间共享 | — | \(64\) |
| \(H\in\mathbb{R}^{n\times d}\) | 输入隐藏矩阵，第 \(t\) 行为 \(\mathbf{h}_t\) | \(H\)\(\in\mathbb{R}^{n\times4096}\) | \(H\)\(\in\mathbb{R}^{n\times7168}\) |
| \(\mathbf{h}_t\in\mathbb{R}^{d}\) | 第 \(t\) 个 token 的隐藏向量 | \(\mathbf{h}_t\)\(\in\mathbb{R}^{4096}\) | \(\mathbf{h}_t\)\(\in\mathbb{R}^{7168}\) |
| \(M\in\mathbb{R}^{n\times n}\) | causal mask，\(M_{t,j}=0\) 当 \(j\le t\)，否则 \(M_{t,j}=-\infty\) | \(n\times n\) | \(n\times n\) |
| \(W_Q^{(i)},W_K^{(i)},W_V^{(i)}\in\mathbb{R}^{d_h\times d}\) | Llama 2-7B 第 \(i\) 个头的 Q/K/V 投影矩阵 | \(128\times4096\) | — |
| \(W^O\in\mathbb{R}^{d\times n_h d_h}\) | 输出投影矩阵 | \(4096\times4096\) | \(7168\times16384\) |
| \(W^{DQ}\in\mathbb{R}^{d_c'\times d}\) | DeepSeek-V3 Query 下投影 | — | \(1536\times7168\) |
| \(W^{UQ}\in\mathbb{R}^{n_h d_h\times d_c'}\) | DeepSeek-V3 Query 上投影 | — | \(16384\times1536\) |
| \(W^{QR}\in\mathbb{R}^{n_h d_h^R\times d_c'}\) | DeepSeek-V3 Query 的 RoPE 投影 | — | \(8192\times1536\) |
| \(W^{DKV}\in\mathbb{R}^{d_c\times d}\) | DeepSeek-V3 KV 联合下投影 | — | \(512\times7168\) |
| \(W^{UK},W^{UV}\in\mathbb{R}^{n_h d_h\times d_c}\) | DeepSeek-V3 K/V 上投影 | — | \(16384\times512\) |
| \(W^{KR}\in\mathbb{R}^{d_h^R\times d}\) | DeepSeek-V3 Key 的 RoPE 投影 | — | \(64\times7168\) |
| \(C^Q,C^{KV}\) | DeepSeek-V3 压缩潜在矩阵 | — | \(C^Q\in\mathbb{R}^{n\times1536}\)<br>\(C^{KV}\in\mathbb{R}^{n\times512}\) |
| \(Q_i,K_i,V_i,O_i\) | 第 \(i\) 个头的 Q/K/V/O 矩阵 | \(Q_i,K_i,V_i,O_i\in\mathbb{R}^{n\times128}\) | \(Q_i,K_i\in\mathbb{R}^{n\times192}\)<br>\(V_i,O_i\in\mathbb{R}^{n\times128}\) |
| \(Q_i^C,K_i^C,V_i^C\) | DeepSeek-V3 第 \(i\) 个头的内容部分矩阵 | — | \(Q_i^C,K_i^C,V_i^C\in\mathbb{R}^{n\times128}\) |
| \(Q^R,K^R\) | DeepSeek-V3 按头拼接的 Query RoPE 部分与所有头共享的 Key RoPE 部分 | — | \(Q^R\in\mathbb{R}^{n\times8192}\)<br>\(K^R\in\mathbb{R}^{n\times64}\) |
| \(S_i,A_i\) | 第 \(i\) 个头的注意力打分矩阵和注意力权重矩阵 | \(S_i,A_i\in\mathbb{R}^{n\times n}\) | \(S_i,A_i\in\mathbb{R}^{n\times n}\) |
| \(\operatorname{RoPE}(\cdot)\) | 旋转位置编码 | 作用于 Q/K 的<br>\(128\) 维头向量 | 作用于每个头 \(64\) 维的 Query RoPE 分量，以及所有头共享的 \(64\) 维 Key RoPE 向量 |
| \([X,Y]\) | 最后一维拼接 | 多头拼接后为<br>\(n\times4096\) | 单头 Q/K 拼接后为<br>\(n\times192\)；<br>内容或 Value 多头拼接后为<br>\(n\times16384\) |
| \(\operatorname{Softmax}\) | 矩阵公式中表示对矩阵每一行沿列维 \(j\) 做归一化；即固定 query 位置 \(t\)，对历史 key 位置 \(j\) 归一化，且已含 causal mask | 作用于<br>\(S_i\in\mathbb{R}^{n\times n}\) | 作用于<br>\(S_i\in\mathbb{R}^{n\times n}\) |

---

| 模型 | 组件 | 向量公式 | 矩阵公式 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 2-7B** | **Q** | \( \mathbf{q}_{t,i} = \operatorname{RoPE}\left(W_Q^{(i)} \mathbf{h}_t\right) \) | \( Q_i=\operatorname{RoPE}\left(H(W_Q^{(i)})^T\right)\in\mathbb{R}^{n\times d_h} \)<br>\( Q=[Q_1,\dots,Q_{n_h}]\in\mathbb{R}^{n\times n_h d_h} \) | \(\mathbf{q}_{t,i}\) 中，\(t=1,\dots,n\) 表示 Query token 位置，\(i=1,\dots,n_h\) 表示注意力头；\(Q_i\) 是第 \(i\) 个头的矩阵，\(Q\) 沿头维拼接 |
| | **K** | \( \mathbf{k}_{j,i} = \operatorname{RoPE}\left(W_K^{(i)} \mathbf{h}_j\right) \) | \( K_i=\operatorname{RoPE}\left(H(W_K^{(i)})^T\right)\in\mathbb{R}^{n\times d_h} \)<br>\( K=[K_1,\dots,K_{n_h}]\in\mathbb{R}^{n\times n_h d_h} \) | \(\mathbf{k}_{j,i}\) 中，\(j=1,\dots,n\) 是 Key 所属 token 的位置，\(i=1,\dots,n_h\) 是注意力头；query 位置为 \(t\) 时，因果注意力只使用 \(j\le t\) 的 Key；每个头独立生成并全量缓存 |
| | **V** | \( \mathbf{v}_{j,i} = W_V^{(i)} \mathbf{h}_j \) | \( V_i=H(W_V^{(i)})^T\in\mathbb{R}^{n\times d_h} \)<br>\( V=[V_1,\dots,V_{n_h}]\in\mathbb{R}^{n\times n_h d_h} \) | \(\mathbf{v}_{j,i}\) 中，\(j=1,\dots,n\) 是 Value 所属 token 的位置，\(i=1,\dots,n_h\) 是注意力头；V 不施加 RoPE |
| | **O** | \( A_{t,j,i} = \operatorname{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h}}+M_{t,j}\right) \)<br>\( \mathbf{o}_{t,i} = \sum_{j=1}^{n} A_{t,j,i} \mathbf{v}_{j,i} \)<br>\( \mathbf{O}_t = W^O [\mathbf{o}_{t,1};\dots;\mathbf{o}_{t,n_h}] \) | \( S_i=\dfrac{Q_iK_i^T}{\sqrt{d_h}}+M \)<br>\( A_i=\operatorname{Softmax}(S_i)\in\mathbb{R}^{n\times n} \)<br>\( O_i=A_iV_i\in\mathbb{R}^{n\times d_h} \)<br>\( O_{\text{cat}}=[O_1,\dots,O_{n_h}]\in\mathbb{R}^{n\times n_h d_h} \)<br>\( O=O_{\text{cat}}(W^O)^T\in\mathbb{R}^{n\times d} \) | \(\operatorname{Softmax}(S_i)\) 表示对 \(S_i\) 每一行沿 \(j\) 做归一化，已含 causal mask |
| | **KV Cache** | \(\operatorname{KVCache}_j=\{\mathbf{k}_{j,i},\mathbf{v}_{j,i}\}_{i=1}^{n_h}\) | \(K_{\mathrm{cache}}=K\in\mathbb{R}^{n\times n_h d_h}=\mathbb{R}^{n\times4096}\)<br>\(V_{\mathrm{cache}}=V\in\mathbb{R}^{n\times n_h d_h}=\mathbb{R}^{n\times4096}\) | 实际保存完整 K/V；每层每个 token 存储 \(2n_h d_h=8192\) 个标量 |
| **DeepSeek-V3** | **Q** | \( \mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \)<br>\( [\mathbf{q}_{t,1}^C;\dots;\mathbf{q}_{t,n_h}^C]=\mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q \)<br>\( [\mathbf{q}_{t,1}^R;\dots;\mathbf{q}_{t,n_h}^R]=\mathbf{q}_t^R=\operatorname{RoPE}(W^{QR}\mathbf{c}_t^Q) \)<br>\( \mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C;\mathbf{q}_{t,i}^R] \) | \( C^Q=H(W^{DQ})^T\in\mathbb{R}^{n\times d_c'} \)<br>\( Q^C=C^Q(W^{UQ})^T\in\mathbb{R}^{n\times n_h d_h} \)<br>按头拆分：\(Q_i^C\in\mathbb{R}^{n\times d_h}\)<br>\( Q^R=\operatorname{RoPE}\left(C^Q(W^{QR})^T\right)\in\mathbb{R}^{n\times n_h d_h^R} \)<br>按头拆分：\(Q_i^R\in\mathbb{R}^{n\times d_h^R}\)<br>\( Q_i=[Q_i^C,Q_i^R]\in\mathbb{R}^{n\times(d_h+d_h^R)} \)<br>\( Q=[Q_1,\dots,Q_{n_h}]\in\mathbb{R}^{n\times n_h(d_h+d_h^R)} \) | \(\mathbf{q}_{t,i}^C,\mathbf{q}_{t,i}^R,\mathbf{q}_{t,i}\) 中，\(t=1,\dots,n\) 表示 Query token 位置，\(i=1,\dots,n_h\) 表示注意力头；\(\mathbf{q}_t^C,\mathbf{q}_t^R\) 分别是按头拼接的内容查询和 RoPE 查询，\(Q_i^R\) 是第 \(i\) 个头的 RoPE 矩阵 |
| | **K** | \( {\color{blue}\boxed{\mathbf{c}_t^{KV}}} = W^{DKV} \mathbf{h}_t \)<br>\( \mathbf{k}_{t,i}^C = W_i^{UK} \mathbf{c}_t^{KV} \)<br>\( {\color{blue}\boxed{\mathbf{k}_t^R}} = \operatorname{RoPE}(W^{KR} \mathbf{h}_t) \)<br>\( \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C;\mathbf{k}_t^R] \) | \( C^{KV}=H(W^{DKV})^T\in\mathbb{R}^{n\times d_c} \)<br>\( K^C=C^{KV}(W^{UK})^T\in\mathbb{R}^{n\times n_h d_h} \)<br>按头拆分：\(K_i^C\in\mathbb{R}^{n\times d_h}\)<br>\( K^R=\operatorname{RoPE}\left(H(W^{KR})^T\right)\in\mathbb{R}^{n\times d_h^R} \)<br>\( K_i=[K_i^C,K^R]\in\mathbb{R}^{n\times(d_h+d_h^R)} \)<br>\( K=[K_1,\dots,K_{n_h}]\in\mathbb{R}^{n\times n_h(d_h+d_h^R)} \) | 生成当前 token 的 Key 时，\(t\) 是 token 位置，\(i\) 是头索引；报告式（2）—（4）用 \(t\) 表示正在生成的 Key，式（10）在注意力中以 \(j\le t\) 遍历历史 Key，故同一向量写作 \(\mathbf{k}_{j,i}\)（RoPE 分量为 \(\mathbf{k}_j^R\)）。\(t\) 与 \(j\) 的用法有意区分，并非笔误 |
| | **V** | \( \left[\mathbf{v}_{t,1}^C;\dots;\mathbf{v}_{t,n_h}^C\right]=\mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV} \) | \( V^C=C^{KV}(W^{UV})^T\in\mathbb{R}^{n\times n_h d_h} \)<br>按头拆分：\(V_i^C\in\mathbb{R}^{n\times d_h}\)<br>\( V=[V_1^C,\dots,V_{n_h}]\in\mathbb{R}^{n\times n_h d_h} \) | \(\mathbf{v}_t^C\) 是 token \(t\) 的多头拼接 Value；注意力中的 \(\mathbf{v}_{j,i}^C\) 以 \(j\) 标记 Value 所属位置、以 \(i\) 标记注意力头，由 \(\mathbf{c}_j^{KV}\) 解压得到 |
| | **O** | \( \mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h+d_h^R}}\right)\mathbf{v}_{j,i}^C\in\mathbb{R}^{d_h} \)<br>\( \mathbf{O}_t = W^O [\mathbf{o}_{t,1};\dots;\mathbf{o}_{t,n_h}] \) | \( S_i=\dfrac{Q_iK_i^T}{\sqrt{d_h+d_h^R}}+M \)<br>\( A_i=\operatorname{Softmax}(S_i)\in\mathbb{R}^{n\times n} \)<br>\( O_i=A_iV_i^C\in\mathbb{R}^{n\times d_h} \)<br>\( O_{\text{cat}}=[O_1,\dots,O_{n_h}]\in\mathbb{R}^{n\times n_h d_h} \)<br>\( O=O_{\text{cat}}(W^O)^T\in\mathbb{R}^{n\times d} \) | \(\operatorname{Softmax}(S_i)\) 表示对 \(S_i\) 每一行沿 \(j\) 归一化，causal mask 已包含于 \(M\)；V 只取内容部分 |
| | **KV Cache** | \(\operatorname{KVCache}_j=\{{\color{blue}\boxed{\mathbf{c}_j^{KV}}},{\color{blue}\boxed{\mathbf{k}_j^R}}\}\) | \(C_{\mathrm{cache}}^{KV}={\color{blue}\boxed{C^{KV}}}\in\mathbb{R}^{n\times d_c}=\mathbb{R}^{n\times512}\)<br>\(K_{\mathrm{cache}}^R={\color{blue}\boxed{K^R}}\in\mathbb{R}^{n\times d_h^R}=\mathbb{R}^{n\times64}\) | 实际只保存 \(C^{KV}\) 和 \(K^R\)，不保存解压后的 \(K^C/V^C\)；每层每个 token 存储 \(d_c+d_h^R=512+64=576\) 个标量。按同配置 MHA（每头缓存 \(d_h+d_h^R\) 维 K 和 \(d_h\) 维 V），\(r_{\mathrm{cache}}=\frac{d_c+d_h^R}{n_h(2d_h+d_h^R)}=\frac{576}{40960}\approx1.41\%\)，即缓存减少约 \(98.59\%\) |


### MLA给大模型推理带来的变化

- _KVCache管理_。潜向量 $C^{KV}$是所有头共享的表示，\(K^R\) 为所有头共享的 RoPE 键，<span style="color: red;">MLA 的 KV Cache 不再具有“头”这一维度，内存管理上必须为 MLA 设计完全不同的缓存块布局。</span> [[kvcache]]
- _矩阵吸收（matrix absorption，出自DeepSeek-V2）_。
- _张量并行_。$C^{KV}$和$K^R$被所有头共享，TP需要全量复制KVCache。
- _算子融合_。FlashMLA，压缩—解压—注意力三段式流程。
- _量化与精度约束_。MLA 在低秩潜空间工作，对量化误差更为敏感。推理引擎在为 MLA 设计 FP8 等低精度解码 kernel 时，通常只对潜向量缓存做量化，而保持查询和 RoPE 分量在较高精度。

#### MLA 矩阵吸收

先约定记号。忽略 batch 和层号，固定一个 query 位置 \(t\)，历史 key 位置 \(j\le t\)。DeepSeek-V3 中：

\[
W^{UQ}\in\mathbb{R}^{n_h d_h\times d_c'},\quad
W^{UK},W^{UV}\in\mathbb{R}^{n_h d_h\times d_c},\quad
W^O\in\mathbb{R}^{d\times n_h d_h}
\]

按头分块：

\[
W^{UQ}=\begin{bmatrix}W_1^{UQ}\\ \vdots\\ W_{n_h}^{UQ}\end{bmatrix},\quad
W^{UK}=\begin{bmatrix}W_1^{UK}\\ \vdots\\ W_{n_h}^{UK}\end{bmatrix},\quad
W^{UV}=\begin{bmatrix}W_1^{UV}\\ \vdots\\ W_{n_h}^{UV}\end{bmatrix}
\]

其中 \(W_i^{UQ}\in\mathbb{R}^{d_h\times d_c'}\)，\(W_i^{UK},W_i^{UV}\in\mathbb{R}^{d_h\times d_c}\)。

##### 第一步：把注意力分数拆成“内容 + RoPE”

原始 MLA 的注意力分数（第 \(i\) 个头）：

\[
S_{t,j,i}=\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h+d_h^R}}
=\frac{(\mathbf{q}_{t,i}^C)^T\mathbf{k}_{j,i}^C}{\sqrt{d_h+d_h^R}}
+\frac{(\mathbf{q}_t^R)^T\mathbf{k}_j^R}{\sqrt{d_h+d_h^R}}
\]

其中：

\[
\mathbf{q}_{t,i}^C=W_i^{UQ}\mathbf{c}_t^Q,\qquad
\mathbf{k}_{j,i}^C=W_i^{UK}\mathbf{c}_j^{KV}
\]

\[
\mathbf{q}_t^R=\operatorname{RoPE}(W^{QR}\mathbf{c}_t^Q),\qquad
\mathbf{k}_j^R=\operatorname{RoPE}(W^{KR}\mathbf{h}_j)
\]

##### 第二步：内容分数的吸收——消掉显式 \(\mathbf{q}^C\) 和 \(\mathbf{k}^C\)

内容分数：

\[
(\mathbf{q}_{t,i}^C)^T\mathbf{k}_{j,i}^C
=(W_i^{UQ}\mathbf{c}_t^Q)^T(W_i^{UK}\mathbf{c}_j^{KV})
=(\mathbf{c}_t^Q)^T\underbrace{(W_i^{UQ})^T W_i^{UK}}_{W_i^{QK}}\mathbf{c}_j^{KV}
\]

定义**吸收后的内容打分矩阵**：

\[
\boxed{W_i^{QK}=(W_i^{UQ})^T W_i^{UK}\in\mathbb{R}^{d_c'\times d_c}}
\]

于是：

\[
(\mathbf{q}_{t,i}^C)^T\mathbf{k}_{j,i}^C=(\mathbf{c}_t^Q)^T W_i^{QK}\mathbf{c}_j^{KV}
\]

**关键变化**：推理时不再需要显式构造 \(\mathbf{k}_{j,i}^C=W_i^{UK}\mathbf{c}_j^{KV}\)。所有 \(n_h\) 个 query 头**直接与同一个共享潜向量 \(\mathbf{c}_j^{KV}\) 交互**，只是各自使用不同的预计算矩阵 \(W_i^{QK}\)。

##### 第三步：V 侧的吸收——消掉显式 \(\mathbf{v}^C\)

第 \(i\) 个头的输出：

\[
\mathbf{o}_{t,i}=\sum_{j\le t}A_{t,j,i}\,\mathbf{v}_{j,i}^C
=\sum_{j\le t}A_{t,j,i}\,W_i^{UV}\mathbf{c}_j^{KV}
\]

因为 \(W_i^{UV}\) 与 \(j\) 无关，可以提出求和号：

\[
\mathbf{o}_{t,i}=W_i^{UV}\underbrace{\sum_{j\le t}A_{t,j,i}\,\mathbf{c}_j^{KV}}_{\tilde{\mathbf{o}}_{t,i}\in\mathbb{R}^{d_c}}
\]

定义**压缩空间中的中间输出**：

\[
\boxed{\tilde{\mathbf{o}}_{t,i}=\sum_{j\le t}A_{t,j,i}\,\mathbf{c}_j^{KV}}
\]

则：

\[
\mathbf{o}_{t,i}=W_i^{UV}\tilde{\mathbf{o}}_{t,i}
\]

把所有头拼起来：

\[
[\mathbf{o}_{t,1};\dots;\mathbf{o}_{t,n_h}]
=W^{UV}_{\text{blk}}\,[\tilde{\mathbf{o}}_{t,1};\dots;\tilde{\mathbf{o}}_{t,n_h}]
\]

其中：

\[
W^{UV}_{\text{blk}}=\operatorname{diag}(W_1^{UV},\dots,W_{n_h}^{UV})\in\mathbb{R}^{n_h d_h\times n_h d_c}
\]

$\operatorname{diag}$是分块对角矩阵。

再乘输出投影 \(W^O\)：

\[
\mathbf{u}_t=W^O[\mathbf{o}_{t,1};\dots;\mathbf{o}_{t,n_h}]
=W^O W^{UV}_{\text{blk}}\,[\tilde{\mathbf{o}}_{t,1};\dots;\tilde{\mathbf{o}}_{t,n_h}]
\]

定义**吸收后的输出投影**：

\[
\boxed{W^{O,UV}=W^O W^{UV}_{\text{blk}}\in\mathbb{R}^{d\times n_h d_c}}
\]

于是：

\[
\mathbf{u}_t=W^{O,UV}\,[\tilde{\mathbf{o}}_{t,1};\dots;\tilde{\mathbf{o}}_{t,n_h}]
\]

**关键变化**：推理时不再需要显式构造 \(\mathbf{v}_{j,i}^C=W_i^{UV}\mathbf{c}_j^{KV}\)。注意力加权直接在压缩潜空间完成，得到 \(\tilde{\mathbf{o}}_{t,i}\)，最后用预计算的 \(W^{O,UV}\) 一次性投影回模型维度。

##### 第四步：NoPE 情形下的完全吸收

若无 RoPE（即去掉 \(\mathbf{q}_t^R,\mathbf{k}_j^R\)），注意力权重为：

\[
A_{t,j,i}=\operatorname{Softmax}_j\left(
\frac{(\mathbf{c}_t^Q)^T W_i^{QK}\mathbf{c}_j^{KV}}{\sqrt{d_h}}
\right)
\]

完整输出：

\[
\mathbf{u}_t=W^{O,UV}\Big[
\tilde{\mathbf{o}}_{t,1};\dots;\tilde{\mathbf{o}}_{t,n_h}
\Big],\qquad
\tilde{\mathbf{o}}_{t,i}=\sum_{j\le t}A_{t,j,i}\,\mathbf{c}_j^{KV}
\]

此时推理的计算图变为：

\[
\mathbf{c}_t^Q \;\longrightarrow\; \big\{(\mathbf{c}_t^Q)^T W_i^{QK}\big\}_{i=1}^{n_h}
\;\longrightarrow\; A_{t,j,i}
\;\longrightarrow\; \tilde{\mathbf{o}}_{t,i}
\;\longrightarrow\; \mathbf{u}_t
\]

**KV Cache 中只存 \(\mathbf{c}_j^{KV}\)**，所有 \(n_h\) 个 query 头共享这一个表示。这在缓存结构和注意力形式上与 MQA 完全同构：

| | MQA | MLA（NoPE 吸收后） |
| :--- | :--- | :--- |
| 共享 KV 缓存 | 一个 \(d_h\) 维 K/V 头 | 一个 \(d_c\) 维潜向量 \(\mathbf{c}^{KV}\) |
| Query 头数 | \(n_h\) | \(n_h\) |
| 每头如何与共享 KV 交互 | 直接点积 | 通过预计算 \(W_i^{QK}\) 变换后点积 |
| 输出如何回到模型维度 | \(W^O\) | 预计算 \(W^{O,UV}\) |

所以“变身 MQA”的准确含义是：**吸收后，KV 侧退化为一个所有头共享的压缩表示，Query 侧仍保留 \(n_h\) 个头的独立打分能力**。它不是严格意义上的 MQA（MQA 的共享 KV 是显式高维向量，MLA 的是低秩潜向量），但在**缓存布局和访存模式**上表现为 MQA 风格。

##### 第五步：RoPE 为什么不能被吸收

有 RoPE 时，位置分数为：

\[
(\mathbf{q}_t^R)^T\mathbf{k}_j^R
=\big(\operatorname{RoPE}(W^{QR}\mathbf{c}_t^Q)\big)^T
\big(\operatorname{RoPE}(W^{KR}\mathbf{h}_j)\big)
\]

RoPE 是对向量施加**位置相关的旋转**：

\[
\operatorname{RoPE}(\mathbf{x},p)=R_p\,\mathbf{x}
\]

其中 \(R_p\) 是随位置 \(p\) 变化的旋转矩阵。因此：

\[
(\mathbf{q}_t^R)^T\mathbf{k}_j^R
=(R_t W^{QR}\mathbf{c}_t^Q)^T(R_j W^{KR}\mathbf{h}_j)
=(\mathbf{c}_t^Q)^T(W^{QR})^T R_t^T R_j W^{KR}\mathbf{h}_j
\]

中间夹着 \(R_t^T R_j\)，它**依赖 \(t\) 和 \(j\) 的相对位置**，不能预先合并成一个与位置无关的矩阵。所以 \(W^{QR}\) 和 \(W^{KR}\) 无法像 \(W^{UQ},W^{UK}\) 那样被吸收掉。这就是 MLA 必须把 RoPE 分量**解耦出来单独计算**的根本原因。

吸收后的完整推理公式为：

\[
\boxed{
A_{t,j,i}=\operatorname{Softmax}_j\left(
\frac{(\mathbf{c}_t^Q)^T W_i^{QK}\mathbf{c}_j^{KV}
+(\mathbf{q}_t^R)^T\mathbf{k}_j^R}
{\sqrt{d_h+d_h^R}}
\right)
}
\]

\[
\boxed{
\mathbf{u}_t=W^{O,UV}\Big[
\tilde{\mathbf{o}}_{t,1};\dots;\tilde{\mathbf{o}}_{t,n_h}
\Big],\qquad
\tilde{\mathbf{o}}_{t,i}=\sum_{j\le t}A_{t,j,i}\,\mathbf{c}_j^{KV}
}
\]

##### 推理前预计算什么，推理时缓存什么

**推理前**（离线融合，不增加运行时开销）：

\[
W_i^{QK}=(W_i^{UQ})^T W_i^{UK},\qquad
W^{O,UV}=W^O\operatorname{diag}(W_1^{UV},\dots,W_{n_h}^{UV})
\]

**推理时 KV Cache 存储**：

\[
\mathbf{c}_j^{KV}\in\mathbb{R}^{d_c}\quad\text{（所有头共享）},\qquad
\mathbf{k}_j^R\in\mathbb{R}^{d_h^R}\quad\text{（所有头共享）}
\]

**不再存储**：\(\mathbf{k}_{j,i}^C\)、\(\mathbf{v}_{j,i}^C\) 这些按头展开的显式高维向量。

##### 总结

\[
\boxed{
\begin{aligned}
&\text{内容分数：}(\mathbf{q}_{t,i}^C)^T\mathbf{k}_{j,i}^C
=(\mathbf{c}_t^Q)^T\underbrace{(W_i^{UQ})^T W_i^{UK}}_{W_i^{QK}}\mathbf{c}_j^{KV}\\
&\text{V 侧：}\mathbf{o}_{t,i}=W_i^{UV}\tilde{\mathbf{o}}_{t,i},\quad
\tilde{\mathbf{o}}_{t,i}=\sum_j A_{t,j,i}\mathbf{c}_j^{KV}\\
&\text{输出：}\mathbf{u}_t=W^{O,UV}[\tilde{\mathbf{o}}_{t,1};\dots;\tilde{\mathbf{o}}_{t,n_h}]
\end{aligned}
}
\]

矩阵吸收的本质是**利用线性映射的结合律，把上投影矩阵提前融合进相邻矩阵**，使推理时 KV 侧只剩一个所有头共享的压缩潜向量 \(\mathbf{c}_j^{KV}\)——这正是“MLA 在推理时变身 MQA 风格”的数学来源。RoPE 因位置相关旋转不可吸收，必须作为独立的共享分量保留。

## DeepSeek MoE

V3 拥有 **671B 总参数**，但每个 token 仅激活 **37B**，激活率约 5.5%。这种解耦使得模型能存储海量知识，同时将推理计算成本控制在一个中等规模密集模型的水平。

传统 MoE（如 GShard）面临两个核心痛点：**知识混杂**（少数大专家被迫学习差异极大的知识）与**知识冗余**（不同专家重复存储通用常识）。DeepSeekMoE 通过两项基础设计来应对：

**1. 细粒度专家分割**
在保持总参数不变的前提下，将专家的 FFN 隐藏维度切得更细，从而用更多数量的小专家来替代少数大专家。V3 的每个 MoE 层包含 **256 个路由专家**，每个专家的中间隐藏维度为 2048。这让专家能更灵活地组合，提升专业化程度。

**2. 共享专家隔离**
设置少量**始终激活**的共享专家，专门负责捕获跨上下文的通用知识；路由专家则得以更专注于各自的特异性模式，从而减少路由专家之间的知识冗余。

在 V3 的 61 层中，**除前三层保留稠密 FFN 外，其余 58 层均采用上述 MoE 结构**。

### 辅助损失自由的全局负载均衡

**问题根源**：在专家并行场景下，负载不均会导致“路由坍塌”——少数专家被过度使用，其余专家因梯度稀少而无法有效训练。传统方案是引入**辅助损失**来惩罚不均衡，但这相当于在语言建模的主目标之外“掺入干扰”，以牺牲模型性能为代价换取负载均衡。

**V3 的解法**：取消以全局负载均衡为目标的辅助损失，并为每个路由专家维护一个**由负载动态调整的偏置项 \(b_i\)**。路由决策时，模型用 **sigmoid 门控**为每个 token 打分，再将偏置项加到分数上，选取分数最高的 Top-K 个专家。关键在于，**偏置项仅用于路由选择，不参与最终的门控权重计算**。专家负载偏高时调低其偏置，负载偏低时调高其偏置。

这样，负载均衡通过一个**工程层面的动态调节机制**来实现，主训练目标不再受到辅助损失的污染，从而在负载均衡与模型性能之间取得了更优的平衡。作为补充，V3 仍保留了一个**极小的序列级辅助损失**，仅用于防止单个序列内部出现极端失衡，而非全局负载调控。

### 训练时路由约束

训练时，每个token被发到最多4个节点。

### 推理阶段的 MoE 公式

以下公式描述单个 MoE 层对 token 的前向计算。设 \(\ell\) 为层编号；DeepSeek-V3 前 3 层使用稠密 FFN，后 58 层使用 MoE。对任一 MoE 层和 token 位置 \(t\)，令 \(\mathbf{u}_t\in\mathbb{R}^{d}\) 为 FFN 输入。为简洁起见，省略层编号；专家参数和路由偏置均属于当前层。

#### 1. 计算路由分数并选择专家

对 \(N_r\) 个路由专家分别计算 token 与专家的亲和分数：

\[
s_{i,t}=\operatorname{Sigmoid}(\mathbf{u}_t^T\mathbf{e}_i),\qquad i=1,\dots,N_r
\]

其中 \(\mathbf{e}_i\) 是第 \(i\) 个路由专家的中心向量（centroid vector）。路由时加入偏置 \(b_i\) 以选择 Top-K 专家，但偏置不参与最终门控权重：

\[
g'_{i,t}=\begin{cases}
s_{i,t}, & s_{i,t}+b_i\in\operatorname{TopK}\left(\{s_{j,t}+b_j\mid 1\le j\le N_r\},K_r\right),\\
0, & \text{otherwise}
\end{cases}
\]

推理时使用训练完成后固定的路由偏置；训练期间根据专家负载调整偏置的过程不属于这次前向计算。

#### 2. 由原始亲和分数计算门控权重

被选中的专家使用未加偏置的亲和分数作为权重，并在选中的专家之间归一化：

\[
g_{i,t}=\frac{g'_{i,t}}{\displaystyle\sum_{j=1}^{N_r}g'_{j,t}}
\]

因此，偏置 \(b_i\) 只影响 Top-K 专家的选择；门控权重 \(g_{i,t}\) 仍由未加偏置的亲和分数 \(s_{i,t}\) 归一化得到。

#### 3. 合并共享专家与路由专家输出

MoE 层输出为残差输入、始终激活的共享专家输出，以及 Top-K 路由专家的加权输出之和：

\[
\mathbf{h}'_t=\mathbf{u}_t+
\sum_{i=1}^{N_s}\operatorname{FFN}^{(s)}_i(\mathbf{u}_t)+
\sum_{i=1}^{N_r}g_{i,t}\operatorname{FFN}^{(r)}_i(\mathbf{u}_t)
\]

其中 \(\operatorname{FFN}^{(s)}_i\) 和 \(\operatorname{FFN}^{(r)}_i\) 分别表示共享专家和路由专家；未满足 Top-K 条件的路由专家具有 \(g_{i,t}=0\)，因此推理时只需计算被选中的 \(K_r\) 个路由专家。共享专家不经过 Top-K 路由，始终参与计算。

#### 4. DeepSeek-V3 配置代入

DeepSeek-V3 每个 MoE 层使用 \(N_s=1\) 个共享专家、\(N_r=256\) 个路由专家，并取 \(K_r=8\)。因此每个 token 在该层激活 1 个共享专家和 8 个路由专家，共 9 个专家；每个路由专家的中间维度为 \(2048\)。模型共 61 层，其中 58 层使用该 MoE 结构。推理时不计算训练用的序列级负载均衡辅助损失；技术报告说明推理阶段不会丢弃 token。

以上路由与合并公式对应 DeepSeek-V3 技术报告 §2.1.2 的式（12）—（16）：[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)。

### 单个专家内部：SwiGLU FFN

**DeepSeek-V3 的单个专家与 Llama 2-7B 的单层 FFN 都采用无偏置的 SwiGLU，区别主要是输入/输出维度和中间维度。** 这里比较的是一个专家与一个稠密 FFN，不是整个 MoE 层与整个 Llama 模型。

沿用前文的 FFN 输入 \(\mathbf{u}_t\in\mathbb{R}^{d}\)，记中间维度为 \(d_{\mathrm{ff}}\)，输出为 \(\mathbf{y}_t\in\mathbb{R}^{d}\)。按两者官方实现中的 `w1`、`w3`、`w2` 命名，单个 FFN 的完整推理公式为：

\[
\boxed{
\mathbf{y}_t=\operatorname{FFN}(\mathbf{u}_t)
=W_2\left[\operatorname{SiLU}(W_1\mathbf{u}_t)\odot(W_3\mathbf{u}_t)\right]
}
\]

其中 \(\operatorname{SiLU}(x)=x\operatorname{Sigmoid}(x)\) 按元素作用，\(\odot\) 表示逐元素乘法。\(W_1\) 是门控分支投影，\(W_3\) 是另一条线性分支投影，\(W_2\) 将中间结果投影回模型隐藏维度；两条输入分支是并行计算，不是依次串联。

| 符号 / 组件 | 逐 token 推理公式或含义 | Llama 2-7B<br>单层 FFN | DeepSeek-V3<br>单个路由专家 |
| :--- | :--- | :--- | :--- |
| \(d\) | 输入和输出隐藏维度 | \(4096\) | \(7168\) |
| \(d_{\mathrm{ff}}\) | 每条分支的中间维度；不是注意力头维度 \(d_h\) | \(11008\) | \(2048\) |
| \(\mathbf{u}_t\in\mathbb{R}^{d}\) | 当前 token 的 FFN 输入 | \(\mathbb{R}^{4096}\) | \(\mathbb{R}^{7168}\) |
| \(W_1\in\mathbb{R}^{d_{\mathrm{ff}}\times d}\) | 门控投影后施加激活：<br>\(\mathbf{a}_t=\operatorname{SiLU}(W_1\mathbf{u}_t)\) | \(W_1\in\mathbb{R}^{11008\times4096}\)<br>\(\mathbf{a}_t\in\mathbb{R}^{11008}\) | \(W_1\in\mathbb{R}^{2048\times7168}\)<br>\(\mathbf{a}_t\in\mathbb{R}^{2048}\) |
| \(W_3\in\mathbb{R}^{d_{\mathrm{ff}}\times d}\) | 线性分支，不施加激活：<br>\(\mathbf{v}_t^{\mathrm{ff}}=W_3\mathbf{u}_t\) | \(W_3\in\mathbb{R}^{11008\times4096}\)<br>\(\mathbf{v}_t^{\mathrm{ff}}\in\mathbb{R}^{11008}\) | \(W_3\in\mathbb{R}^{2048\times7168}\)<br>\(\mathbf{v}_t^{\mathrm{ff}}\in\mathbb{R}^{2048}\) |
| \(\mathbf{z}_t\in\mathbb{R}^{d_{\mathrm{ff}}}\) | SwiGLU 逐元素门控：<br>\(\mathbf{z}_t=\mathbf{a}_t\odot\mathbf{v}_t^{\mathrm{ff}}\) | \(\mathbb{R}^{11008}\) | \(\mathbb{R}^{2048}\) |
| \(W_2\in\mathbb{R}^{d\times d_{\mathrm{ff}}}\) | 输出投影：<br>\(\mathbf{y}_t=W_2\mathbf{z}_t\) | \(W_2\in\mathbb{R}^{4096\times11008}\)<br>\(\mathbf{y}_t\in\mathbb{R}^{4096}\) | \(W_2\in\mathbb{R}^{7168\times2048}\)<br>\(\mathbf{y}_t\in\mathbb{R}^{7168}\) |
| 线性层偏置 | 三个投影均不带 bias；与前文路由偏置 \(b_i\) 无关 | 无 | 无 |
| \(P_{\mathrm{FFN}}=3d\,d_{\mathrm{ff}}\) | 仅计三个投影矩阵的权重参数；不计路由器、归一化及量化元数据 | \(3\times4096\times11008\)<br>\(=135{,}266{,}304\)<br>约 \(135.27\) M | \(3\times7168\times2048\)<br>\(=44{,}040{,}192\)<br>约 \(44.04\) M |
| 何时执行 | 指当前层中该 FFN / 专家是否参与某个 token 的计算 | 每个 token 都执行该 FFN | 仅在该专家被 Top-8 路由选中时执行 |

**与上文 MoE 公式的衔接**：对第 \(i\) 个路由专家，将上式 \(W_1,W_2,W_3\) 换成它独立的 \(W_{1,i}^{(r)},W_{2,i}^{(r)},W_{3,i}^{(r)}\)，即可得到 \(\operatorname{FFN}^{(r)}_i(\mathbf{u}_t)\)。

依据：[DeepSeek-V3 官方实现 `Expert` / `MLP`](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)、[V3 671B 配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json)、[Llama 官方实现 `FeedForward`](https://github.com/meta-llama/llama/blob/main/llama/model.py)。Llama 2-7B 的中间维度按官方实现由 \(\frac{8}{3}d\) 向上对齐至 \(256\) 的整数倍，得到 \(11008\)。

## DeepSeek V3 Dense FFN和MoE FFN对比

| 结构 | 层数 / 配置 | 中间维度 /<br>路由维度 | 单个 FFN 参数量<br>（估算） | Router 每层参数量<br>（含偏置，估算） | 所有 FFN 总参数量<br>（不含 Router） | 所有 FFN + Router<br>总参数量（估算） |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense FFN**<br>（前 3 层） | 3 层，每层 1 个密集 FFN | \( d_{\text{dense}} = 18432 \)<br>\(d=7168\) | \( 3 \times 7168 \times 18432 \)<br>\( = 396.361728\text{M} \) | 0 | \( 3 \times 396.361728\text{M} \)<br>\( = 1.189085184\text{B} \) | \( 1.189085184\text{B} \) |
| **MoE 专家 FFN**<br>（后 58 层） | 58 层，每层<br>256 路由专家 + 1 共享专家<br>= 257 个专家 | \( d_{\text{moe}} = 2048 \)<br>路由：\(N_r=256\)<br>\(d=7168\) | \( 3 \times 7168 \times 2048 \)<br>\( = 44.040192\text{M} \) | \( 256 \times 7168 + 256 \)<br>\( = 1{,}835{,}264 \)<br>\( \approx 1.835264\text{M} \)<br>其中偏置 \(b_i\)<br>共 256 个标量 | 每层：<br>\(257 \times 44.040192\text{M}\)<br>\(= 11.318329344\text{B}\)<br>58 层：<br>\(58 \times 11.318329344\text{B}\)<br>\(= 656.463101952\text{B}\) | 每层：<br>\(11.318329344\text{B}\)<br>\(+ 1.835264\text{M}\)<br>\(= 11.320164608\text{B}\)<br>58 层：<br>\(58 \times 11.320164608\text{B}\)<br>\(= 656.569547264\text{B}\) |
| **全模型所有**<br>**FFN + Router 合计** | 3 层密集 + 58 层 MoE | — | — | 总 Router：<br>\(58 \times 1{,}835{,}264\)<br>\(= 106{,}445{,}312\)<br>\(\approx 0.106445312\text{B}\) | \(1.189085184\text{B}\)<br>\(+ 656.463101952\text{B}\)<br>\(= 657.652187136\text{B}\) | \(1.189085184\text{B}\)<br>\(+ 656.569547264\text{B}\)<br>\(= \mathbf{657.758632448\text{B}}\) |


### MoE算子实现

- router算子实现：打分 (Sigmoid/Softmax) → 加选择偏置 → (可选) 分组海选 → Top-K 选择 → 提取原始权重 → 归一化 → 缩放
- MoE Group GEMM
  - persistent kernel
  - 只有token数量（M轴）在变，专家形状相同且固定（N、K轴固定），“M-grouped”调度
- SwiGLU 融合实现
- 通信算子：Dispatch和Combine

#### Persistent Kernel

1. 减少Kernel反复启动开销。
2. 减轻wave quantization现象。波量化是指当总工作量（如输出 tile 数）不能被 GPU 的 SM 数量整除时，会导致最后一“波”计算中只有部分 SM 在工作，其余闲置，造成利用率的骤降。在任务数量动态变化的 MoE 中，这个问题会被进一步放大。Persistent Kernel 让固定的 CTA 集合动态地从全局队列中“领取”下一个可用 tile，而不是静态地预先分配任务。这意味着先完成的 CTA 可以立即接手新任务，从而自然地在各个 SM 之间实现了动态负载均衡，有效“填平”了波量化造成的利用率低谷。
3. 跨 Tile 的流水线化。传统模式下，CTA 在完成一个 tile 后便会退出，下一个 tile 需要重新经历完整的“启动-加载数据-计算-写回”流程，这被称为“序言”（prologue）和“尾声”（epilogue）开销。
Persistent CTA 因为持续存活，可以在处理完当前 tile 后，立即开始加载下一个 tile 所需的数据，从而将不同 tile 的“尾声”与“序言”重叠起来。这种跨 tile 的流水线化隐藏了数据搬运和初始化延迟，使得计算单元能够更持续地保持忙碌。
4. 提升缓存复用与数据局部性。在 MoE 的 Grouped GEMM 中，不同专家可能共享相同的激活或权重分片。Persistent CTA 在处理连续或相关的 tile 时，有机会在片上缓存（如 L2 cache）中复用数据。
  - 共享权重包括：
    - 共享专家。始终被激活的专家。
    - 专家绑定。跨层复用专家参数。
5. Persistent Kernel 的“常驻”特性使其成为构建 Mega-kernel。

代价：实现复杂度更高（需要手动管理任务队列和同步），并且可能因常驻而带来更高的寄存器压力或与并发内核的资源争用问题。

## 打印模型权重

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
