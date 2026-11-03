## Llama 2-7B（MHA）的 KV Cache 索引维度

Llama 2-7B 的配置：层数 \(L=32\)，头数 \(n_h=32\)，每头维度 \(d_h=128\)，隐藏维度 \(d=4096\)。MHA 中 **KV 头数 = Q 头数 = 32**，每个 Q 头有自己独立的 K/V 头。

---

### 1. 逻辑维度（数学张量形状）

每层单独维护一份 KV Cache。**单层**的 KV Cache 逻辑形状为：

\[
\text{K\_cache}^{(\ell)} \in \mathbb{R}^{B \times n_h \times S \times d_h} = \mathbb{R}^{B \times 32 \times S \times 128}
\]

\[
\text{V\_cache}^{(\ell)} \in \mathbb{R}^{B \times n_h \times S \times d_h} = \mathbb{R}^{B \times 32 \times S \times 128}
\]

其中：

| 维度 | 含义 | Llama 2-7B 取值 |
| :--- | :--- | :--- |
| \(B\) | batch size | 运行时决定 |
| \(n_h\) | KV 头数 | 32 |
| \(S\) | 已缓存序列长度（max_seq_len 或当前长度） | 运行时决定，上限 4096 |
| \(d_h\) | 每头维度 | 128 |

全模型 32 层的总逻辑形状可视为在层维上堆叠：

\[
\text{K\_cache} \in \mathbb{R}^{L \times B \times n_h \times S \times d_h} = \mathbb{R}^{32 \times B \times 32 \times S \times 128}
\]

K 和 V 各一份，共两份。

---

### 2. 逻辑索引：如何定位一个元素

对于第 \(\ell\) 层、第 \(b\) 个样本、第 \(h\) 个头、第 \(s\) 个位置、第 \(d\) 维的 K 元素，逻辑索引为五元组：

\[
(\ell,\ b,\ h,\ s,\ d)
\]

若采用**连续行优先布局** `[L, B, n_h, S, d_h]`，其线性偏移为：

\[
\text{offset} = \big(((\ell \cdot B + b) \cdot n_h + h) \cdot S + s\big) \cdot d_h + d
\]

**关键点**：在 MHA 中，头维度 \(h\) **是缓存布局中的一等维度**。每个头拥有独立的、长度为 \(S \times d_h\) 的连续存储区。推理引擎可以按头切分缓存、按头并行加载、按头做 TP 切分——这是 MHA 缓存索引最本质的特征。

---

### 3. 物理布局：推理引擎实际怎么存

上面的 `[B, n_h, S, d_h]` 是**逻辑视图**。实际推理引擎（如 vLLM 的 PagedAttention、TensorRT-LLM 的 paged KV cache）采用**分块（block）布局**，逻辑形状与物理形状分离。

**PagedAttention 风格的物理布局**：

\[
\text{physical\_cache} \in \mathbb{R}^{N_{\text{blk}} \times n_h \times \text{blk\_size} \times d_h}
\]

或按头优先组织为：

\[
\text{physical\_cache} \in \mathbb{R}^{n_h \times N_{\text{blk}} \times \text{blk\_size} \times d_h}
\]

其中：

| 符号 | 含义 |
| :--- | :--- |
| \(N_{\text{blk}}\) | 物理 block 总数 |
| \(\text{blk\_size}\) | 每个 block 容纳的 token 数（常见 16） |
| \(n_h\) | 头数，仍是一等维度 |

每个序列维护一张 **block table**，把逻辑位置 \(s\) 映射到物理 block：

\[
s \;\longmapsto\; \big(\text{block\_table}[b][\lfloor s / \text{blk\_size} \rfloor],\ s \bmod \text{blk\_size}\big)
\]

因此完整的物理索引是：

\[
(\ell,\ b,\ h,\ s) \;\rightarrow\; \big(\ell,\ h,\ \text{block\_table}[b][\lfloor s/\text{blk\_size}\rfloor],\ s \bmod \text{blk\_size}\big)
\]

**无论逻辑还是物理布局，头维度 \(h\) 始终保留**，这使得 MHA 的缓存可以：

- 按头切分到不同 TP rank；
- 按头并行做注意力 kernel 的 head 维展开；
- 按头独立做量化/压缩。

---

### 4. 与 GQA / MLA 的索引维度对比

| 模型类型 | KV 头数 | 单层 KV Cache 逻辑形状 | 头维度是否可切分 | 每 token 每层 KV 字节数（FP16） |
| :--- | :--- | :--- | :--- | :--- |
| **MHA**（Llama 2-7B） | \(n_h=32\) | \([B,\ 32,\ S,\ 128]\) | ✅ 可按 32 头切分 | \(2 \times 32 \times 128 \times 2 = 16\text{ KB}\) |
| **GQA**（Llama 2-70B） | \(n_{kv}=8\) | \([B,\ 8,\ S,\ 128]\) | ✅ 可按 8 头切分 | \(2 \times 8 \times 128 \times 2 = 4\text{ KB}\) |
| **MLA**（DeepSeek-V3） | 无“头”概念 | \([B,\ S,\ d_c]\) + RoPE 共享 | ❌ 不可按头切分 | 约 576 B（含 RoPE 分量） |

**MHA 与 GQA 的索引结构是同构的**，区别仅在 KV 头数从 32 降到 8，缓存缩小 4 倍，但“头是一等索引维度”这一点不变。**MLA 则彻底取消了 KV 缓存中的头维度**：缓存的是所有头共享的 \(C^{KV}\)，索引退化为 \((\ell, b, s, d)\) 四元组，头维度消失——这正是 MLA 无法沿头做 TP 切分、必须全量复制的根本原因。

