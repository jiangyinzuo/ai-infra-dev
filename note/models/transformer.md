## Cross-Attention

在 Transformer 架构中，Decoder 的输入主要来自 Encoder 的**最后一层的输出（Output of the final encoder layer）**。

具体来说，在标准的 Transformer 结构（如 Vaswani 等人提出的经典 Seq2Seq 模型）中，这部分是通过 **Cross-Attention（交叉注意力机制，也常被称为 Encoder-Decoder Attention）** 进入 Decoder 的。

我们可以从以下几个维度来具体拆解这个“输入”是什么、长什么样、以及它是如何被 Decoder 使用的：

### 1. 数据的具体维度和形态（是什么东西）

假设输入到 Encoder 的源语言序列长度为 $S$，Encoder 一共有 $L$ 层，模型的隐藏层维度（Hidden Dimension）为 $d_{model}$（例如 512 或 768）。

* **Encoder 的输出矩阵**：Encoder 最后一层对整个输入序列进行编码后，会输出一个形状为 **$(S, d_{model})$** 的张量（如果带 Batch 则是 **$(B, S, d_{model})$**）。
* **Key 与 Value 的来源**：这个矩阵会被线性映射（通过权重矩阵 $W_K$ 和 $W_V$）转化为 **Key (K)** 和 **Value (V)** 矩阵，输入到 Decoder 的每一个 Cross-Attention 模块中。

### 2. Decoder 是怎么接收和使用它的

在 Decoder 的每个残差块中，包含两个核心的注意力机制：

1. **Masked Self-Attention**：计算 Decoder 自身已生成文本之间的依赖关系（防止看到未来的信息）。
2. **Cross-Attention（交叉注意力）**：**这里就是接收 Encoder 输入的地方。**

在 Cross-Attention 的计算过程中：

* **Query (Q)**：来自 **Decoder 上一层**的输出（代表当前 Decoder 想要去“查询”什么信息），形状为 $(T, d_{model})$，其中 $T$ 是目标序列的长度。
* **Key (K) & Value (V)**：来自 **Encoder 的最终输出**（代表源语言/上下文的全局特征表示）。

其核心公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

通过这个操作，Decoder 的每一个 Token（或者说当前正在生成的词）都可以“回看” Encoder 输出的整句话，并根据相关性权重，从 Encoder 提供的 $V$ 中提取出与当前生成任务最相关的上下文信息。
