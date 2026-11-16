attention: 上下文关系
ffn：通用知识，ffn参数量占的很多

embedding: 相似的名词应该高向量相似度，word embedding所在的空间叫feature space

## Encoder - Decoder 发展脉络

https://zenodo.org/records/22785090#1

从 Transformer 到 DeepSeek V4.1，架构并非盲目变动，而是每一次都被当时最尖锐的瓶颈所驱动。

### 序列转导（2017 Transformer）

Transformer encoder-decoder，为固定输入到固定输出的序列转导（如机器翻译）量身定制。
- Encoder 双向读取源序列，建立完整的上下文表示；
- Decoder 通过 cross-attention 在生成每个目标词时反复“查阅”这个表示。
- “先完整理解，再逐步生成”，适合输入边界清晰、输出长度可控的任务

### 理解与生成分岔（2018-2019）

BERT（encoder-only） 保留了双向注意力，用掩码语言建模（MLM）强迫模型从两侧语境中还原被遮蔽的词。这赋予了它当时最强的固定文本理解能力，但它天然不擅长开放式生成。

GPT（decoder-only） 则砍掉了 encoder 和 cross-attention，只保留因果自注意力，目标简化为“根据左侧上下文预测下一个词”。在当时，这看起来像是一种功能上的“阉割”——它没有 BERT 那种对输入的完整双向理解。但它有一个隐蔽的结构优势：阅读提示和生成回答共享同一组参数、同一条计算路径，这为后来的规模化埋下了伏笔。

T5（encoder-decoder） 试图用“一切皆 text-to-text”的统一框架弥合两者，并在翻译、摘要等输入边界清晰的序列到序列任务上保持了优势。但它的结构更复杂，且在开放式生成上并不比 decoder-only 更“通用”。

### decoder-only规模化（2020-2022）

规模催生in-context learning：只要在提示中给几个示例，decoder-only 模型就能在不更新权重的情况下执行新任务。
模型能否在scale时吸收更多数据，比它在特定任务上的精致分工更重要。

Decoder-only 胜出的原因：

- 训练信号更密集：每个非填充位置都贡献损失，而 BERT 的 MLM 只对约 15% 的掩码位置计算损失。
- 工程与推理路径统一：没有 encoder/decoder 的分离，一套权重同时用于“读”和“写”，KV cache 的复用逻辑也更直接。
- 架构更简单：单一栈的scaling law更容易观测和优化。

到 2022 年前后，decoder-only 几乎“统治”了通用大语言模型，encoder-decoder 则退守到翻译、语音识别等输入边界依然清晰的场景。

### KVCache成为推理效率瓶颈，decoder-only 内部改进（2023–2024）

推理成本成为主要矛盾
- 围绕KVCache做改进：GQA、MLA、滑动窗口、分页注意力
- 围绕FFN的改进：MoE 路由……

该阶段论调：结构已经收敛，围绕推理效率做改进。

### Agentic长上下文workload，输入远大于输出，重新引入encoder（2025-2026）

在 decoder-only 中，prefill（处理输入）和 decode（生成输出）共享同一套全层计算，即使输入只需要被“读一遍”，也要承担与生成阶段相近的单位 token 计算成本。

DeepSeek V4.1 的 Causal Encoder-Decoder（CED）：把 40 层 Transformer 切成两半：20 层 causal encoder 专门处理提示，20 层 decoder 专门生成回答。关键的非对称设计在于：
- 激活参数不对称：prefill 时每 token 仅激活 8B 参数，decode 时激活 16B。输入路径更轻，输出路径保留更多容量。
- KV 的来源被重构：decoder 的全局 KV 不是从每个 decoder 层自己的隐藏状态逐层计算，而是从 encoder 最终隐藏状态投影而来。Decoder 只需重建生成起始所需的短滑动窗口状态。这使得全局 KV cache 被压缩到约 890 字节/token，约为前代 V4-Flash 的 1/4

这本质上是一种部署感知的架构分工：它承认 decoder-only 在通用生成上的成功，但也承认 prefill 和 decode 的计算/内存特征根本不同，因而值得在结构上为它们分配不同的层、不同的激活规模和不同的 KV 生成路径。CED不是对 BERT 式双向 encoder 的回归——这里的 encoder 仍然是因果的，保留左到右的顺序结构以兼容生成。


| 时期 | 主导结构 | 被解决的瓶颈 | 被牺牲/搁置的东西 |
|---|---|---|---|
| 2017 | Encoder-Decoder | 序列转导的表达力与并行性 | — |
| 2018–2019 | 三派竞争 | 预训练目标与结构匹配 | 通用性尚未成为焦点 |
| 2020–2022 | Decoder-only 胜出 | 规模化与能力涌现 | 输入/输出分工的精细度 |
| 2023–2024 | Decoder-only 内部精修 | 推理时的 KV cache 成本 | 结构层面的再分工 |
| 2025–2026 | **CED 等混合/非对称回归** | **输入重度负载下的 prefill/decode 经济性不对称** | **单一栈的“结构简洁性”** |



