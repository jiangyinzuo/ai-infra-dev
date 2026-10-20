vllm/vllm:docs/design/hisparse.md

# HiSparse local KV offload architecture

---
generated-by: DeepSeek-V4.1-flash

vLLM 中的 **HiSparse** 是一个**实验性**的、针对稀疏注意力模型（如 GLM 5.3、DeepSeek 系列等使用 DSA/MLA 架构的模型）的**分层 KV 缓存管理架构**。其核心思想是：既然稀疏注意力在解码时只读取 Top-K 个 token 的 KV，那么就没必要把完整的 KV 历史都留在宝贵的 GPU 显存（HBM）里，可以将完整历史放在 CPU 主机内存，GPU 只保留一个小型的“热”缓存，按需取用。

### 🎯 核心机制：按需取用，而非全量驻留

HiSparse 利用稀疏注意力的特性来突破 GPU 显存对并发请求数的限制。在传统的推理中，每个请求的完整 KV 缓存都必须驻留在 GPU 上，上下文越长，占用的显存就越多，能同时处理的请求就越少。HiSparse 改变了这一点：

*   **完整历史存主机**：每个请求的完整 KV 历史都保存在 CPU 的 pinned memory（锁页内存）中。
*   **GPU 只留“热”数据**：GPU 上仅为每个请求保留一个固定大小的“热缓冲区”（hot buffer）或工作集。在解码的每一步，系统根据注意力机制选出的 Top-K 索引，只将当前需要的那部分 KV 数据（可能只是历史中的几行）从主机内存调入 GPU 的热缓存中参与计算。
*   **效果**：这样，单个请求在解码时的 GPU 显存占用就不再随上下文长度线性增长，而是被限制在一个较小的固定值附近，从而可以大幅提升长上下文场景下的并发处理能力。根据 vLLM 的测试，在 8×H200 上运行 GLM 5.3 时，该方案能让显存容量支持的并发请求数提升约 **9.75 倍**，并使得 100 万 token 的上下文长度成为可能。

### 🧩 主要组件与工作方式

HiSparse 不是一个单一的模块，而是集成在 vLLM 调度器和 Worker 中的一套协作机制：

*   **`HiSparseCoordinator`**：运行在调度器侧，负责管理主机端的逻辑块、跟踪前缀身份，并决定 GPU 块的驻留与迁移。
*   **`HiSparseConnector`**：作为 KV 连接器，负责在调度器和 Worker 之间传递数据迁移的指令和状态。
*   **`HiSparseWorker` & `HiSparseRuntime`**：运行在 Worker 侧，负责执行实际的数据传输，并管理每个注意力缓存对应的主机/GPU 视图。

在具体实现上，它还会利用 **LRU（最近最少使用）** 策略来管理 GPU 上的热缓存，以提高 Top-K 数据的命中率；并通过融合 CUDA 内核来处理命中、未命中以及从主机取数的操作，力求不让数据搬运阻塞解码的关键路径。

### ⚠️ 注意事项

*   **实验性质**：在 vLLM 的官方文档中，该功能被明确标记为“实验性”（experimental）。
*   **模型限制**：它主要适用于使用稀疏注意力机制的模型，例如采用 DSA（DeepSeek Sparse Attention）或 MLA（Multi-head Latent Attention）的模型（如 GLM 5.3）。对于使用标准稠密注意力的模型，由于每一步都需要访问完整的 KV 历史，这套机制并不适用。
*   **配置与依赖**：使用它需要配置特定的连接器（`HiSparseConnector`）或注意力配置（`HiSparseConfig`），并且可能依赖特定的后端（如 Model Runner V2）。

---
