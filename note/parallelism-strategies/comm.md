

## User

大模型推理中，TP和PCP / DCP 会用哪些通信原语？ all gather?

## Assistant

**会用 All-Gather，但不能概括成“TP、PCP、DCP 都是 All-Gather”。** 核心区别是：**交换什么张量，以及交换后要拼接还是归约。**

这里 PCP 指 **Prefill Context Parallelism**，DCP 指 **Decode Context Parallelism**；具体通信组合取决于框架和 Attention 后端。

## 1. 先看典型通信

| 并行方式 | 切分维度 | 典型通信原语 | 通信目的 |
|---|---|---|---|
| **TP** | 权重、Attention heads、hidden/intermediate 维度 | **All-Reduce**；部分位置用 All-Gather、Reduce-Scatter | 合并矩阵乘法的部分结果，或转换激活布局 |
| **PCP** | Prefill 的 token/序列维度 | **All-Gather KV**；Ring Attention 则用 **Send/Recv** | 让本地 Q 能访问其他卡上的 K/V |
| **DCP** | 历史 KV Cache 的 token/序列维度 | **All-Gather Q + All-Gather LSE + Reduce-Scatter 输出**；也可用 **All-to-All** 交换局部结果 | 各卡计算局部 Attention，再合并为完整上下文的结果 |

**这是典型路径，不是每个实现必须固定调用的列表。** TP 与 CP 叠加后，两类通信也会同时出现。

## 2. TP：重点通常是 All-Reduce，不是 All-Gather

以经典两段式线性层为例：

- **Column-parallel**：每卡计算不同输出通道，结果可以继续保持分片，不必立即通信。
- **Row-parallel**：每卡计算同一个输出的部分和，需要相加：

$$
Y=\sum_r X_rW_r
$$

如果后续每卡都需要完整的 $Y$，通常用 **All-Reduce**。

典型 Dense Transformer 中，Attention 的 `o_proj` 和 MLP 的 `down_proj` 都可能在这里做 All-Reduce。因此经典 TP 常见的是**每层两次 All-Reduce**，而不是每层把所有分片 All-Gather。

补充两点：

- 某个算子确实需要完整拼接结果时，才用 **All-Gather**。
- 结合 Sequence Parallel 等布局时，可以改用 **Reduce-Scatter / All-Gather**。虽然求和 All-Reduce 在语义上可以分解成两者，但不代表代码一定显式调用两次。

## 3. PCP：常见的是 All-Gather KV

假设两张卡分别持有：

- GPU 0：$Q_0,K_0,V_0$
- GPU 1：$Q_1,K_1,V_1$

每张卡只负责自己那部分 query，但这些 query 可能需要关注其他卡上的历史 token。一种直接实现是：

**All-Gather K/V → 每卡用本地 Q 和完整 K/V 算 Attention。**

此时，各卡负责的是**不同 query 的完整输出**，所以 Attention 结果本身不需要跨卡求和。

另一种实现是 **Ring Attention**：K/V 分块通过 **Send/Recv** 环形传递，各卡边接收边计算、更新在线 Softmax，避免同时放下完整 K/V。vLLM 的设计文档明确区分了这两条路线。[来源](https://docs.vllm.ai/en/v0.31.0/serving/context_parallel_deployment/)

## 4. DCP：不仅要 All-Gather，还要正确合并 Softmax

DCP 下，历史 KV 分布在不同卡上。各卡需要针对**同一个 query/head**，计算不同上下文分片的贡献。

典型过程：

1. **All-Gather Q**：如果 Q 原本按 TP heads 分片，先让各 DCP rank 拿到所需的 Q。
2. 用本地 KV 计算局部输出 $o_r$ 和局部 **LSE**。
3. 交换并合并局部结果。

**不能直接把局部 Attention 输出相加或平均**，因为各卡的 Softmax 分母不同。正确合并为：

$$
L=\operatorname{logsumexp}_r(L_r),\qquad
o=\sum_r e^{L_r-L}o_r
$$

因此有两种常见实现：

- **All-Gather LSE → 本地算归一化权重 → Reduce-Scatter 加权输出**，恢复 TP 分片布局。
- **All-to-All 交换局部 output 与 LSE → 接收端本地合并**。

vLLM-Ascend 文档描述了这两条路径；SGLang 也提供相应通信后端。另外，若复制 Q 投影权重、在本地计算所需 Q，可以省掉逐层的 Q All-Gather，代价是更多权重显存和计算。[vLLM-Ascend](https://docs.vllm.ai/projects/ascend/en/v0.23.0/developer_guide/Design_Documents/context_parallel.html) · [SGLang DCP](https://docs.sglang.io/docs/advanced_features/dcp)

**记忆重点：TP 合并矩阵乘法部分和；PCP 交换跨 token 的 K/V；DCP 合并跨 KV 分片的 Attention 结果，而且必须带上 Softmax 归一化信息。**

