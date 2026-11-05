## 参考材料

[The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
[Efficient Decode Context Parallelism with vLLM for Long Context Workloads](https://vllm-project.github.io/2026/08/07/decode-context-parallelism.html)
[Elastic Expert Parallelism in vLLM](https://vllm-project.github.io/2026/05/14/elastic-expert-parallelism.html)
[Context parallelism and MoE DP + TP + CP + EP](https://docs.google.com/presentation/d/1aeZw7dF7zTLzTSwavZYjU5XRENRp5Dk7rMXDkhoD2zw/edit?pli=1&slide=id.p#slide=id.p)

---

HBM的容量有限，放
- 权重
- KVCache
- 激活

多卡的计算结果要和单卡相同!

切分方式：

切输入：PCP
切权重
切模型层
切一批请求

各种并行策略可以组合: 分配rank的优先级DP > PP > PCP > TP

所有机器全局rank、单机器内local rank、通信域内的组（group）rank

快慢卡：一张卡很慢，拖累了其它卡。

## 通信算子

AllToAll通信算子用于MoE

还有Send和Recieve通信算子（PP）

AllReduce = ReduceScatter + AllGather

## TP

单卡放不下完整模型时，可以切分权重。

矩阵乘法的切分：AxB，mxk, kxn，可以把B竖着切（切n），也可以把A横着切（切m）
切分Linear。
- $W_Q$ $W_K$ $W_V$，多头注意力下，可以按头切分,按列竖着切。按照头来切，能省掉softmax的计算量？

0卡  1卡
W_{Q0}  W_{Q1}
W_{K0}  W_{K1}
W_{V0}  W_{V1}

AllGather后，和单卡结果等价，Q、K、V都有了

- $W_O$ 按行横着切

Scatter: 

$H_0$    $H_1$
W_{O0} W_{O1}

AllReduce

不过，上面的AllGather和Scatter实际上可以抵消。

$W_Q$ $W_K$ $W_V$竖着切，$W_O$横着切，是最省通信量和计算量的操作。
疑问：为什么$W_O$不能竖着切？


**对 \(W_Q, W_K, W_V\) 按列切分（Column Parallel），对输出投影 \(W_O\) 按行切分（Row Parallel）**。

这种“列切 → 行切”的配对，目的是让每个设备独立计算一部分注意力头，最后通过一次通信规约（All-Reduce）得到最终结果，与 MLP 层的切分骨架完全一致。

假设：
*   输入隐藏状态为 \(X \in \mathbb{R}^{b \times s \times d}\)（\(b\): batch，\(s\): 序列长度，\(d\): 隐藏维度）。
*   注意力头数为 \(h\)，每个头的维度为 \(d_k\)（通常 \(d = h \cdot d_k\)）。
*   张量并行度（设备数）为 \(P\)，且通常 \(h\) 能被 \(P\) 整除。

#### 1. \(W_Q, W_K, W_V\) 的列切分（竖切）

\(W_Q, W_K, W_V\) 的形状通常为 \([d, d]\)（或 \([d, h \cdot d_k]\)）。按列切分，就是将输出维度（即头的维度）划分给不同的设备。

将 \(W_Q\) 按列分成 \(P\) 块：
\[
W_Q = [W_Q^{(1)}, W_Q^{(2)}, \dots, W_Q^{(P)}]
\]
其中每一块 \(W_Q^{(p)} \in \mathbb{R}^{d \times (d/P)}\)，对应分配给第 \(p\) 个设备的 \(h/P\) 个注意力头。\(W_K, W_V\) 同理。

在第 \(p\) 个设备上，独立计算属于自己的那部分 Q、K、V：
\[
Q^{(p)} = X W_Q^{(p)}, \quad K^{(p)} = X W_K^{(p)}, \quad V^{(p)} = X W_V^{(p)}
\]
由于不同注意力头的计算是完全独立的，每个设备可以**在本地、零通信**地完成自己那几个头的注意力计算：
\[
\text{Head}^{(p)} = \text{Softmax}\left( \frac{Q^{(p)} (K^{(p)})^\top}{\sqrt{d_k}} \right) V^{(p)}
\]

#### 2. \(W_O\) 的行切分（横切）

\(W_O\) 的形状通常为 \([d, d]\)（或 \([h \cdot d_k, d]\)）。它的作用是将所有头的输出拼接后，再映射回隐藏维度 \(d\)。按行切分，就是将输入维度（头的维度）划分开。

将 \(W_O\) 按行分成 \(P\) 块：
\[
W_O = \begin{bmatrix} W_O^{(1)} \\ W_O^{(2)} \\ \vdots \\ W_O^{(P)} \end{bmatrix}
\]
其中每一块 \(W_O^{(p)} \in \mathbb{R}^{(d/P) \times d}\)，其行数恰好对应第 \(p\) 个设备所持有的那些头的输出维度。

第 \(p\) 个设备用自己的局部注意力输出 \(\text{Head}^{(p)}\)（可视为一个 \(b \times s \times (d/P)\) 的张量）乘以自己那份 \(W_O^{(p)}\)，得到一个**部分和**：
\[
Z^{(p)} = \text{Head}^{(p)} W_O^{(p)}
\]
这个 \(Z^{(p)}\) 的维度是 \(b \times s \times d\)，但它只是最终结果的一部分。

#### 3. 最终合并

最终的注意力输出 \(Z\) 是所有设备上部分和的**和**：
\[
Z = \sum_{p=1}^{P} Z^{(p)}
\]
这个求和操作，正是通过一次 **All-Reduce** 通信完成的。

用公式完整写出来就是（参考 Megatron-LM 的表述）：
\[
Z = \sum_{p=1}^{P} \text{Attention}\left(X W_Q^{(p)}, X W_K^{(p)}, X W_V^{(p)}\right) W_O^{(p)}
\]
其中 \(\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\)。

### 总结

| 权重矩阵 | 切分方式 | 切分的维度 | 对应的并行策略 | 通信操作 |
| :--- | :--- | :--- | :--- | :--- |
| **\(W_Q, W_K, W_V\)** | **竖切（列切）** | **输出维度**（头维度） | Column Parallel | 无（各头独立计算） |
| **\(W_O\)** | **横切（行切）** | **输入维度**（头维度） | Row Parallel | 一次 All-Reduce |

因此，你的理解“先竖着切再横着切”是准确的：QKV 三个投影矩阵先进行列切分，让每个设备负责一部分头；最后的输出投影 \(W_O\) 再进行行切分，将各设备的部分结果汇总起来。

只有一个头的MLA，切TP性价比低。

## SP（序列并行）

Embedding, Attention, lmhead都会切TP

### Attention的SP（和TP结合）

TP有重复计算的问题

Embedding操作后有AllReduce？
AllReduce -> ReduceScatter + AllScatter

切分SP后，配合TP能减少Norm计算、减少激活。训练场景中对SP更加依赖。

### MoE的SP（和TP结合）

省专家选择、topk、linear等操作

0卡: H_0  H_4
1卡: H_0  H_4

## PP（流水线并行）

Send / Recieve
## DP（数据并行）

一个DP对应一个EngineCore/Scheduler，TP共用EngineCore。

用户变动了，用多个模型副本分担请求。
DP：输入不一样

DP也有一个AllReduce操作，用于收集元数据。

DP要跑一个假的请求？陪跑

- 两个不同的通信域，同一时刻发起通信有问题
- 图模式下，DP的padding也要统一？
    - dp0  3->4
    - dp1  5->8
    - 实际上需要统一为8

ModelRunner下多DP 通信同步，是bug高发场景？
注意和DP的叠加

DP间负载均衡
## External DP
DP和直接开两个vllm服务的区别：在MoE。

## EP

开了EP后，专家好像不会被复制（不考虑EPLB）

dispatch, gmm, combine

AllToAll prefill
MC2 <512 token

## 大EP

“大EP”是将数百个专家分布到更多的卡上，形成跨节点的集群。
传统的 EP 通常局限在单台服务器（如 8卡）内部，
而“大EP”意味着专家被分散部署到了数十甚至数百张加速卡上，这些卡往往跨越了多个物理节点。

Decode是Memory Bound。每张卡上的专家数量越少，读取的内存越少，就可以提升性能。

大EP（每张卡上专家数量很少）的问题：放大了专家负载不均衡 -> EPLB

跨机器TP的通信操作性能很差，所以大EP下不开TP。

## PD分离

- prefill: PP + TP + EP
- decode: DP + EP

纯DP没有EP下也去切，权重矩阵。
