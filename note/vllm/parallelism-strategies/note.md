## 参考材料

[The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
[Efficient Decode Context Parallelism with vLLM for Long Context Workloads](https://vllm-project.github.io/2026/08/07/decode-context-parallelism.html)
[Elastic Expert Parallelism in vLLM](https://vllm-project.github.io/2026/05/14/elastic-expert-parallelism.html)
[Context parallelism and MoE DP + TP + CP + EP](https://docs.google.com/presentation/d/1aeZw7dF7zTLzTSwavZYjU5XRENRp5Dk7rMXDkhoD2zw/edit?pli=1&slide=id.p#slide=id.p)
[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)

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
