# PCP / DCP

参考资料：
- https://pisceskkk.github.io/CP-Viz/va-onboarding/#slide=3
- https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html
- vllm并行策略之DCP(Decode Context Parallel) - 梦初的文章 - 知乎 https://zhuanlan.zhihu.com/p/2020086868914499979

PCP(Prefill Context Parallel): 切输入序列（切算力）, 降低TTFT
- 由于masked attention的计算是下三角，所以是第一个token和最后一个token放同一个rank。（首尾拼接）

DCP(Decode Context Parallel): 切KVCache（切显存）, 降低TPOT
- 交错（Interleave）的形式切分KVCache
- Slot Mapping:  token 的逻辑位置转换为KV Cache物理位置的映射，属于KVCache管理模块
- Virtual Block
    - Block interleave size太小的时候，block内存利用率不足，本来16个token可以放在同一个物理block的，但却被放到了4个物理block。

## PD分离场景，P节点开DCP，D节点DCP=1

## 为什么池化场景不建议开PCP？
