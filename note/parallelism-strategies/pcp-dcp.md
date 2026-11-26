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

## DCP Decode流程

### AllGather Q
设一个 GQA 层有 **32 个 Q heads、4 个 KV heads**，每 8 个 Q heads 共用 1 个 KV head。取一个请求的一步 decode，本步有 1 个 query token。

`TP=8、DCP=2、PCP=1` 时，8 个 TP ranks 两两组成 4 个 DCP 组。例子里每组将对应的 KV head 按 token 位置交错存储，一张卡存偶数位置，一张卡存奇数位置。

| DCP 组（TP ranks） | TP rank | 本地 KV cache | `all_gather` 前的 Q heads | `all_gather` 后的 Q heads |
| --- | ---: | --- | --- | --- |
| {0, 1} | 0 | KV head 0，偶数位置 | Q0–Q3 | **Q0–Q7** |
| {0, 1} | 1 | KV head 0，奇数位置 | Q4–Q7 | **Q0–Q7** |
| {2, 3} | 2 | KV head 1，偶数位置 | Q8–Q11 | **Q8–Q15** |
| {2, 3} | 3 | KV head 1，奇数位置 | Q12–Q15 | **Q8–Q15** |
| {4, 5} | 4 | KV head 2，偶数位置 | Q16–Q19 | **Q16–Q23** |
| {4, 5} | 5 | KV head 2，奇数位置 | Q20–Q23 | **Q16–Q23** |
| {6, 7} | 6 | KV head 3，偶数位置 | Q24–Q27 | **Q24–Q31** |
| {6, 7} | 7 | KV head 3，奇数位置 | Q28–Q31 | **Q24–Q31** |


#### AllGather LSE + ReduceScatter 求和（ag_rs路径）
