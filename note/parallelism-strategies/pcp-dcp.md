# PCP / DCP

## PCP

- Prefill Context Parallel：按 token／Query 分片。
- 各 rank 处理本地 Query，并收集所需 KV／hidden states。
- 重点：分块、KV 映射、通信和输出恢复。

## DCP

- Decode Context Parallel：按历史上下文分片。
- 各 rank 读取本地 KV 分片。
- 通过 LSE 归并和 ReduceScatter 得到最终 attention 输出。

与 TP 区分：PCP/DCP 切分上下文或 Query；TP 切分模型张量。
