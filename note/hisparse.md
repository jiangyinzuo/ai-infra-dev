GPU上可以写hashtable，也不大需要融合。

但NPU没有SIMT，可能不好实现hashtable，可能需要把indexer和后面的算子融合？

indexer算子 hisparse_cache_handle.swap_in算子 sfa算子（sparse flash attention算子）

可以在上游社区添加接口，方便后续优化？

社区（vLLM和SGLang似乎都有把Scheduler重构为Rust的趋势？）
Scheduler --> Worker --> Attention Backend


hisparse整体流程/接口：hisparse.swap_in

---

动机：稀疏注意力减少GPU / NPU计算量，
