GLM 1M上下文，KVCache占比100G？无法放在显存上。

计算通信掩盖

MLA中，
RoPE位置编码好像不能fp8量化，它对精度很敏感，得是BF16


计算kvcache容量、申请block、每个block内部存了什么。
