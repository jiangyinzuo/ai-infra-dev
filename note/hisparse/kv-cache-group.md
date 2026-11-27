# KV Cache Group

vllm/vllm-ascend:/kv_cache_interface.py:1543-1546
```python
class KVCacheGroupRole(StrEnum):
    DEFAULT = "default"
    HISPARSE_SOURCE = "hisparse_source"
    HISPARSE_INDEXER = "hisparse_indexer"
```


- 一组共享同一套 KV block table 的模型层。
- 共享 block 映射，不共享 KV 数值。
- Full Attention、Sliding Window、GDN 等缓存规格不同 → 分组管理。
- `KVCacheGroupSpec.layer_names`：组内层。
- `KVCacheGroupSpec.kv_cache_spec`：组规格。

## Qwen3-Next 示例

- GDN 层：使用 `MambaManager`，保存递归状态。
- Full Attention 层：使用 `FullAttentionManager`，保存历史 K/V。
- 每组有独立 block table；底层物理 block pool 可以共享。

关键源码：

```text
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_planning.py
vllm/v1/core/kv_cache_coordinator.py
```
