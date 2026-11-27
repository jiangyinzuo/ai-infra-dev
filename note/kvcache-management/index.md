## KVCacheSpec和KVCacheGroupSpec
模型不同的结构会采用不同的KVCache管理方式。比如Gamma3，Full Attention就用`FullAttentionSpec`，滑动窗口Attention就用`SlidingWindowSpec`。
可以见`Attention.get_kv_cache_spec`函数。

```
  󰌗 KVCacheSpec
    󰌗 _TrulyUnregisteredSpec
    󰌗 HiSparseHotSpec
    󰌗 HiSparseResidentSpec
   │  󰌗 AscendStoreHiSparseResidentSpec
    󰌗 AttentionSpec
   │  󰌗 FullAttentionSpec
   │  󰌗 ChunkedLocalAttentionSpec
   │  󰌗 SlidingWindowSpec
   │  󰌗 CircularBufferSpec
   │  󰌗 EncoderOnlyAttentionSpec
   │  󰌗 CrossAttentionSpec
    󰌗 MambaSpec
    󰌗 UniformTypeKVCacheSpecs
```

`get_kv_cache_groups()`函数将模型的多个层的`KVCacheSpec`分组，产生每个组的`KVCacheGroupSpec`

```python
@dataclass
class KVCacheGroupSpec:
    """Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group holds draft layers and takes the EAGLE last-block drop.
    is_eagle_group: bool = False
    # Host groups use the dedicated HiSparse pool; others share the device pool.
    host_resident: bool = False
    # Whether this group is part of the externally transferable KV state.
    # Ephemeral accelerator-side replicas are rebuilt from their transferable
    # source group after a connector load.
    enable_kv_transfer: bool = True
    role: KVCacheGroupRole = KVCacheGroupRole.DEFAULT
```

## SingleTypeKVCacheManager

SingleType: 指一组 KV Cache 具有同一种管理语义，例如全注意力、滑动窗口、HiSparse Host Source

```vllmclass
class SingleTypeKVCacheManager(ABC):
    def __init__(self, kv_cache_spec: KVCacheSpec, block_pool: BlockPool, enable_caching: bool, kv_cache_group_id: int, scheduler_block_size: int, dcp_world_size: int=1, pcp_world_size: int=1, needs_kv_cache_zeroing: bool=False, max_admission_blocks_per_request: int | None=None) -> None:
        self.block_pool = block_pool
    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]: ...
    # 将已完成 block 注册到 prefix cache
    def cache_blocks(self, request: Request, num_tokens: int, retention_interval: int | None=None, *, replay_boundaries: Sequence[int]) -> None: ...
    def free(self, request_id: str) -> None: ...
```

```
  󰌗 SingleTypeKVCacheManager
    󰌗 FullAttentionManager
   │  󰌗 RSWAManager
   │  󰌗 CircularBufferManager
   │  󰌗 SinkFullAttentionManager
   │  󰌗 HiSparseSourceManager
   │  󰌗 KpoolTailManager
    󰌗 SlidingWindowManager
    󰌗 ChunkedLocalAttentionManager
    󰌗 MambaManager
    󰌗 CrossAttentionManager
    󰌗 _HiSparseAuxiliaryManager
      󰌗 HiSparseHotManager
      󰌗 HiSparseResidentManager
        󰌗 AscendStoreHiSparseResidentManager
```

## HiSparse特性的KVCache管理需求

HiSparse 需要把同一批 MLA 层的主 KV组织为 Host Source、Resident 和 Hot 三类group。

三者有不同的block table和生命周期：Source 记录 Host 页，Resident 记录当前仍在设备上的页，Hot 为请求提供固定容量的换入槽。

`resolve_hisparse_specs` -> `_fill_hisparse_groups` -> `HiSparseResidentSpec`

### KVCacheGroupRole

[HiSparse特性引入了这个类](https://github.com/vllm-project/vllm/commit/d43bb2f37f87a63d3a9a299c97050443cf6a2520)
