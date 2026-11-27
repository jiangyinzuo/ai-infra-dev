# vLLM显存管理

在 vLLM 的 Python 层面：
- torch.Tensor: 持有GPU显存起始地址
    - Worker 分配一整块原始 Tensor: allocate_kv_cache -> _allocate_kv_cache_buffer -> torch.zeros
- KVCacheBlock: 只管理元数据（如block_id），不直接持有地址
    - 由BlockPool管理

## GPU 核函数的输入会传入block table或者slot mapping

读取路径（Attention），需要block ID转显存地址：
- `physical_block_id = block_table[logical_block_idx]`
- `target_block_addr = base_ptr + physical_block_id * row_stride`

写入路径（KV Cache 更新）：需要知道“刚刚计算出的这个新 token 的 K/V，应该精确地写到显存的哪一个槽位”
- slot_mapping: token position -> 物理 slot

## KVCacheBlock和BlockPool定义
```vllmclass
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # Deferred frees may contain blocks from multiple pools. Weak to avoid ref cycles.
    pool: "weakref.ref[BlockPool] | None" = field(
        default=None, repr=False, compare=False, kw_only=True
    )
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    block_hash: BlockHashWithGroupId | None = None
    # Number of prefix tokens covered by block_hash. For full blocks this is
    # the full block boundary; partial entries can end inside a cache block.
    block_hash_num_tokens: int | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False
```

```vllmclass
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    def __init__(self, num_gpu_blocks: int, enable_caching: bool, hash_block_size: int, enable_kv_cache_events: bool=False, metrics_collector: KVCacheMetricsCollector | None=None, num_hash_lookahead_tokens: int=0, medium: str=MEDIUM_GPU):
        self.num_gpu_blocks = num_gpu_blocks
        self.medium = medium
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.num_hash_lookahead_tokens = num_hash_lookahead_tokens
        self._weak_self = weakref.ref(...)
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(...) for idx in range(...)]
```

## Kernel函数定义
```python
@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mapping_enabled,  # [num_kv_cache_groups]
    dcp_sharded,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
```

```python
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            block_table, # req_to_token=block_table
            seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
            sliding_window=self.decode_sliding_window,
        )
```
---

普通的full attention，每一层的block pool都一样。
发起一次请求，就可以每一层都返回 。
逻辑上每一层的kvcache是独立的。

通过layer stride定位

Qwen 3.x: 分成多个group，复用kvcache block

一个 KV Cache Group 是引擎内部的一个逻辑管理单元。引擎会为每个组维护独立的内存池、块表（block table）和元数据（metadata）。

同组层共享管理策略：同一个组内的层，其 KV cache 的物理页大小、量化方式、回收策略是统一的。

不同组按需定制：例如，所有 Mamba 层可能构成几个组，它们的“缓存”实际上是状态缓存，管理方式有别于 FullAttention 组。全注意力层则根据是否属于草稿器、是否共享 KV 等，分入不同的组。

权衡“填充浪费”与“管理开销”：分组大小的选择是一个权衡（trade-off）。组设得太大，内部形状差异大的层会被迫按最大规格分配内存，造成填充浪费（padding）；组设得太小（甚至一层一组），组数量激增，每步推理遍历组、重建元数据的CPU 开销会变得不可忽视。vLLM 的相关 PR 正是在动态调整这个分组策略，以平衡内存浪费和调度开销。

所以，你看到的“分 group”，是推理引擎为了在一个模型内部同时优雅地容纳全注意力层、线性注意力层、推测解码草稿层等异构缓存需求而构建的管理框架。Qwen3.x 正是这类异构性突出的模型，因此“分组”在它的部署中显得尤为重要。
