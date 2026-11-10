---
title: KVCache 布局
---

llm请求具有动态性、变长性，无法预知一个请求最终会生成多少token。
如果要求静态、连续分配显存，就只能给所有请求按最大上下文长度来预留空间，
造成极其严重的显存浪费（内部碎片）。因此，需要借鉴操作系统的虚拟内存分页机制：逻辑连续，物理离散，通过块表映射。

解决三个问题
- 内部碎片
- 外部碎片
- 前缀共享，复用KVCache

在 vLLM 的 PagedAttention 机制中，“页内布局”和“页间布局”是两个不同层级的抽象。

- 页间布局解决：“逻辑块如何映射到物理块”的寻址问题；
- 页内布局解决：“一个物理块内部，KV 数据按什么张量轴序排列”的访存效率问题。

## 页间布局

### 逻辑块和物理块

逻辑块与物理块：每个请求的 KV Cache 在逻辑上被划分为固定大小的
“块”（Block/Page），通常包含 16、32 或 64 个 Token 的数据。这些逻辑块在物理 GPU
显存中无需连续存放，可以分散在不同的物理块中。

块表（Block
Table）：这是实现映射的关键数据结构。每个请求都维护一张私有的块表，记录了其逻辑块索引到物理块指针的对应关系。Attention
内核在计算时，会通过查询块表来“间接寻址”（indirection），找到每个逻辑块实际对应的物理内存地址。

### BLOCK_SIZE `--block-size`

vLLM实例启动初始化完成后，每个 KV cache group 的 block_size 就固定了，正常推理期间不会动态改变。

block size的值可以由用户指定`--block-size`，或者取默认值16。其含义为，一个Block包含多少个token的数据。

vllm/vllm:vllm/config/cache.py:181-185
```python
@config
class CacheConfig:
    """Configuration for the KV cache."""

    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16
```

## 页内布局

当一个物理块被找到后，其内部字节如何组织，就是页内布局要解决的问题。vLLM 用 `KVCacheLayout` 这个枚举来定义不同的物理轴顺序。

vllm/vllm:vllm/v1/kv_cache_layout.py:15-28
```python
class KVCacheLayout(Enum):
    """Physical layout descriptor for a KV cache group.

    The logical shape is always [L, B, H, N, <content>] (RFC #42082).
    Each member's value is a stride permutation that maps logical axes
    to physical (memory) order.
    """

    LBHNC = (0, 1, 2, 3, 4)  # [L, B, H, N, C] (identity)
    LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]
    LHBNC = (0, 2, 1, 3, 4)  # [L, H, B, N, C]
    BLHNC = (1, 0, 2, 3, 4)  # [B, L, H, N, C]
    BLNHC = (1, 0, 3, 2, 4)  # [B, L, N, H, C]
    BHLNC = (1, 2, 0, 3, 4)  # [B, H, L, N, C]
```

vLLM中，KV Cache 的逻辑形状始终是 `[L, B, H, N, C]`，分别代表
- Layer
- Block
- Head，GQA下是KV head
- Token数量 / State数量
- Content

逻辑坐标`cache[l, b, h, n, c]`始终为: 某层、某个缓存块、某个 head、块内某个 state 的某个内容位置，不应该随物理布局变化而变化。

真正的内存寻址由strides决定：
$offset = base + l * s_L + b * s_B + h * s_H + n * s_N + c * s_C$

页内布局改变的只是这些维度在物理内存中的**排列顺序（stride permutation）**。

### 给定请求、token position，定位物理块号和偏移

对普通未压缩、无上下文并行分片的缓存：
```
# 输入：请求 r、历史 token 位置 p

logical_block_idx = p // block_size

# block table 真正完成的查找：
物理block编号 B = block_table[r][logical_block_idx]

# 页内偏移不需要查表：
N = p % block_size

# 最终得到该 token 的缓存位置：
location = (B, N)
```

```
slot_id = B * block_size + N
```

### GQA的布局

`[B, H_kv, N, 2D]`
- N即`--block-size`参数
- H_kv: 当前 TP rank 的 KV head 数
    - H_kv 的 TP 划分在 vllm/model_executor/models/llama.py#L143：KV heads 足够时按 TP 划分，不足时在部分 ranks 间复制。因此，不能直接把模型配置中的总 KV head 数当作每个 GPU 上的 H_kv。
- 2D: KCache和VCache，head dimension

### DeepSeek V3 MLA的布局

缓存latent KV + RoPE key

`MLAAttention.get_kv_cache_spec()` (vllm/model_executor/layers/attention/mla_attention.py#L1373)

H维度是1
```
[L, B, H=1, N, C=576]
               ├── 前 512 个元素：latent KV（normed）
               └── 后  64 个元素：RoPE key
```

#### 量化格式


| 格式 | 内核侧内容 | 每 token 有效载荷 |
|---|---|---:|
| BF16 | 512 维 latent + 64 维 RoPE，全部 BF16 | 1152 B |
| 普通 FP8 路径（不常用，RoPE最好别直接量化为FP8） | 576 个 FP8 值，缩放因子另行传入 | 576 B |
| `fp8_ds_mla` | 512 个 FP8 E4M3 latent 值 +  4 个 FP32 scale，每个对应 128 个 latent 值 + 64 个BF16 RoPE | 656 B |
| `nvfp4_ds_mla`，SM100 可选 | packed FP4 latent + FP8 RoPE + scale | 352 B |


还原公式：$\boxed{x_{\text{restored}} = x_{\text{fp8}} \times \text{scale}}$

### DeepSeek V3.2: MLA Cache + Index Cache

Index Cache有点像没有Value的MQA：所有Query共享一个Key。所以H=1

```
[L, B, H=1, N=block_size, C=132 bytes]，dtype=uint8
```

| 内容 | 数量 | 存储格式 | 占用 |
|---|---:|---|---:|
| Indexer key | 128 个元素 | FP8 E4M3，每元素 1 B | 128 B |
| 量化 scale | 1 个 | FP32 | 4 B |
| **合计** | | | **132 B** |


> TopK算子实现的时候，一般是直接返回token position，而不是返回(B, N)
> 后续需要经过token position转(B, N)

### 一些KVCacheLayout属性

vllm/vllm:vllm/v1/kv_cache_layout.py:39-58
```python
    @property
    def is_layer_compact(self) -> bool:
        """True when the layer is compact; i.e. the L dimension is outermost."""
        return self.value[_DIM_L] == 0

    @property
    def is_block_contiguous(self) -> bool:
        """True when [H, N, C] is contiguous within a block."""
        return self.value[-3:] == (_DIM_H, _DIM_N, _DIM_C)

    @property
    def is_block_compact(self) -> bool:
        """True when each page's [H, N, C] bytes form one contiguous run; i.e.
        the L and B dimensions are outermost."""
        return set(self.value[:2]) == {_DIM_L, _DIM_B}

    @property
    def is_block_outermost(self) -> bool:
        """True when B is the outermost physical dimension."""
        return self.value[0] == _DIM_B
```

## 页内布局和页间布局的协同

这两个概念在常规dense模型中相对独立，但在 **DeepSeek Sparse MLA** 这类场景下，页间与页内布局的**协同**变得至关重要。

稀疏注意力内核（如 FlashMLA Sparse）的 Top-K 遍历逻辑，期望将绑定的缓存视为**按层（Layer）组织的 `[B, N, C]` 行**。因此，它要求页内布局必须是 **“Layer-Compact”** 的（例如 **`LBNHC`** 或 **`LBHNC`**），以保证每一层的“行视图”在物理内存中是成立的。

如果错误地使用了 **`BLHNC`**（块最外层）布局，所有层的 KV 数据会被打包进一个大的、以块为最外维度的内存区域中，导致**逐层的行视图在物理上无法构建**，稀疏内核的索引逻辑就会失效并崩溃。因此，vLLM 在稀疏 MLA 后端中会**明确声明支持 Layer-Compact 布局，并排除 `BLHNC`**，以确保页内布局与稀疏访存模式兼容。
