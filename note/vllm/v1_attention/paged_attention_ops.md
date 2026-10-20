# Paged Attention算子

**通常不单独做一个“映射算子”，而是把“查 block table → 计算 KV 地址 → 加载 KV”融合到 Attention kernel 内。** 不需要先把分页 KV 拷贝、拼接成连续张量。

可以分成**管理侧建表、计算侧查表**两部分。

## 1. 管理侧：维护 block table

假设每个 block 存放 `B=16` 个 token 的 KV：

```text
请求 A 的 block table = [7, 2, 19]
```

含义是：

- token 0～15 → GPU cache block 7；
- token 16～31 → GPU cache block 2；
- token 32～47 → GPU cache block 19。

管理侧负责分配、回收实际 block，并更新这张表。**kernel 只消费映射，不负责决定分配哪个 block。**

## 2. 计算侧：逻辑 token 位置转换成 cache 地址

假设 KV cache 的示意布局为：

```text
K_cache[num_blocks, block_size, num_kv_heads, head_dim]
```

对请求 `r` 的逻辑 token 位置 `t`，读取 KV head `h`、维度 `d`：

```python
logical_block = t // block_size
offset_in_block = t % block_size

physical_block = block_table[r, logical_block]

k = K_cache[physical_block, offset_in_block, h, d]
```

底层就是指针偏移计算。若各维 stride 以元素为单位：

```text
offset = physical_block  × stride_block
       + offset_in_block × stride_token
       + h               × stride_head
       + d               × stride_dim
```

这里的 `physical_block` 是 **GPU cache 池的 block 下标**，不是 GPU 物理内存地址；最终仍然用普通 GPU 指针加载数据。

## 3. vLLM 中的实际代码

当前仓库的 [`triton_decode_attention.py:128–146`](vllm/v1/attention/ops/triton_decode_attention.py#L128) 就实现了这个过程。核心代码摘录如下，其中 `Req_to_tokens` 在此处充当页表：

```python
# offs_n：这一轮要处理的逻辑 token 位置
kv_page_number = tl.load(
    Req_to_tokens
    + stride_req_to_tokens_b * cur_batch_req_idx
    + offs_n // PAGE_SIZE,
    mask=offs_n < split_kv_end,
    other=0,
).to(tl.int64)

kv_in_page = offs_n % PAGE_SIZE

offs_buf_k = (
    (kv_page_number * stride_buf_kpbs + kv_in_page * stride_buf_kbs)[
        :, None
    ]
    + cur_kv_head * stride_buf_kh
    + offs_d[None, :]
)

k = tl.load(
    K_Buffer + offs_buf_k,
    mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
    other=0.0,
)
```

这里一次处理一组 token 和 head 维度，而不是用 Python 循环逐个 token 查表。加载 K 后，代码直接计算 `qk`，再通过同样的页映射加载 V。

**查表是计算路径的一部分，没有先生成一份连续 KV 副本。**

## 4. 写入新 token 时怎么映射？

读取历史 KV 常用 `block_table`；写入当前 batch 的新 K/V，常用预先准备好的 `slot_mapping`。

概念上，对逻辑位置 `t`：

```python
physical_block = block_table[r, t // block_size]
slot = physical_block * block_size + t % block_size
```

把这个 `slot` 放入当前输入 token 对应的 `slot_mapping` 项。写 cache 的 kernel 再解码：

```python
physical_block = slot // block_size
offset_in_block = slot % block_size

K_cache[physical_block, offset_in_block, ...] = new_k
V_cache[physical_block, offset_in_block, ...] = new_v
```

实际 K/V 布局可能不同，因此最终地址必须使用对应后端的布局和 stride。

## 5. 实现时最重要的边界

- **最后一页可能没填满**：加载必须按有效序列长度做 mask，不能把未写入的 KV 算进去。
- **地址偏移可能溢出 `int32`**：上面的实际代码在页号乘 stride 前转成了 `int64`。
- **不要假设 cache 连续布局**：应使用真实 stride；当前代码明确考虑了跨层 view 带来的间隙。
- **生命周期必须同步**：kernel 使用期间，block 不能被回收或重新分配；若数据被 offload，需要先完成加载和必要同步。

**本质上，它是软件实现的间接寻址：`逻辑 token → 页表项 → cache block + 页内偏移`，并把这段寻址融合进 KV 读写 kernel。**

------------
User:

------------

有没有相对实现简单的算子代码可供参考

------------
Assistant:

------------

**有。若重点是理解“分页映射怎么落到 GPU 地址”，建议先看缓存写入 kernel，再看分页读取 kernel，不必一开始读 FlashInfer 的完整后端。**

## 1. 最容易入门：写入分页 KV cache

文件：[`triton_reshape_and_cache_flash.py`](vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L33)  
函数：`reshape_and_cache_kernel_flash`

先只看 **`USE_HEAD_MAJOR_LAYOUT=False`、`FP8_KV_CACHE=False`** 的普通路径，暂时忽略其他布局和量化分支。

核心流程如下，代码是从原实现简化的示意：

```python
# 1. 当前 program 处理哪个输入 token
token_idx = tl.program_id(0)

# 2. 查该 token 的目标 cache slot
slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
if slot < 0:
    return  # padding token，不写入

# 3. slot 拆成：实际 block 编号 + block 内偏移
block_idx = slot // block_size
block_offset = slot % block_size

# 4. 一组线程处理多个 head/dim 元素
tile_pos = tl.program_id(1) * TILE_SIZE + tl.arange(0, TILE_SIZE)
head = tile_pos // head_size
dim = tile_pos % head_size
mask = tile_pos < num_heads * head_size

# 5. 计算目标 cache 地址
dst = (
    block_idx * block_stride
    + block_offset * page_stride
    + head * head_stride
    + dim
)

# 6. 从输入 K 加载，写到分页 cache
k = tl.load(key_ptr + token_idx * key_stride + tile_pos, mask=mask)
tl.store(key_cache_ptr + dst, k, mask=mask)
```

V 的写入同理。

**这段最简单，但要注意：它消费的是已经算好的 `slot_mapping`，不包含“请求逻辑页 → 实际 block”的查表过程。** 适合先理解 cache 的地址计算与写入。

## 2. 看真正的页表查找：分页读取

文件：[`triton_decode_attention.py`](vllm/v1/attention/ops/triton_decode_attention.py#L128)  
函数：`_fwd_kernel_stage1`

不必先读整个 Attention，只看 **128～148 行**，就能看到完整的分页寻址：

```python
# 逻辑 token 位置，一次处理一组
offs_n = start_n + tl.arange(0, BLOCK_N)

# 请求的页表：逻辑页编号 → 实际 cache 页编号
kv_page_number = tl.load(
    Req_to_tokens
    + stride_req_to_tokens_b * cur_batch_req_idx
    + offs_n // PAGE_SIZE,
    mask=offs_n < split_kv_end,
    other=0,
).to(tl.int64)

# 页内偏移
kv_in_page = offs_n % PAGE_SIZE
```

紧接着的代码用下面的结构计算 K 地址并加载：

```text
实际页编号 × 页 stride
+ 页内偏移 × token stride
+ KV head 编号 × head stride
+ head 内维度偏移
```

这是你前面问的 **“Paged Attention 映射”最直接的真实实现**。再往下读，才是 `QK`、softmax 和 V 加权计算。

## 推荐阅读顺序

| 顺序 | 代码范围 | 理解目标 |
|---|---|---|
| 1 | 写入 kernel 的 [62～75 行](vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L62) | slot 怎么拆成 block 和页内偏移 |
| 2 | 写入 kernel 的 [108～153 行](vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L108) | 如何计算地址并写入 K/V |
| 3 | 读取 kernel 的 [128～148 行](vllm/v1/attention/ops/triton_decode_attention.py#L128) | 如何通过 block table 读取非连续的 KV |

另外，`ops/paged_attn.py` 虽然只有几十行，但主要是调用底层算子的**封装**，看不到真正的映射实现，不适合作为 kernel 入门代码。

------------
User:

------------

trition实现中，背后的页表管理是用了什么算法思想？看起来这时候的算子已经不再单纯是个pure function算子，更像是一个 有状态的小系统了？

------------
Assistant:

------------

**你的判断在“整个 KV cache 子系统”层面成立，但要区分：Triton kernel 通常只是查表、读写缓存；真正维护页表和缓存生命周期的是外部管理层。** 它不是在 kernel 内实现了一个小操作系统。

## 1. 背后的算法思想是什么？

核心是 **固定大小内存池 + 间接寻址 + 按需分配**。

### 固定大小内存池

预先准备一批 GPU KV cache block，每个 block 容纳固定数量 token 的 KV。

请求增长时，不必申请一块更大的连续空间再复制旧 KV，而是分配新的 block，并追加映射。

例如，每 block 容纳 16 个 token：

```text
请求有 16 个 token：block table = [7]
请求有 17 个 token：block table = [7, 2]
请求有 33 个 token：block table = [7, 2, 19]
```

这是**分页分配**的思想。好处是避免为每个请求预留最大长度的连续 KV 空间，也减少外部碎片；最后一个未填满的 block 仍可能存在内部碎片。

### 间接寻址

kernel 不需要知道 block 是如何分配的，只执行：

```python
logical_block = token_position // block_size
physical_block = block_table[request_id, logical_block]
offset = token_position % block_size
```

这更接近“**软件管理的分页存储**”，不是硬件 MMU，也不是多级页表遍历。我们刚看的 Triton 实现，使用的是按请求索引的直接页表项查找。

### 共享和复用是进一步的管理策略

在分页存储之上，可以再实现：

- **引用计数**：一个 block 被多个请求引用时，不能提前回收。
- **前缀复用**：匹配已有前缀后，复用对应 KV block。
- **Copy-on-write**：需要修改共享 block 时，先获得私有副本。

这些不是那几行 Triton 查表代码自动提供的能力，而是需要管理层维护相应策略和不变量。

## 2. 它到底是不是 pure function？

**严格按函数式编程的定义，写入 KV cache 的 kernel 不是纯函数，因为它原地修改了外部传入的存储。** 但“不是纯函数”不等于“自己拥有一个状态管理系统”。

可以分三层看：

| 层次 | 状态在哪里 | 负责什么 |
|---|---|---|
| KV cache 管理层 | block 分配情况、映射、引用关系等 | 分配、共享、回收，维护生命周期 |
| GPU cache 和元数据 | K/V 张量、block table、slot mapping | 保存跨推理步持续存在的数据 |
| Triton kernel | 本次启动的局部变量、累加器等 | 消费映射，读取/写入指定存储 |

Triton kernel 的局部状态一般随本次调用结束而消失。**跨调用持久存在的是传入的 GPU 内存，不是 kernel 自己私下保存的一份页表。**

对于我们看的两个路径：

- **读取路径**：`_fwd_kernel_stage1` 查页表、读历史 KV，写计算结果；这段寻址代码不修改页表。
- **缓存写入路径**：`reshape_and_cache_kernel_flash` 按 `slot_mapping` 原地更新 K/V；它不决定 slot 归哪个请求。

从数学上仍可以显式建模成状态转换：

\[
(\text{output},\ \text{KV}_{t+1})
= F(\text{Q},\text{K}_{new},\text{V}_{new},\text{KV}_{t},\text{mapping})
\]

实现上为了性能，通常直接更新同一块 KV 存储，而不是每一步复制出完整的 \(\text{KV}_{t+1}\)。

## 3. 为什么看起来像一个有状态的小系统？

因为 **自回归推理本来就是有状态的**：下一步计算需要之前各步累积的 KV。分页机制又增加了“这些 KV 放在哪里”的管理状态。

真正复杂的地方不是除法和取模，而是保证：

- 页表指向的 block 已分配，并包含正确请求的 KV；
- 新 KV 写完之后，读取它的计算才能执行；
- kernel 使用期间，相关 block 不能被回收、复用；
- 共享 block 不能被错误地原地覆盖。

**所以：整个分页 KV cache 子系统确实是一个有状态系统；Triton 算子通常只是这个系统的数据面执行器，而不是页表管理器。**
