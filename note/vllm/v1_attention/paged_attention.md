- **讲系统设计时**：PagedAttention 是一种 KV cache 管理方案，重点是减少显存碎片和过量预留，提高并发容量。
- **讲算子实现时**：Paged Attention 是能够直接消费分页 KV cache 的 Attention 算法/kernel。

# Paged Attention 负责如何组织、寻址并计算分块的 KV cache

**会参与 Paged Attention 的计算，但不主要负责 HBM → DRAM 的 KV cache offload。这里要先纠正一个概念：Paged Attention ≠ KV cache offload。**

## 1. Paged Attention 不意味着把 KV 搬到 CPU

Paged Attention 的核心是：**把 KV cache 分成 block，通过 block table 将请求的逻辑 block 映射到实际存储 block**，不要求一个请求的 KV 在显存里连续存放。

即使 KV cache **全部留在 GPU HBM 中**，也可以使用 Paged Attention。

| 机制 | 解决的问题 | 是否必须搬到 CPU |
|---|---|---|
| Paged Attention | 如何组织、寻址并计算分块的 KV cache | 不需要 |
| KV cache offload | 如何将 KV 保存到 CPU DRAM，并在需要时加载回 GPU | 需要 GPU ↔ CPU 数据传输 |

“分页”的类比容易让人想到操作系统换页，但**不能据此理解为 GPU kernel 访问缺页时自动从 CPU 调入**。

## 2. `v1/attention/` 负责哪一部分？

它负责**消费分页信息，执行 Attention**：

- `block_table_tensor`：描述请求的逻辑 KV block 对应哪些实际 block。
- `slot_mapping`：描述当前 token 的 K/V 应写入哪个 cache slot。
- 后端根据这些信息组织 kernel 调用，读取 KV cache 并计算。
- KV cache 写入可能包含在后端 `forward()` 中，也可能拆开执行；你贴的 `forward_includes_kv_cache_update` 就是在描述这种差异。

这些公共元数据定义在 [`CommonAttentionMetadata`](vllm/v1/attention/backend.py#L386) 中。

**但决定哪些 block 要卸载、分配 CPU 存储、发起 GPU ↔ CPU 拷贝，不是 `AttentionBackend` 的主要职责。**

## 3. 当前仓库里，offload 在哪里？

| 位置 | 职责 |
|---|---|
| [`distributed/kv_transfer/kv_connector/v1/offloading_connector.py`](vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py#L53) | Offloading connector 接入层 |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 调度侧卸载逻辑 |
| `distributed/kv_transfer/kv_connector/v1/offloading/worker.py` | Worker 侧卸载逻辑 |
| [`v1/kv_offload/`](vllm/v1/kv_offload/base.py#L650) | 卸载管理与传输接口 |
| [`v1/kv_offload/cpu/gpu_worker.py`](vllm/v1/kv_offload/cpu/gpu_worker.py#L793) | CPU offload 的实际传输实现 |

传输接口明确区分两个方向：

```python
# vllm/v1/kv_offload/base.py
submit_store(...)  # GPU → 卸载介质，例如 CPU DRAM
submit_load(...)   # 卸载介质 → GPU
```

另外，当前代码还有 `v1/simple_kv_offload/` 路径。原生 offload 配置会根据 `VLLM_USE_SIMPLE_KV_OFFLOAD` 选择 `SimpleCPUOffloadConnector` 或 `OffloadingConnector`，见 [`config/vllm.py:1143`](vllm/config/vllm.py#L1143)。

**所以可以这样记：`v1/attention/` 管“如何利用 GPU 上的分块 KV 做计算”；KV offload 子系统管“如何在 GPU HBM 与 CPU DRAM 之间保存和加载 KV”。两者协作，但不是同一件事。**

# Paged Attention的思想，类似操作系统让一个进程从逻辑上看到完整连续的内存地址

**是，核心类比是对的：逻辑上连续，物理上可以不连续，通过映射表连接起来。** 但 Paged Attention 通常不是依赖操作系统/MMU 的虚拟地址机制，而是由 Attention kernel 显式使用 block table 寻址。

### 对应关系

| 操作系统分页 | Paged Attention |
|---|---|
| 一个进程的虚拟地址空间 | 一个请求按 token 顺序排列的逻辑 KV 序列 |
| 虚拟页 | 逻辑 KV block，容纳固定数量 token 的 KV |
| 物理页框 | GPU KV cache 池中的实际 block |
| 页表 | 请求的 block table |
| 页内偏移 | token 在 block 内的偏移 |

例如，假设每个 block 容纳 16 个 token，一个请求有 48 个 token：

```text
逻辑 token 范围：  [0..15]  [16..31]  [32..47]
逻辑 block 编号：     0         1         2
                     │         │         │
block table：        7         2        19
                     │         │         │
GPU cache block：   #7        #2       #19
```

请求的 KV **在逻辑上仍然按 token 0～47 排列**，但实际放在 GPU cache 池的第 7、2、19 个 block 中。

要读 token 20 的 KV，寻址思路是：

```python
logical_block = 20 // 16       # 1
offset_in_block = 20 % 16      # 4
physical_block = block_table[1]  # 2

# 读取 GPU cache 中 block #2 内偏移为 4 的 KV
```

实际张量布局因后端而异，但这个映射思想相同。

### 和操作系统虚拟内存的关键区别

1. **不是给 kernel 一个假的连续指针。**  
   操作系统分页通常由硬件 MMU 透明地翻译地址；Paged Attention 的 kernel 知道 KV 是分块的，显式读取 block table。

2. **分页不等于换出。**  
   GPU 上的 KV block 可以始终驻留 HBM；CPU offload 是另一套存储与传输机制，不是这里天然附带的“缺页自动调入”。

3. **映射对象是 KV 序列，不是通用进程地址空间。**  
   模型仍按逻辑 token 顺序做 Attention；底层不必为每个请求预留一整块连续、足够容纳最大序列长度的显存。

所以，更准确的说法是：**Paged Attention 借用了操作系统“分页存储 + 页表映射”的思想，把请求的逻辑 KV 序列与 GPU 上的实际 KV 存储位置解耦。**
