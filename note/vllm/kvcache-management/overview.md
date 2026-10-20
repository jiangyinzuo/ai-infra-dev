# 和kvcache management相关的vllm的代码

**主要分散在 `v1/core/`、`v1/worker/`、`v1/attention/` 和 KV transfer/offload 相关目录。** 原因是 KV cache management 横跨了**调度、逻辑 block 管理、实际存储、设备寻址和数据搬运**，不是一个单独模块。

下面按当前仓库的职责划分。

## 1. 核心管理：`vllm/v1/core/`

**如果你想理解“页表背后的管理系统”，优先读这里，而不是 Attention kernel。**

| 文件 | 职责 |
|---|---|
| [`kv_cache_manager.py`](vllm/v1/core/kv_cache_manager.py#L131) | 请求级管理入口：查询已缓存的 block、分配新 token 所需空间、释放请求的 block |
| [`block_pool.py`](vllm/v1/core/block_pool.py#L136) | 管理 block 池：分配、释放、缓存 block，维护空闲/可淘汰队列 |
| [`single_type_kv_cache_manager.py`](vllm/v1/core/single_type_kv_cache_manager.py#L49) | 不同类型缓存的管理策略，如 Full Attention、Sliding Window、Mamba |
| [`kv_cache_coordinator.py`](vllm/v1/core/kv_cache_coordinator.py#L89) | 协调多个 KV cache group，处理混合层类型 |
| `kv_cache_utils.py` | block 数据结构及缓存相关公共工具 |
| `kv_cache_planning.py`、`kv_cache_config_builder.py` | KV cache 容量、分组与配置规划 |

`KVCacheManager` 的几个关键入口：

```python
get_computed_blocks(...)  # 查询可复用的本地缓存
allocate_slots(...)      # 为请求接下来的计算分配空间
free(...)                # 释放请求持有的 block
```

这里的“分配 block”主要是**分配已有 cache 池中的 block 使用权**，不要理解成每次都调用 GPU allocator 新建张量。

## 2. 调度与生命周期：`vllm/v1/core/sched/`

关键文件：[`scheduler.py`](vllm/v1/core/sched/scheduler.py)

调度器决定：

- 哪些请求本轮执行、各执行多少 token；
- 何时调用 `allocate_slots()`；
- 请求结束或发生抢占时，何时释放缓存；
- 有异步操作尚未完成时，是否需要延迟回收 block。

例如，当前代码的 [`_free_request_blocks()`](vllm/v1/core/sched/scheduler.py#L2857) 就包含**延迟释放**逻辑，避免仍在执行的 GPU 操作写到已被复用的 block。

**管理器负责“怎么分配”；调度器负责“什么时候分配、什么时候释放”。**

## 3. 实际存储与设备页表：`vllm/v1/worker/`

这一层将调度侧的 block 信息变成 GPU 执行可以使用的张量。

| 文件 | 职责 |
|---|---|
| [`gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py#L6943) | KV cache 张量初始化及模型执行接入 |
| [`block_table.py`](vllm/v1/worker/block_table.py#L49) | Worker 侧 block table 数据结构 |
| [`gpu/model_runner.py`](vllm/v1/worker/gpu/model_runner.py#L622) | 另一条 model runner 路径中的 KV cache 初始化与执行接入 |
| [`gpu/block_table.py`](vllm/v1/worker/gpu/block_table.py#L17) | 设备侧页表组织、batch 页表收集、slot mapping 计算 |

特别是 `gpu/block_table.py`：

- `append_block_ids()`：追加请求的 block ID；
- `gather_block_tables()`：组织当前 batch 使用的页表；
- `compute_slot_mappings()`：计算当前 token 的 KV 写入位置；
- `_compute_slot_mappings_kernel()`：对应的 Triton kernel。

**这是连接“逻辑 block 管理”和“GPU 算子寻址”的关键桥梁。**

## 4. KV 读写与计算：`vllm/v1/attention/`、`csrc/`

| 位置 | 职责 |
|---|---|
| `v1/attention/backend.py` | 定义公共元数据与后端契约，包括 block table、slot mapping、布局要求 |
| `v1/attention/backends/` | 将元数据适配为各后端的执行参数，调用具体 kernel |
| `v1/attention/ops/` | Triton 等实现：分页读取 KV、写入 KV、Attention 计算 |
| `csrc/cache_kernels.cu` 等 | CUDA/HIP 底层缓存操作实现 |

你前面看的两个文件就属于这一层：

- `triton_reshape_and_cache_flash.py`：按 `slot_mapping` 写 KV；
- `triton_decode_attention.py`：按页表读 KV 并计算。

**这里消费映射，通常不负责请求级的 block 分配和回收。**

## 5. Offload 与跨实例传输

| 目录 | 职责 |
|---|---|
| `vllm/distributed/kv_transfer/` | KV transfer 接入与传输状态管理 |
| `.../kv_connector/v1/` | 各种 connector，包括 offload、外部缓存等集成 |
| `.../kv_connector/v1/offloading/` | Offload 的 scheduler/worker 两侧逻辑 |
| `vllm/v1/kv_offload/` | 卸载管理、存储介质与传输接口 |
| `vllm/v1/kv_offload/cpu/` | CPU offload 的具体实现 |
| `vllm/v1/simple_kv_offload/` | 另一套简化 offload 路径 |

这一部分才涉及你前面关注的 **HBM ↔ CPU DRAM 搬运**。KV transfer 也不只服务于 CPU offload，还用于跨实例 KV 传输等场景。

## 6. 公共契约与配置

- **`vllm/v1/kv_cache_interface.py`**：KV cache spec、配置等公共类型，连接管理层与执行层。
- **`vllm/config/cache.py`**：缓存配置，包括 offload 容量和后端等选项。
- **`vllm/distributed/kv_events.py`**：KV cache 事件相关接口。
- **`vllm/v1/hisparse/`**：HiSparse 专用缓存管理路径，涉及该功能时再看，不必作为普通分页 KV 的入门入口。

### 建议阅读主线

针对你现在关注的“分页映射背后如何管理”，顺序是：

**`scheduler.py` → `kv_cache_manager.py` → `block_pool.py` → `worker/gpu/block_table.py` → Attention kernel。**

这条线能把 **“请求需要更多 KV → 分配 block → 更新页表 → 生成写入位置 → kernel 读写”** 串起来；先不展开 offload、混合缓存和特殊模型路径。
