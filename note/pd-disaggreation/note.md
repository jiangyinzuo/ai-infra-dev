## nixl
```bash
uv pip install --python .venv/bin/python \
  'nixl==1.5.2' \
  'nixl-cu13==1.5.2'
```

## Pull模式和Push模式

**Pull 模式（拉取）**
由 **Decode 节点主动发起**。当 Decode 节点需要 KV Cache 时，向 Prefill 节点（或共享存储）发出读取请求，类似“缺页中断”式的按需加载。Decode 节点掌握传输的**控制权**，可以根据自身的显存状态和缓存命中情况决定何时拉取。

优点
- **Decode 侧自主控制**：Decode 节点可以根据当前显存压力、Prefix Caching 命中情况，按需决定拉取的内容和时机，避免拉取无效数据。
- **实现与解耦简单**：无需 Prefill 节点预先知道 Decode 节点的地址，Decode 可自主选择从哪个 Prefill 或存储层拉取，降低了 Prefill 侧的编排复杂度。
- **有利于 Prefill 显存释放**：在理想情况下，Decode 拉取完成后通知 Prefill，后者即可立即释放 KV Cache，节省 Prefill 端显存。

缺点
- **TTFT 被拉取延迟直接阻塞**：首个 Decode 步骤必须等待 KV Cache 完全拉取到位后才能开始，拉取时间直接计入首 Token 延迟（TTFT）。
- **存在串行路径与竞态风险**：如果采用“拉完再通知释放”的语义，跨节点场景下容易出现 Prefill 侧 KV 释放被拖延的竞态问题，增加了显存管理的复杂性。
- **传输与计算较难重叠**：Decode 侧发起拉取时，Prefill 侧可能已经完成计算，传输过程难以与 Prefill 的计算过程有效重叠。

**Push 模式（推送）**
由 **Prefill 节点主动发起**。Prefill 节点在生成 KV Cache 后，将其直接写入 Decode 节点预先分配好的内存中，Decode 节点被动接收。传输的**时机**主要由 Prefill 侧决定，而非 Decode 侧的需求驱动。

优点
- **优化 TTFT**：KV Cache 在 Decode 需要时已经在本地，消除了首 Token 生成前的拉取等待，确保推理计算无缝衔接。
- **易于实现流水化（Pipelining）**：Prefill 节点可以在**逐层计算**的同时，将已算完层的 KV Cache 立即推送到 Decode 端，使传输与后续层的 Prefill 计算并行重叠，从而隐藏部分传输延迟。
- **Prefill 侧显存即时释放**：Prefill 推送完成后即可释放本地 KV Cache，不存在“等待 Decode 拉取”造成的显存驻留延迟。

缺点
- **耦合度更高，需要外部编排**：Prefill 节点必须知道将数据推送给哪个 Decode 节点，依赖外部调度器进行配对和内存预分配。
- **Decode 侧被动，可能浪费显存/带宽**：Decode 无法根据自身的 Prefix Caching 命中情况拒绝接收。如果 Decode 本地已有部分缓存，Prefill 仍可能推送重复数据，造成带宽和显存浪费。
- **接收端就绪依赖**：如果 Decode 节点尚未准备好接收缓冲区，Push 操作可能无处写入，需要额外的同步或流控机制。

**选型参考**

两种模式并无绝对的优劣，选择取决于系统的优化目标与部署约束：

- 若首要目标是**降低 TTFT、追求极致响应速度**，且 Prefill 与 Decode 之间具备稳定的高速互联和中心化调度能力，**Push 模式（尤其是层级流水线 Push）** 通常是更直接的选择，LMCache 的 PDBackend 即采用了这一路线。
- 若更看重**Decode 侧的灵活性与系统解耦**，愿意以一定的 TTFT 代价换取更简单的资源管理和更低的 Prefill 编排复杂度，**Pull 模式**更为合适。Mooncake 采用的“Pull-with-cache”策略，正是在 Pull 的基础上通过全局缓存索引来尽量减少不必要的传输。
- 实际上，部分框架如 **vLLM KV Connector V1** 并未绑定单一模式，而是同时提供 Push 和 Pull 的 API 挂钩，由外部编排器根据实际负载和拓扑动态选择。

## 工厂

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:27-27
```python
class KVConnectorFactory:
```

nixl默认是pull，也可以手动设置push

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/factory.py:125-131
```python
        elif connector_name in cls._registry:
            if connector_name == "NixlConnector":
                mode = kv_transfer_config.get_from_extra_config("transfer_mode", "pull")
                if mode not in ("pull", "push"):
                    raise ValueError(f"Invalid NixlConnector transfer_mode: {mode!r}")
                connector_name = f"Nixl{mode.capitalize()}Connector"
            connector_cls = cls._registry[connector_name]()
```

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:36-37
```python
class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""
```

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:82-83
```python
class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""
```

## 公共接口

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:182-182
```python
class KVConnectorBase_V1(ABC):
```

### start_load_kv: pull和push共用

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/base.py:360-375
```python
    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector to vLLM's paged
        KV buffer. Loads required by the current forward start before it;
        independent asynchronous loads may start after it is submitted.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        pass
```

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:399-403
```python
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self.connector_worker, NixlPullConnectorWorker)
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)
```

vllm/vllm:vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:433-443
```python
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Drive push processing on the worker.

        The worker enqueues registrations / finished blocks for the
        background `nixl-push-writer` thread; the writer issues the
        WRITE transfers and polls NIXL notifs without further
        engine-thread involvement.
        """
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)
```
