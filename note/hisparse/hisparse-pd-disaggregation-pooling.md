https://github.com/PrimeIntellect-ai/prime-rl/pull/4023

```mermaid
flowchart LR
    STORE["Mooncake Store<br/>跨实例共享 KV 池"]
    P["Prefill<br/>NIXL + MooncakeStoreConnector"]
    D["Decode<br/>NIXL + HiSparseConnector"]
    HOST["Decode 本地 Host KV"]
    HOT["Decode GPU hot buffer"]

    STORE <-->|"前缀加载 / 保存"| P
    P -->|"NIXL：本轮 P→D KV 交接"| D
    D --- HOST
    HOST -->|"按需加载 top-k 缺失行"| HOT
```


## 如果优先要“完整共享池”，先不叠 HiSparse

vLLM 官方完整方案是：

- `MooncakeConnector`：P→D 点对点传输。
- `MooncakeStoreConnector`：跨实例共享 KV。
- `MultiConnector`：组合两者。
- D 可用 `save_decode_cache=true`，将生成阶段完成的 KV blocks 写入池，供后续请求复用。

[官方指南](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/#disaggregated-prefill-decode-xpyd) 给出了 master、Store 配置、P/D 命令和代理入口。
