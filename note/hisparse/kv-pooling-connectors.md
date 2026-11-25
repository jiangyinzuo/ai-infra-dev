# KV Pooling Connectors

## 观察重点

- Store cold save：本地 KV 计算后写入远端 Store。
- Store hit：清理本地 APC，不删除远端对象，再次请求触发 GET。
- 观察 `MooncakeStoreScheduler`、`MooncakeStoreWorker`、传输线程和成功字节数。

## 验收边界

- 同一引擎、同一 Store namespace。
- 只验证本地 APC reset 后的 TCP Store GET。
- 不等价于 HiSparse + P/D + Store 的组合验证。
