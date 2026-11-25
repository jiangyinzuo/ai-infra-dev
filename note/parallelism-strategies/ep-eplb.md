# EP / EPLB

## EP

- Expert Parallel：MoE experts 分布在多个 rank。
- Router 选择逻辑 expert。
- dispatch 将 token 送到目标 rank，expert 计算后 combine。

## EPLB

- 根据窗口内 expert load 调整 logical-to-physical 映射。
- 可能触发 expert weight 的 rank 间搬运。
- 观察路由、dispatch/combine、重平衡、映射提交和权重迁移。
