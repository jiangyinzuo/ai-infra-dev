`select_moe_comm_method`方法

vllm/vllm-ascend:vllm_ascend/ascend_forward_context.py:41-45
```python
class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3
```

MC^2 FusedMC^2 都是AllToAll

MC^2的buffer有限，token量太大有问题，所以只支持decode，而且decode也会判断是否超过buffer上限。
MC^2没有对通信-计算掩盖

不同芯片代际，最佳通信算子不同。A3 A5支持all2allV
EP size < topk allgather
MC^2, fuse MC^2 只能decode阶段（只能容纳token量小的）
图模式：MC^2 fuse MC^2 可以入图。AllGather可以入图。AllToAllV不能入图，有HtoD操作。
和并行策略结合？

后续要推进MegaMoE分支

MegaMoE算子内部融合多流，后续合入MegaMoE。 知乎文章

“共享专家多流”是大模型推理中，特别是在混合专家（MoE） 模型上的一项关键性能优化技术。它的核心思想是让“共享专家”的计算与“路由专家”的通信/计算在两条独立的执行流（Stream）上重叠进行，从而隐藏等待时间，提升推理速度。

## 代码

patch: 上游通过函数实现工厂。写死了

MoERunner, AscendMoERunner

_forward_impl
路由专家、共享专家

三段式代码架构

```
vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py|93 col 7-20| class MoECommMethod(ABC):
vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py|211 col 25-38| class AllGatherCommImpl(MoECommMethod):
vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py|241 col 19-32| class MC2CommImpl(MoECommMethod):
vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py|263 col 24-37| class AlltoAllCommImpl(MoECommMethod):
vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py|289 col 24-37| class FusedMC2CommImpl(MoECommMethod):
```

共享专家
vllm/vllm-ascend:vllm_ascend/ops/fused_moe/shared_experts.py:99-99
```python
class AscendSharedExperts:
```

需要考虑纯DP，SP等4种情况（见jiaxian社区）

量化里面也有MoE

vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py

force_eplb 不是给客户用的，是开发用于评估极限性能的。
