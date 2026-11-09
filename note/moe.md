MoE的FFN中，虽然是降维，但代码中仍然叫“上投影”。这是历史遗留问题。稠密模型的中间维度都大于hidden state，但MoE中，单个FFN的中间维度小于hidden state，只有乘上选中的专家数量，维度才大于hidden state。

工厂方法

vllm/vllm:vllm/model_executor/layers/fused_moe/layer.py:93-95
```python
def FusedMoEFactory(
    num_experts: int,  # Global number of experts
    top_k: int,
```

MoeRunner

vllm/vllm:vllm/model_executor/layers/fused_moe/runner/moe_runner.py:229-230
```python
class MoERunner(MoERunnerInterface):
    """Standard MoE runner implementation for executing Mixture of Experts layers.
```

vLLM Ascend: patch_fused_moe.py

```python
class AscendMoERunner(MoERunner):
class AscendRoutedExperts(RoutedExperts):
class AscendFusedTopKRouter(AscendGroupedTopKRouter):
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
```

MoE通信算子

大EP走AllToAll性能好，卡数比较少（小于专家数量？）的时候走AllGather性能好。

vllm/vllm-ascend:vllm_ascend/ops/fused_moe/moe_comm_method.py:68-75
```python
def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[_moe_config_key(MoECommType.ALLTOALL, moe_config)] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[_moe_config_key(MoECommType.ALLGATHER, moe_config)] = AllGatherCommImpl(moe_config)
        _MoECommMethods[_moe_config_key(MoECommType.MC2, moe_config)] = MC2CommImpl(moe_config)
        _MoECommMethods[_moe_config_key(MoECommType.FUSED_MC2, moe_config)] = FusedMC2CommImpl(moe_config)
    else:
        _MoECommMethods[_moe_config_key(MoECommType.ALLGATHER, moe_config)] = AllGatherCommImpl(moe_config)
```


```
vllm_ascend/ops/fused_moe/token_dispatcher.py|68 col 7-25| class MoETokenDispatcher(ABC, Generic[TMoECombineMetadata]):
vllm_ascend/ops/fused_moe/token_dispatcher.py|110 col 30-48| class TokenDispatcherWithMC2(MoETokenDispatcher[MoEMC2CombineMetadata]):
vllm_ascend/ops/fused_moe/token_dispatcher.py|353 col 36-54| class TokenDispatcherWithAllGather(MoETokenDispatcher[MoEAllGatherCombineMetadata]):
vllm_ascend/ops/fused_moe/token_dispatcher.py|469 col 35-53| class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
```
