上游注册插件

vllm/vllm-ascend:setup.py:506-517
```python
    entry_points={
        "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
        "vllm.general_plugins": [
            "ascend_kv_connector = vllm_ascend:register_connector",
            "ascend_model_loader = vllm_ascend:register_model_loader",
            "ascend_service_profiling = vllm_ascend:register_service_profiling",
            "ascend_model = vllm_ascend:register_model",
        ],
        "ms_service_metric.providers": [
            "vllm-ascend = vllm_ascend.observability.ms_metrics:get_metric_provider",
        ],
    },
```

vllm/vllm-ascend:vllm_ascend/utils.py:864-869
```python
    global REGISTERED_ASCEND_OPS
    REGISTERED_ASCEND_OPS = {
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        "SiluAndMulWithClamp": AscendSiluAndMulWithClamp,
        "RotaryEmbedding": AscendRotaryEmbedding,
```
