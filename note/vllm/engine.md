---
title: Engine层
---

## LLMEngine

旧的类
```py
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

## EngineCoreClient

LLMEngine通过EngineCoreClient调用EngineCore

vllm/vllm:vllm/v1/engine/llm_engine.py:116-116
```python
class LLMEngine:
...
        self.engine_core = EngineCoreClient.make_client(
```

vllm/vllm:vllm/v1/engine/core_client.py:83-83
```python
class EngineCoreClient(ABC):
```

vllm/vllm:vllm/v1/engine/core_client.py:364-364
```python
class InprocClient(EngineCoreClient):
```
