"""函数追踪公共接口：受控场景配置和可传入子进程的采集会话。"""

from .configuration import Target, TraceConfig
from .runtime import TraceSession

__all__ = ["Target", "TraceConfig", "TraceSession"]
