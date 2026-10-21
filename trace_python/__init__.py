"""Python 函数追踪配置接口。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Target:
    file: str
    function: str  # co_qualname，例如 KVCacheManager.allocate_slots
    # 要保存的局部变量或属性路径，如 request.num_computed_tokens；不执行表达式。
    # 空元组表示只记录调用栈，不采集字段值。
    fields: tuple[str, ...] = ()
    # 每个目标最多保存前 limit 次命中的详细事件；超出后仍累计命中次数。
    # 0 表示只计数，不保存详细事件。
    limit: int = 2


@dataclass(frozen=True)
class TraceConfig:
    # 源码根目录，必须为绝对路径；也是目标程序运行时的工作目录。
    root: Path
    # 执行场景的入口，接收无参 start 回调；调用 start() 后才开始记录目标函数。
    run: Callable[[Callable[[], None]], None]
    # 要采集的函数及其字段、详细事件数量上限；文件路径相对于 root。
    targets: tuple[Target, ...]
    # 仅在执行场景期间设置的环境变量；结束或异常时恢复，不改变启动 shell。
    # 使用工厂为每个配置创建独立字典，避免多个配置共享可变默认值。
    env: dict[str, str] = field(default_factory=dict)
