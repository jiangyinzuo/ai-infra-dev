"""采集目标与受控场景配置；不持有进程内运行状态。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Target:
    file: str
    function: str  # co_qualname，例如 KVCacheManager.allocate_slots
    # 要保存的局部变量或属性路径；读取属性可能执行 property。
    fields: tuple[str, ...] = ()
    # 每进程最多保存 limit 个详细事件；超出后仍计数，0 表示仅计数。
    limit: int = 2


@dataclass(frozen=True)
class TraceConfig:
    # 绝对源码根目录，也是受控场景的工作目录。
    root: Path
    # 仅由命令行入口调用，不传入子进程，也不在子进程重新加载配置。
    run: Callable[[Callable[[], None]], None]
    targets: tuple[Target, ...]
    # 仅在受控场景期间覆盖环境，退出时恢复。
    env: dict[str, str] = field(default_factory=dict)
