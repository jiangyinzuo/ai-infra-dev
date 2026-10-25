"""基于运行时对象身份的 Python 对象流观察器。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Origin:
    """起点使用原源码赋值语句的位置，不依赖转换后的指令偏移。"""

    file: str
    function: str
    line: int
    variable: str


@dataclass(frozen=True)
class FlowConfig:
    root: Path
    origin: Origin
    run: Callable[[], object]
    # None 表示整个 root；否则仅这些相对目录或文件接受字节码转换。
    include: tuple[str, ...] | None = None
    max_events: int = 10000
    max_objects: int = 10000
    # 关闭时仅保留起点、变量绑定和同步函数边界，不建立载体或复制关系。
    observe_containers: bool = True
