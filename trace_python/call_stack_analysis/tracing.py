"""各进程共用的函数匹配、字段快照与真实 Python 栈采集。"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

from .configuration import Target
from .output import Writer


def snapshot(value: Any, depth: int = 0) -> Any:
    """不调用 repr 或搬运 tensor；容器有界，未知对象只保存类型。"""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        result: dict[str, Any] = {"type": type(value).__name__, "length": len(value)}
        if depth >= 3:
            return {**result, "truncated": True}
        if isinstance(value, dict):
            result["items"] = [
                [snapshot(key, depth + 1), snapshot(item, depth + 1)]
                for key, item in islice(value.items(), 16)
            ]
        else:
            result["items"] = [snapshot(item, depth + 1) for item in islice(value, 16)]
        return result
    if depth >= 3:
        return {"type": type(value).__qualname__, "truncated": True}
    if all(hasattr(value, attr) for attr in ("shape", "dtype", "device")):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return {"type": type(value).__qualname__}


def capture_fields(frame: FrameType | None, fields: tuple[str, ...]) -> dict[str, Any]:
    if frame is None:
        return {path: {"unavailable": "NoCallerFrame"} for path in fields}
    result = {}
    for path in fields:
        parts = path.split(".")
        try:
            value = frame.f_locals[parts[0]]
            for part in parts[1:]:
                value = getattr(value, part)
            result[path] = snapshot(value)
        except (KeyError, AttributeError) as exc:
            result[path] = {"unavailable": type(exc).__name__}
    return result


def locate(
    root: Path, targets: tuple[Target, ...]
) -> dict[tuple[str, int, str], Target]:
    """编译但不执行源码；按实际 code object 的限定名验证目标。"""
    found = {}
    for target in targets:
        if target.limit < 0 or any(
            not all(p.isidentifier() for p in f.split("."))
            for f in (*target.fields, *target.caller_fields)
        ):
            raise ValueError(f"无效 limit 或字段路径：{target}")
        path = (root / target.file).resolve()
        code = compile(path.read_bytes(), str(path), "exec")
        matches = []

        def walk(node: CodeType) -> None:
            if node.co_qualname == target.function:
                matches.append(node)
            for child in node.co_consts:
                if isinstance(child, CodeType):
                    walk(child)

        walk(code)
        if len(matches) != 1:
            raise ValueError(
                f"{target.file}:{target.function} 匹配到 {len(matches)} 个函数"
            )
        key = (str(path), matches[0].co_firstlineno, target.function)
        if key in found:
            raise ValueError(f"重复采集目标：{target.file}:{target.function}")
        found[key] = target
    return found


def collect_stack(frame: FrameType, root: Path) -> list[dict[str, Any]]:
    stack = []
    parent: FrameType | None = frame
    while parent is not None:
        path = Path(parent.f_code.co_filename)
        filename = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        stack.append({
            "file": filename,
            "line": parent.f_lineno,
            "function": parent.f_code.co_qualname,
        })
        parent = parent.f_back
    stack.reverse()
    return stack


class Tracer:
    def __init__(
        self, root: Path, targets: tuple[Target, ...], writer: Writer,
        identity: dict[str, Any], *, active: bool,
    ) -> None:
        self.root, self.writer = root, writer
        self.identity = identity
        self.pid = os.getpid()
        self.targets = locate(root, targets)
        self.counts: Counter[str] = Counter()
        self.sequence = 0
        self.active = active
        self.lock = threading.Lock()

    def start(self) -> None:
        """只启用当前进程，不改变已创建子进程的开关。"""
        with self.lock:
            self.active = True

    def stop(self) -> None:
        with self.lock:
            self.active = False

    def trace(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "call" or not self.active:
            return
        code = frame.f_code
        target = self.targets.get(
            (code.co_filename, code.co_firstlineno, code.co_qualname)
        )
        # fork 会复制 hook 和锁；在子进程显式安装前，绝不触碰父进程的状态或 writer。
        if target is None or os.getpid() != self.pid:
            return
        observed_at = time.monotonic_ns()
        with self.lock:
            # 关闭采集可能发生在另一个线程等待这把锁期间。
            if not self.active:
                return
            name = f"{target.file}:{target.function}"
            self.counts[name] += 1
            if self.counts[name] > target.limit:
                return
            self.sequence += 1
            record = {
                "type": "event",
                **self.identity,
                "thread_id": threading.get_ident(),
                "native_thread_id": threading.get_native_id(),
                "monotonic_ns": observed_at,
                "sequence": self.sequence,
                "target": name,
                "hit": self.counts[name],
                "values": capture_fields(frame, target.fields),
                "stack": collect_stack(frame, self.root),
            }
            if target.caller_fields:
                record["caller_values"] = capture_fields(frame.f_back, target.caller_fields)
            self.writer.emit(record)
