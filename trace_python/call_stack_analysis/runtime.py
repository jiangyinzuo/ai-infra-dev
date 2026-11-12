"""可序列化的会话配置与每个进程独立的采集生命周期。"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import shlex
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from .configuration import Target
from .output import open_writer
from .tracing import Tracer

_Result = TypeVar("_Result")
_current: Tracer | None = None


def _without_inherited_hook(hook: Any) -> Any:
    owner = getattr(hook, "__self__", None)
    if isinstance(owner, Tracer) and owner.pid != os.getpid():
        return None
    return hook


@contextlib.contextmanager
def _label_environment(label: str | None) -> Iterator[None]:
    previous = os.environ.get("TRACE_PROCESS_LABEL")
    if label is not None:
        os.environ["TRACE_PROCESS_LABEL"] = label
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRACE_PROCESS_LABEL", None)
        else:
            os.environ["TRACE_PROCESS_LABEL"] = previous


@dataclass(frozen=True)
class TraceSession:
    """只传递配置；writer、锁、计数器都在进入目标进程后创建。"""

    root: Path
    targets: tuple[Target, ...]
    output_prefix: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    config_path: Path | None = None
    process_label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("root 必须为绝对路径")
        object.__setattr__(self, "root", self.root.resolve())
        object.__setattr__(self, "output_prefix", self.output_prefix.resolve())

    def run(self, target: Callable[..., _Result], /,
            *args: Any, **kwargs: Any) -> _Result:
        """用作 multiprocessing.Process 的 target；保留业务返回值和异常。"""
        with self.capture():
            return target(*args, **kwargs)

    @contextlib.contextmanager
    def capture(
        self, *, active: bool = True, output_prefix: Path | None = None,
    ) -> Iterator[Tracer]:
        """当前进程安装一次；默认立即启用，退出时写摘要并恢复 hooks。"""
        global _current
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("请在进程主线程安装采集器")
        if _current is not None and _current.pid == os.getpid():
            raise RuntimeError("当前进程已安装采集器，不支持嵌套采集")

        pid = os.getpid()
        instance = f"{pid}-{uuid.uuid4().hex}"
        identity = {
            "run_id": self.run_id,
            "process_instance_id": instance,
            "pid": pid,
            "ppid": os.getppid(),
            "process_name": multiprocessing.current_process().name,
            "process_label": (
                self.process_label if self.process_label is not None
                else os.environ.get("TRACE_PROCESS_LABEL")
            ),
        }
        prefix = (
            output_prefix.resolve() if output_prefix is not None
            else self.output_prefix
        )
        # 在任何 cwd 切换之前记录真实启动上下文。
        invocation = {
            "argv": list(sys.orig_argv),
            "command": shlex.join(sys.orig_argv),
            "cwd": str(Path.cwd()),
        }
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started_monotonic_ns = time.monotonic_ns()
        common = {**identity, **invocation, "started_at": started_at,
                  "trace_start_monotonic_ns": started_monotonic_ns}
        previous_trace = _without_inherited_hook(sys.gettrace())
        previous_thread_trace = _without_inherited_hook(threading.gettrace())

        with _label_environment(identity["process_label"]), open_writer(
            prefix, identity=identity if output_prefix is None else None,
        ) as writer:
            writer.update_identity(identity)
            tracer = Tracer(self.root, self.targets, writer, identity, active=active)
            writer.emit({
                "type": "run",
                **common,
                **identity,
                "config": str(self.config_path) if self.config_path else None,
                "root": str(self.root),
                "python": sys.executable,
                "start_method": multiprocessing.get_start_method(allow_none=True),
                "scope": "current-process Python threads; no CUDA kernel tracing",
            })
            started = time.monotonic()
            status, error = "ok", None
            _current = tracer
            try:
                # 3.12 的接口同时覆盖已有 Python 线程与随后创建的线程。
                threading.settrace_all_threads(tracer.trace)
                sys.settrace(tracer.trace)
                yield tracer
            except BaseException as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
                raise
            finally:
                # 未包装的 fork 分支也可能展开父上下文；不能替父进程关闭或写摘要。
                if os.getpid() == pid:
                    tracer.stop()
                    threading.settrace_all_threads(previous_thread_trace)
                    sys.settrace(previous_trace)
                    _current = None
                    writer.emit({
                        "type": "summary",
                        **common,
                        **identity,
                        "status": status,
                        "error": error,
                        "events": tracer.sequence,
                        "counts": dict(tracer.counts),
                        "elapsed_seconds": time.monotonic() - started,
                    })
