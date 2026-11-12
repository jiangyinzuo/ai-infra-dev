"""远程请求的进程内入口；配置仅使用控制端序列化的数据。"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .configuration import Target
from .runtime import TraceSession
from .tracing import Tracer


@dataclass
class _Capture:
    pid: int
    run_id: str
    context: contextlib.AbstractContextManager[Tracer]
    tracer: Tracer
    prefix: str

    def response(self, action: str, status: str = "ok") -> dict[str, Any]:
        return {
            "status": status,
            "action": action,
            "pid": self.pid,
            "run_id": self.run_id,
            "output_prefix": self.prefix,
            "process_instance_id": self.tracer.identity["process_instance_id"],
        }


_active: _Capture | None = None
_last_closed: dict[str, Any] | None = None


def _close_capture(capture: _Capture) -> dict[str, Any]:
    global _active, _last_closed
    if capture.pid != os.getpid():
        raise RuntimeError("不能在 fork 子进程中关闭父进程的采集器")
    # __exit__ 即使因输出失败而抛出也不能再调用；失败不可被后续 stop 当作成功。
    _active = None
    _last_closed = None
    capture.context.__exit__(None, None, None)
    response = capture.response("stop")
    _last_closed = response
    return response


def _close_at_exit() -> None:
    capture = _active
    if capture is None or capture.pid != os.getpid():
        return
    try:
        _close_capture(capture)
    except BaseException:
        _report_failure()


def _report_failure() -> None:
    # 响应目录不可写时没有可用的控制通道；保留原始异常，不抛入业务进程。
    try:
        traceback.print_exc(file=sys.stderr)
    except BaseException:
        pass


def _write_response(path: Path, response: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(response, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _absolute_path(request: dict[str, Any], key: str) -> Path:
    value = request[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} 必须为路径字符串")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{key} 必须为绝对路径")
    return path


def _start(request: dict[str, Any]) -> _Capture:
    global _active
    if _active is not None and _active.pid == os.getpid():
        raise RuntimeError(f"当前进程已有 attach 采集：{_active.run_id}")
    targets = tuple(
        Target(
            file=target["file"], function=target["function"],
            fields=tuple(target["fields"]), limit=target["limit"],
            caller_fields=tuple(target["caller_fields"]),
        )
        for target in request["targets"]
    )
    session = TraceSession(
        root=_absolute_path(request, "root"), targets=targets,
        output_prefix=_absolute_path(request, "output_prefix"),
        run_id=request["run_id"], config_path=_absolute_path(request, "config_path"),
    )
    context = session.capture()
    tracer = context.__enter__()
    capture = _Capture(
        pid=os.getpid(), run_id=session.run_id, context=context, tracer=tracer,
        prefix=f"{session.output_prefix}.{tracer.identity['process_instance_id']}",
    )
    # 必须持有生成器上下文，dispatch 返回时不能触发 GC 而提前关闭采集。
    _active = capture
    return capture


def _stop(run_id: str) -> dict[str, Any]:
    capture = _active
    if capture is not None and capture.pid == os.getpid():
        if capture.run_id != run_id:
            raise RuntimeError(f"attach run_id 不匹配：当前为 {capture.run_id}")
        return _close_capture(capture)
    if (
        _last_closed is not None
        and _last_closed["pid"] == os.getpid()
        and _last_closed["run_id"] == run_id
    ):
        return dict(_last_closed)
    raise RuntimeError("当前进程没有匹配的 attach 采集")


def dispatch(request_path: str, response_path: str, cancel_path: str) -> None:
    """处理一条请求；任何失败均通过响应或 stderr 报告，不影响业务调用栈。"""
    request: dict[str, Any] = {}
    installed: _Capture | None = None
    try:
        response_file = Path(response_path)
        cancellation = Path(cancel_path)
        payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("attach 请求必须为 JSON object")
        request = payload
        action = request["action"]
        if action not in ("start", "stop"):
            raise ValueError(f"未知 attach action：{action}")
        if type(request["pid"]) is not int or request["pid"] != os.getpid():
            raise ValueError("attach 请求 PID 与当前进程不匹配")
        if not isinstance(request["run_id"], str) or not request["run_id"]:
            raise ValueError("attach run_id 必须为非空字符串")
        if action == "stop":
            # 即使控制端已超时，匹配的 stop 仍须真正关闭采集。
            _write_response(response_file, _stop(request["run_id"]))
            return
        if cancellation.exists():
            _write_response(response_file, {
                "status": "cancelled", "action": action,
                "pid": os.getpid(), "run_id": request["run_id"],
            })
            return
        installed = _start(request)
        # 先发布 ack，再检查取消：控制端先写 marker 后读 ack，双方不会漏掉取消。
        _write_response(response_file, installed.response("start"))
        if cancellation.exists():
            _close_capture(installed)
            _write_response(response_file, installed.response("start", "cancelled"))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        details = traceback.format_exc()
        if installed is not None and _active is installed:
            try:
                _close_capture(installed)
            except BaseException as close_error:
                error += f"; cleanup {type(close_error).__name__}: {close_error}"
                details += "\nCleanup failure:\n" + traceback.format_exc()
        response = {
            "status": "error", "action": request.get("action"),
            "pid": os.getpid(), "run_id": request.get("run_id"),
            "error": error, "traceback": details,
        }
        try:
            _write_response(Path(response_path), response)
        except BaseException:
            _report_failure()


atexit.register(_close_at_exit)
