"""CPython 3.14+：给已运行的进程安装采集器，定时采集后卸载。"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .configuration import load_config


class AttachError(RuntimeError):
    """保留部分成功、卸载状态和未完成远程请求的路径。"""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__("; ".join(report["errors"]))


@dataclass
class _Request:
    pid: int
    action: str
    script: Path
    response: Path
    cancel: Path
    submitted: bool = False

    def read_response(self) -> dict[str, Any] | None:
        if not self.response.exists():
            return None
        return json.loads(self.response.read_text(encoding="utf-8"))

    def wait(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            response = self.read_response()
            if response is not None:
                return response
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"PID {self.pid} {self.action} 未在 {timeout:g} 秒内确认；"
                    "目标主线程可能尚未到达 Python 安全执行点"
                )
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))


def _bootstrap(request: Path, response: Path, cancel: Path) -> str:
    # 注入文件必须在目标真正读取前一直存在，不能用临时文件上下文过早删除。
    repository = str(Path(__file__).resolve().parents[2])
    return f'''def _call_stack_attach():
    import json
    import os
    import sys
    import traceback
    from pathlib import Path
    repository = {repository!r}
    inserted = repository not in sys.path
    if inserted:
        sys.path.insert(0, repository)
    try:
        from trace_python.call_stack_analysis.attach_target import dispatch
        dispatch({str(request)!r}, {str(response)!r}, {str(cancel)!r})
    except BaseException as exc:
        response = Path({str(response)!r})
        temporary = response.with_suffix('.tmp')
        temporary.write_text(json.dumps({{
            'status': 'error', 'pid': os.getpid(),
            'error': f'{{type(exc).__name__}}: {{exc}}',
            'traceback': traceback.format_exc(),
        }}, ensure_ascii=False), encoding='utf-8')
        temporary.replace(response)
    finally:
        if inserted:
            sys.path.remove(repository)
_call_stack_attach()
'''


def _make_request(directory: Path, payload: dict[str, Any]) -> _Request:
    stem = directory / f"{payload['action']}-{payload['pid']}"
    request = stem.with_suffix(".json")
    response = stem.with_suffix(".response")
    cancel = stem.with_suffix(".cancel")
    script = stem.with_suffix(".py")
    request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script.write_text(_bootstrap(request, response, cancel), encoding="utf-8")
    return _Request(payload["pid"], payload["action"], script, response, cancel)


def _submit(request: _Request) -> None:
    # 调用立即返回；只有目标写回确认才能证明采集器已经安装或卸载。
    request.submitted = True
    try:
        sys.remote_exec(request.pid, str(request.script))
    except Exception:
        request.submitted = False
        raise


def _check_response(response: dict[str, Any], request: _Request) -> None:
    if response["status"] != "ok":
        raise RuntimeError(
            f"PID {request.pid} {request.action}: "
            f"{response.get('error', response['status'])}"
        )


def _positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为有限正秒数")
    return number


def capture_processes(
    pids: list[int], config_path: Path, output_prefix: Path,
    duration: float = 10.0, timeout: float = 10.0,
) -> dict[str, Any]:
    """采集明确指定的 PID；异常时 AttachError.report 保留部分结果和未知状态。"""
    if sys.version_info < (3, 14) or not hasattr(sys, "remote_exec"):
        raise RuntimeError("attach 需要支持 sys.remote_exec 的 CPython 3.14+ 控制端")
    if not pids or any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
        raise ValueError("PID 列表必须非空、为正数且无重复")
    if os.getpid() in pids:
        raise ValueError("不能 attach 控制脚本自身")
    if any(not math.isfinite(value) or value <= 0 for value in (duration, timeout)):
        raise ValueError("duration 和 timeout 必须为有限正秒数")
    config_path, output_prefix = config_path.resolve(), output_prefix.resolve()
    with contextlib.redirect_stdout(sys.stderr):
        config = load_config(config_path)
    if not output_prefix.parent.is_dir():
        raise ValueError(f"输出目录不存在：{output_prefix.parent}")

    directory = Path(tempfile.mkdtemp(prefix=".call-stack-attach-", dir=output_prefix.parent))
    report: dict[str, Any] = {
        "run_id": uuid.uuid4().hex, "pids": list(pids),
        "captures": [], "errors": [], "control_directory": str(directory),
    }
    requests: list[_Request] = []
    starts: list[_Request] = []
    interrupted = False
    try:
        for pid in pids:
            request = _make_request(directory, {
                "action": "start", "pid": pid, "run_id": report["run_id"],
                "root": str(config.root), "targets": [asdict(target) for target in config.targets],
                "output_prefix": str(output_prefix), "config_path": str(config_path),
            })
            requests.append(request)
            starts.append(request)
            _submit(request)
            response = request.wait(timeout)
            _check_response(response, request)
        time.sleep(duration)
    except KeyboardInterrupt:
        interrupted = True
        report["errors"].append("采集被中断；正在请求卸载已安装采集器")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        for start in starts:
            # 先取消，再检查 ack：目标也在发 ack 后检查取消，封闭延迟执行的竞态。
            start.cancel.touch()
            response = start.read_response()
            if response is None or response.get("status") != "ok":
                continue
            capture = {**response, "stopped": False}
            report["captures"].append(capture)
            try:
                stop = _make_request(directory, {
                    "action": "stop", "pid": start.pid, "run_id": report["run_id"],
                })
                requests.append(stop)
                _submit(stop)
                _check_response(stop.wait(timeout), stop)
                capture["stopped"] = True
            except (Exception, KeyboardInterrupt) as exc:
                report["errors"].append(
                    f"PID {start.pid} 卸载未确认，采集器可能仍在运行：{type(exc).__name__}: {exc}"
                )
        pending = any(request.submitted and request.read_response() is None for request in requests)
        unclosed = any(not capture["stopped"] for capture in report["captures"])
        if not pending and not unclosed:
            shutil.rmtree(directory)
            report["control_directory"] = None
        else:
            report["errors"].append(
                f"保留远程控制文件以供延迟执行：{directory}；不要在请求确认前删除"
            )
    if report["errors"]:
        raise AttachError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pid", type=int, nargs="+", required=True, help="已运行的 Python PID，不递归追踪后代")
    parser.add_argument("--config", type=Path, required=True, help="可信 TraceConfig；仅加载 root/targets，不调用 run 或应用 env")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--duration", type=_positive_seconds, default=10.0, help="所有 PID 确认安装后采集秒数（默认 10）")
    parser.add_argument("--timeout", type=_positive_seconds, default=10.0, help="每次远程请求确认超时秒数（默认 10）")
    args = parser.parse_args(argv)
    try:
        report = capture_processes(args.pid, args.config, args.output_prefix, args.duration, args.timeout)
    except AttachError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2))
        print(str(exc), file=sys.stderr)
        return 130 if exc.report.get("interrupted") else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
