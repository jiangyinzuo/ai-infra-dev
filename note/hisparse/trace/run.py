#!/usr/bin/env python3
"""附加当前 P/D EngineCore 与 Proxy，持续请求直到采集器卸载；不启动或重配服务。"""

import argparse
import importlib
import json
import math
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/dev"))

serve = importlib.import_module("serve")
from trace_python.call_stack_analysis.attach import AttachError, capture_processes
from trace_python.call_stack_analysis.configuration import load_config
from trace_python.call_stack_analysis.tracing import locate

_REQUEST_TIMEOUT = 120
_METRICS = (
    "vllm:hisparse_cache_misses_total",
    "vllm:hisparse_host_to_device_bytes_total",
    "vllm:nixl_bytes_transferred_sum",
    "vllm:nixl_num_failed_transfers_total",
)


def _seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须为有限正秒数")
    return value


def _read_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def _engine_pid(parent: int) -> int:
    pending = [parent]
    engines = []
    while pending:
        pid = pending.pop()
        try:
            root = Path(f"/proc/{pid}")
            command = root.joinpath("cmdline").read_bytes().split(b"\0", 1)[0]
            if command.startswith(b"VLLM::EngineCore"):
                engines.append(pid)
            pending.extend(map(int, root.joinpath(f"task/{pid}/children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(engines) != 1:
        raise RuntimeError(f"服务 PID {parent} 下找到 {len(engines)} 个 EngineCore，需要恰好一个")
    return engines[0]


def _services() -> dict:
    state = json.loads((serve.state_directory("vllm") / "service.json").read_text())
    if not serve.running(state):
        raise RuntimeError("当前 vLLM 服务已退出")
    services = {
        row["name"]: row
        for row in serve.child_service_status(state["pid"], check_health=False)
    }
    for role in ("Prefill", "Decode", "Proxy"):
        if role not in services or not services[role]["alive"]:
            raise RuntimeError(f"找不到运行中的 {role} 服务")
    return services


def _metrics(address: str) -> dict[str, float]:
    with urlopen(address + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    return {
        name: sum(float(line.split()[-1]) for line in lines if line.startswith(name + "{"))
        for name in _METRICS
    }


def _load(address: str, model: str, stop: threading.Event, results: list, errors: list) -> None:
    try:
        while not stop.is_set():
            index = len(results)
            body = {
                "model": model,
                "prompt": [32000 + (i * 17 + index * 997 + 661) % 1000 for i in range(15000)],
                "max_tokens": 256,
                "ignore_eos": True,
                "temperature": 0,
            }
            request = Request(
                address + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.monotonic()
            with urlopen(request, timeout=_REQUEST_TIMEOUT) as response:
                result = json.load(response)
            row = {
                "request": index + 1,
                "id": result["id"],
                "usage": result.get("usage"),
                "elapsed_seconds": time.monotonic() - started,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
        print("请求失败：" + errors[-1], file=sys.stderr, flush=True)


def _refresh(pids: list[int], directory: Path, timeout: float, errors: list) -> None:
    # remote_exec 没有 __file__；路径必须由控制端传入。仅刷新已卸载的采集器。
    script = directory / "refresh.py"
    script.write_text(f'''
def refresh():
    import atexit
    import importlib
    import json
    import os
    import sys
    from pathlib import Path
    repository = {str(ROOT)!r}
    inserted = repository not in sys.path
    if inserted:
        sys.path.insert(0, repository)
    try:
        prefix = "trace_python.call_stack_analysis"
        target = importlib.import_module(prefix + ".attach_target")
        runtime = importlib.import_module(prefix + ".runtime")
        if target._active is not None or runtime._current is not None:
            raise RuntimeError("目标已有采集器，拒绝覆盖")
        for name in ("configuration", "call_tree", "output", "tracing", "runtime"):
            importlib.reload(importlib.import_module(prefix + "." + name))
        atexit.unregister(target._close_at_exit)
        importlib.reload(target)
        importlib.reload(importlib.import_module(prefix))
        result = {{"status": "ok", "pid": os.getpid()}}
    except Exception as error:
        result = {{"status": "error", "error": str(error)}}
    finally:
        if inserted:
            sys.path.remove(repository)
    Path({str(directory)!r}, f"refresh-{{os.getpid()}}.json").write_text(json.dumps(result))
refresh()
''', encoding="utf-8")
    for pid in pids:
        sys.remote_exec(pid, str(script))
    for pid in pids:
        response = directory / f"refresh-{pid}.json"
        deadline = time.monotonic() + timeout
        while not response.exists():
            if errors or time.monotonic() >= deadline:
                raise RuntimeError(f"PID {pid} 刷新未确认；保留 {script}")
            time.sleep(0.05)
        result = json.loads(response.read_text())
        if result["status"] != "ok":
            raise RuntimeError(f"PID {pid}: {result}")
    script.unlink()


def _late_detach(report: dict, timeout: float) -> dict:
    directory = report.get("control_directory")
    if directory is None:
        return {}
    directory = Path(directory)
    deadline = time.monotonic() + timeout
    responses = {}
    while True:
        pending = False
        for pid in report["pids"]:
            start = directory / f"start-{pid}.response"
            if not start.exists():
                pending = True
                continue
            status = json.loads(start.read_text()).get("status")
            if status != "ok":
                responses[str(pid)] = {"start_status": status}
                continue
            stop = directory / f"stop-{pid}.response"
            if stop.exists():
                responses[str(pid)] = json.loads(stop.read_text())
            if responses.get(str(pid), {}).get("status") != "ok":
                pending = True
        if not pending or time.monotonic() >= deadline:
            return responses
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=_seconds, default=60, help="采集秒数，默认 60")
    parser.add_argument("--timeout", type=_seconds, default=10, help="远程确认超时秒数，默认 10")
    args = parser.parse_args()
    if sys.version_info < (3, 14) or not hasattr(sys, "remote_exec"):
        parser.error("请使用 vLLM .venv 中的 CPython 3.14+")
    services = _services()
    model = _read_json(services["Decode"]["address"] + "/v1/models")["data"][0]["id"]
    roles = {
        "Prefill": _engine_pid(services["Prefill"]["pid"]),
        "Decode": _engine_pid(services["Decode"]["pid"]),
        "Proxy": services["Proxy"]["pid"],
    }
    labels = {
        roles["Prefill"]: "P",
        roles["Decode"]: "D",
        roles["Proxy"]: "Proxy",
    }
    pids = list(labels)
    config = ROOT / "trace_python/call_stack_analysis/config/pd_hisparse_attach.py"
    loaded = load_config(config)
    locate(loaded.root, loaded.targets)
    directory = Path(tempfile.mkdtemp(prefix=time.strftime("pd-trace-%Y%m%d-%H%M%S-"), dir="/tmp"))
    context = {
        "command": sys.orig_argv,
        "cwd": str(Path.cwd()),
        "roles": roles,
        "process_labels": labels,
        "model": model,
        "config": str(config),
        "duration": args.duration,
        "timeout": args.timeout,
        "input_tokens": 15000,
        "output_tokens": 256,
        "services": services,
    }
    (directory / "command.json").write_text(json.dumps(context, ensure_ascii=False, indent=2))
    print(
        f"P={roles['Prefill']} D={roles['Decode']} "
        f"Proxy={roles['Proxy']} model={model}\n输出：{directory}",
        flush=True,
    )
    before = _metrics(services["Decode"]["address"])
    stop = threading.Event()
    results, errors = [], []
    loader = threading.Thread(
        target=_load,
        args=(services["Proxy"]["address"], model, stop, results, errors),
    )
    loader.start()
    report = None
    try:
        _refresh(pids, directory, args.timeout, errors)
        try:
            report = capture_processes(
                pids, config, directory / "capture", args.duration, args.timeout,
                process_labels=labels,
            )
        except AttachError as error:
            report = error.report
            confirmations = _late_detach(report, args.timeout)
            (directory / "detach-confirmation.json").write_text(
                json.dumps(confirmations, indent=2)
            )
        finally:
            if report is not None:
                (directory / "attach-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        stop.set()
        loader.join()
        (directory / "requests.json").write_text(json.dumps({"requests": results, "errors": errors}, indent=2))
    if report is None or report["errors"] or errors:
        raise RuntimeError(f"采集未完整成功，请检查 {directory}")
    time.sleep(12)  # 采样统计异步发布；请求结束后再读最终 counter。
    after = _metrics(services["Decode"]["address"])
    delta = {name: after[name] - before[name] for name in _METRICS}
    (directory / "metrics.json").write_text(json.dumps({"before": before, "after": after, "delta": delta}, indent=2))
    for capture in report["captures"]:
        print("调用树：" + capture["output_prefix"] + ".tree.txt")
    print("采集器均已卸载；指标增量：" + json.dumps(delta), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
