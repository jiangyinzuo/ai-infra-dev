"""Uninstrumented subject for attach smoke tests; only Python standard library."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def hook_state() -> dict[str, bool]:
    return {"current": sys.gettrace() is not None, "future": threading.gettrace() is not None}


def main_work(value: int, batch: str) -> int:
    return value * 2


def existing_work(value: int, batch: str) -> int:
    return value * 3


def future_work(value: int, batch: str) -> int:
    return value * 4


def count_only(value: int) -> int:
    return value


def existing_loop(tasks: queue.Queue[Any], ready: threading.Event) -> None:
    ready.set()
    while True:
        task = tasks.get()
        if task is None:
            return
        command, result, done = task
        result["hooks"] = hook_state()
        result["thread_id"] = threading.get_ident()
        result["values"] = [existing_work(value, command["batch"]) for value in command["values"]]
        done.set()


def run_future(command: dict[str, Any], result: dict[str, Any]) -> None:
    result["hooks"] = hook_state()
    result["thread_id"] = threading.get_ident()
    result["values"] = [future_work(value, command["batch"]) for value in command["values"]]


def main() -> None:
    directory = Path(sys.argv[1])
    tasks: queue.Queue[Any] = queue.Queue()
    ready = threading.Event()
    existing = threading.Thread(target=existing_loop, args=(tasks, ready), name="existing")
    existing.start()
    if not ready.wait(10):
        raise TimeoutError("existing thread did not start")
    original_path = list(sys.path)
    baseline = {
        "pid": os.getpid(), "cwd": str(Path.cwd()), "environment": dict(os.environ),
        "sys_path": original_path, "hooks": hook_state(),
        "tracing_imported": any(name.startswith("trace_python") for name in sys.modules),
    }
    atomic_json(directory / "ready.json", baseline)
    print("business:ready", flush=True)
    try:
        while True:
            request = directory / "request.json"
            if not request.exists():
                # Polling also lets the main interpreter service remote-exec requests.
                time.sleep(0.005)
                continue
            command = json.loads(request.read_text(encoding="utf-8"))
            request.unlink()
            if command["action"] == "quit":
                atomic_json(directory / "response.json", {"stopped": True})
                return
            existing_result: dict[str, Any] = {}
            future_result: dict[str, Any] = {}
            done = threading.Event()
            tasks.put((command, existing_result, done))
            future = threading.Thread(target=run_future, args=(command, future_result), name="future")
            future.start()
            result = {
                "batch": command["batch"], "hooks": hook_state(),
                "thread_id": threading.get_ident(),
                "values": [main_work(value, command["batch"]) for value in command["values"]],
                "count_only": [count_only(value) for value in command["values"]],
            }
            future.join(10)
            if future.is_alive() or not done.wait(10):
                raise TimeoutError("business thread did not finish")
            result.update({
                "existing": existing_result, "future": future_result,
                "cwd": str(Path.cwd()), "environment": dict(os.environ),
                "sys_path": list(sys.path),
            })
            print(f"business:{command['batch']}", flush=True)
            atomic_json(directory / "response.json", result)
    finally:
        tasks.put(None)
        existing.join(10)


if __name__ == "__main__":
    main()
