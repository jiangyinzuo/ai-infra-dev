"""Offline process/thread scenario; run directly with Python 3.12 or newer."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

# 直接执行及 spawn 重新导入时，都不依赖调用者当前目录或 PYTHONPATH。
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trace_python.call_stack_analysis import Target, TraceSession

SOURCE = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
TIMEOUT = 25.0
REQUESTS = (5, 7, 11)


def checkpoint(role: str, value: int) -> int:
    return value


def count_only(role: str) -> str:
    return role


def thread_work(label: str, value: int) -> int:
    return value * 2


def _fork_bootstrap_probe() -> None:
    # 这发生在 Process.target 之前；继承的父 hook 不得写入任何进程的文件。
    checkpoint("inherited-bootstrap", -999)


def _identity() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "process_name": multiprocessing.current_process().name,
    }


def _receive(connection: Connection, label: str) -> dict[str, Any]:
    if not connection.poll(TIMEOUT):
        raise TimeoutError(f"Timed out waiting for {label}")
    try:
        message = connection.recv()
    except EOFError as exc:
        raise RuntimeError(f"{label} closed its response pipe") from exc
    if message.get("type") == "error":
        raise RuntimeError(f"{label}: {message['error']}")
    return message


def _join_successfully(processes: list[Any]) -> None:
    deadline = time.monotonic() + TIMEOUT
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            raise TimeoutError(f"{process.name} did not exit")
        if process.exitcode != 0:
            raise RuntimeError(f"{process.name} exited with code {process.exitcode}")


def _cleanup(processes: list[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        if process.is_alive():
            raise RuntimeError(f"Could not reap {process.name} (pid={process.pid})")
        process.close()


def _initial_work(role: str) -> None:
    for value in range(3):
        checkpoint(role, value)
        count_only(role)


class WorkerProc:
    @staticmethod
    def execute_request(worker_index: int, request_id: int, value: int) -> int:
        return value * (worker_index + 2) + worker_index

    @staticmethod
    def worker_main(worker_index: int, connection: Connection) -> None:
        try:
            _initial_work(f"worker-{worker_index}")
            connection.send({"type": "ready", "worker_index": worker_index, **_identity()})
            while True:
                message = _receive(connection, "engine request")
                if message["type"] == "shutdown":
                    connection.send({"type": "stopped", **_identity()})
                    return
                if message["type"] != "request":
                    raise ValueError(f"Unknown worker request: {message}")
                result = WorkerProc.execute_request(
                    worker_index, message["request_id"], message["value"]
                )
                connection.send({
                    "type": "result", "request_id": message["request_id"],
                    "worker_index": worker_index, "result": result,
                })
        finally:
            connection.close()


class EngineCoreProc:
    @staticmethod
    def execute_request(
        connections: list[Connection], request_id: int, value: int
    ) -> int:
        for connection in connections:
            connection.send({"type": "request", "request_id": request_id, "value": value})
        result = 0
        for index, connection in enumerate(connections):
            response = _receive(connection, f"worker-{index} response")
            assert response["type"] == "result", response
            assert response["request_id"] == request_id, response
            assert response["worker_index"] == index, response
            result += response["result"]
        return result

    @staticmethod
    def run_engine_core(
        session: TraceSession, connection: Connection, start_method: str,
        unused_api_connection: Connection,
    ) -> None:
        unused_api_connection.close()
        context = multiprocessing.get_context(start_method)
        processes: list[Any] = []
        connections: list[Connection] = []
        try:
            _initial_work("engine")
            for index in range(2):
                engine_end, worker_end = context.Pipe()
                connections.append(engine_end)
                process = context.Process(
                    name=f"WorkerProc-{index}", target=session.run,
                    args=(WorkerProc.worker_main, index, worker_end),
                )
                try:
                    process.start()
                finally:
                    worker_end.close()
                processes.append(process)
            workers = [
                _receive(worker, f"worker-{index} startup")
                for index, worker in enumerate(connections)
            ]
            assert all(worker["type"] == "ready" for worker in workers), workers
            connection.send({"type": "ready", "workers": workers, **_identity()})
            while True:
                message = _receive(connection, "API request")
                if message["type"] == "shutdown":
                    for worker in connections:
                        worker.send({"type": "shutdown"})
                    for index, worker in enumerate(connections):
                        response = _receive(worker, f"worker-{index} shutdown")
                        assert response["type"] == "stopped", response
                    _join_successfully(processes)
                    connection.send({"type": "stopped", **_identity()})
                    return
                if message["type"] != "request":
                    raise ValueError(f"Unknown engine request: {message}")
                result = EngineCoreProc.execute_request(
                    connections, message["request_id"], message["value"]
                )
                connection.send({
                    "type": "result", "request_id": message["request_id"],
                    "result": result,
                })
        except BaseException as exc:
            try:
                connection.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            raise
        finally:
            _cleanup(processes)
            for worker in connections:
                worker.close()
            connection.close()


class API:
    @staticmethod
    def submit_request(connection: Connection, request_id: int, value: int) -> int:
        connection.send({"type": "request", "request_id": request_id, "value": value})
        response = _receive(connection, "engine response")
        assert response["type"] == "result", response
        assert response["request_id"] == request_id, response
        return response["result"]


def _thread_entry(
    label: str, value: int, ready: threading.Event, release: threading.Event,
    results: dict[str, int], failures: list[BaseException],
) -> None:
    ready.set()
    try:
        if not release.wait(TIMEOUT):
            raise TimeoutError(f"{label} thread was not released")
        results[label] = thread_work(label, value)
    except BaseException as exc:
        failures.append(exc)


def targets() -> tuple[Target, ...]:
    return (
        Target(SOURCE, "checkpoint", ("role", "value"), limit=2),
        Target(SOURCE, "count_only", ("role",), limit=0),
        Target(SOURCE, "thread_work", ("label", "value"), limit=4),
        Target(SOURCE, "API.submit_request", ("request_id", "value"), limit=2),
        Target(SOURCE, "EngineCoreProc.run_engine_core", ("start_method",), limit=1),
        Target(SOURCE, "EngineCoreProc.execute_request", ("request_id", "value"), limit=2),
        Target(SOURCE, "WorkerProc.worker_main", ("worker_index",), limit=1),
        Target(
            SOURCE, "WorkerProc.execute_request",
            ("worker_index", "request_id", "value"), limit=2,
        ),
    )


def read_capture(jsonl: Path, expected_status: str = "ok") -> dict[str, Any]:
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert records and records[0]["type"] == "run", records
    assert records[-1]["type"] == "summary", records
    run, summary = records[0], records[-1]
    events = records[1:-1]
    assert all(event["type"] == "event" for event in events), records
    assert summary["status"] == expected_status, summary
    assert summary["events"] == len(events), summary
    assert isinstance(summary["elapsed_seconds"], (int, float)), summary
    assert summary["elapsed_seconds"] >= 0, summary
    if expected_status == "ok":
        assert summary["error"] is None, summary
    for field in ("started_at", "argv", "command", "cwd"):
        assert summary[field] == run[field], (field, run, summary)
    identity = ("run_id", "process_instance_id", "pid", "ppid", "process_name")
    for record in records:
        assert all(record[field] == run[field] for field in identity), record
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1)), events
    hits: Counter[str] = Counter()
    for event in events:
        hits[event["target"]] += 1
        assert event["hit"] == hits[event["target"]], event
        assert event["thread_id"] > 0 and event["native_thread_id"] > 0, event
        assert isinstance(event["monotonic_ns"], int) and event["monotonic_ns"] > 0, event
        assert isinstance(event["values"], dict), event
        assert event["stack"][-1]["function"] == event["target"].rsplit(":", 1)[1], event
        for frame in event["stack"]:
            assert frame["file"] and frame["function"] and frame["line"] > 0, frame
    assert all(summary["counts"][name] >= count for name, count in hits.items()), summary
    base = str(jsonl)[:-len(".jsonl")]
    paths = [Path(base + suffix) for suffix in (".jsonl", ".txt", ".tree.json", ".tree.txt")]
    tree = json.loads(paths[2].read_text(encoding="utf-8"))
    assert tree["summary"] == summary, tree
    assert isinstance(tree["tree"], dict), tree
    assert bool(tree["tree"]) == bool(events), tree
    return {"run": run, "events": events, "summary": summary, "paths": paths}


def _events(capture: dict[str, Any], function: str) -> list[dict[str, Any]]:
    return [event for event in capture["events"] if event["target"] == f"{SOURCE}:{function}"]


def _check_captures(
    prefix: Path, session: TraceSession, identities: list[dict[str, Any]], start_method: str
) -> list[dict[str, Any]]:
    paths = []
    for path in prefix.parent.iterdir():
        if not path.name.startswith(prefix.name + ".") or path.suffix != ".jsonl":
            continue
        with path.open(encoding="utf-8") as stream:
            first_record = json.loads(stream.readline())
        if first_record.get("run_id") == session.run_id:
            paths.append(path)
    paths.sort()
    assert len(paths) == 4, paths
    captures = [read_capture(path) for path in paths]
    assert {capture["run"]["pid"] for capture in captures} == {
        identity["pid"] for identity in identities
    }, captures
    assert len({capture["run"]["process_instance_id"] for capture in captures}) == 4, captures
    by_pid = {capture["run"]["pid"]: capture for capture in captures}
    for identity in identities:
        capture = by_pid[identity["pid"]]
        run, summary = capture["run"], capture["summary"]
        assert run["run_id"] == session.run_id, run
        assert all(run[field] == identity[field] for field in ("pid", "ppid", "process_name")), run
        role = identity["role"]
        common = _events(capture, "checkpoint")
        assert [event["values"] for event in common] == [
            {"role": role, "value": 0}, {"role": role, "value": 1},
        ], common
        expected_counts = {f"{SOURCE}:checkpoint": 3, f"{SOURCE}:count_only": 3}
        assert not _events(capture, "count_only"), capture
        if role == "api":
            expected_counts.update({f"{SOURCE}:API.submit_request": 3, f"{SOURCE}:thread_work": 2})
            business = _events(capture, "API.submit_request")
            assert not any("Proc." in event["target"] for event in capture["events"]), capture
            threads = _events(capture, "thread_work")
            assert {event["values"]["label"] for event in threads} == {"existing", "future"}, threads
            assert {event["values"]["value"] for event in threads} == {17, 19}, threads
            assert len({event["thread_id"] for event in threads}) == 2, threads
            assert len({event["native_thread_id"] for event in threads}) == 2, threads
            assert all(event["thread_id"] != common[0]["thread_id"] for event in threads), threads
            required_stack = "run_smoke"
        elif role == "engine":
            expected_counts.update({
                f"{SOURCE}:EngineCoreProc.run_engine_core": 1,
                f"{SOURCE}:EngineCoreProc.execute_request": 3,
            })
            entry = _events(capture, "EngineCoreProc.run_engine_core")
            assert entry[0]["values"] == {"start_method": start_method}, entry
            business = _events(capture, "EngineCoreProc.execute_request")
            required_stack = "EngineCoreProc.run_engine_core"
        else:
            expected_counts.update({
                f"{SOURCE}:WorkerProc.worker_main": 1,
                f"{SOURCE}:WorkerProc.execute_request": 3,
            })
            index = identity["worker_index"]
            entry = _events(capture, "WorkerProc.worker_main")
            assert entry[0]["values"] == {"worker_index": index}, entry
            business = _events(capture, "WorkerProc.execute_request")
            assert all(event["values"]["worker_index"] == index for event in business), business
            required_stack = "WorkerProc.worker_main"
        assert summary["counts"] == expected_counts, summary
        assert [event["values"]["value"] for event in business] == list(REQUESTS[:2]), business
        assert [event["values"]["request_id"] for event in business] == [0, 1], business
        for event in business:
            assert required_stack in {frame["function"] for frame in event["stack"]}, event
    return captures


def run_smoke(start_method: str, output_prefix: Path) -> dict[str, Any]:
    if sys.version_info < (3, 12):
        raise RuntimeError("The smoke requires Python 3.12 or newer")
    context = multiprocessing.get_context(start_method)
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    session = TraceSession(root=REPOSITORY_ROOT, targets=targets(), output_prefix=output_prefix)
    release = threading.Event()
    existing_ready, future_ready = threading.Event(), threading.Event()
    thread_results: dict[str, int] = {}
    thread_failures: list[BaseException] = []
    existing = threading.Thread(
        name="ExistingBusinessThread", target=_thread_entry,
        args=("existing", 17, existing_ready, release, thread_results, thread_failures),
    )
    future = threading.Thread(
        name="FutureBusinessThread", target=_thread_entry,
        args=("future", 19, future_ready, release, thread_results, thread_failures),
    )
    threads = [existing]
    processes: list[Any] = []
    api_end, engine_end = context.Pipe()
    identities = [{"role": "api", **_identity()}]
    results: list[int] = []
    existing.start()
    try:
        assert existing_ready.wait(TIMEOUT), "Existing thread did not start"
        if start_method == "fork":
            os.register_at_fork(after_in_child=_fork_bootstrap_probe)
        with session.capture(active=False) as tracer:
            checkpoint("inactive", -1)
            count_only("inactive")
            thread_work("inactive", -1)
            tracer.start()
            _initial_work("api")
            future.start()
            threads.append(future)
            assert future_ready.wait(TIMEOUT), "Future thread did not start"
            release.set()
            for thread in threads:
                thread.join(TIMEOUT)
                assert not thread.is_alive(), f"{thread.name} did not exit"
            assert not thread_failures, thread_failures
            assert thread_results == {"existing": 34, "future": 38}, thread_results
            engine = context.Process(
                name="EngineCoreProc", target=session.run,
                args=(EngineCoreProc.run_engine_core, session, engine_end, start_method, api_end),
            )
            try:
                engine.start()
            finally:
                engine_end.close()
            processes.append(engine)
            ready = _receive(api_end, "engine startup")
            assert ready["type"] == "ready", ready
            assert ready["process_name"] == "EngineCoreProc", ready
            identities.append({"role": "engine", **{
                key: ready[key] for key in ("pid", "ppid", "process_name")
            }})
            identities.extend({"role": f"worker-{worker['worker_index']}", **worker}
                              for worker in ready["workers"])
            # forkserver 的 OS 父进程是辅助服务，不是发起 Process 的业务进程。
            if start_method != "forkserver":
                assert ready["ppid"] == os.getpid(), ready
            assert len(ready["workers"]) == 2, ready
            if start_method != "forkserver":
                assert all(worker["ppid"] == ready["pid"] for worker in ready["workers"]), ready
            assert {worker["worker_index"] for worker in ready["workers"]} == {0, 1}, ready
            assert all(worker["process_name"] == f"WorkerProc-{worker['worker_index']}"
                       for worker in ready["workers"]), ready
            results = [API.submit_request(api_end, index, value) for index, value in enumerate(REQUESTS)]
            assert results == [5 * value + 1 for value in REQUESTS], results
            api_end.send({"type": "shutdown"})
            assert _receive(api_end, "engine shutdown")["type"] == "stopped"
            _join_successfully(processes)
    finally:
        release.set()
        for thread in threads:
            thread.join(TIMEOUT)
        # 先请求合作式退出，让 engine 有机会回收自己的 worker，再做强制兜底。
        if processes and processes[0].is_alive():
            try:
                api_end.send({"type": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            processes[0].join(TIMEOUT + 5.0)
        _cleanup(processes)
        api_end.close()
        engine_end.close()
    captures = _check_captures(output_prefix, session, identities, start_method)
    return {
        "start_method": start_method, "run_id": session.run_id,
        "processes": identities, "results": results, "total": sum(results),
        "thread_results": thread_results,
        "files": [str(path) for capture in captures for path in capture["paths"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--start-method", choices=multiprocessing.get_all_start_methods(), required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args(argv)
    if args.output_prefix is None:
        with tempfile.TemporaryDirectory(prefix="call-stack-process-smoke-") as directory:
            result = run_smoke(args.start_method, Path(directory) / "trace")
            result["artifacts_retained"] = False
    else:
        result = run_smoke(args.start_method, args.output_prefix)
        result["artifacts_retained"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
