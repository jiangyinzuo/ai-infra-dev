"""Real CPython 3.14 PID-attach smoke; no third-party dependencies or mocks."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SUBJECT = Path(__file__).with_name("attach_subject.py")
SOURCE = SUBJECT.relative_to(REPOSITORY_ROOT).as_posix()
TIMEOUT = 20.0
VALUES = [5, 7, 11]
FUNCTIONS = {"main_work": 2, "existing_work": 3, "future_work": 4}


class PermissionUnavailable(RuntimeError):
    """The runtime exists but this host forbids real remote attachment."""


def wait_for(check: Callable[[], Any], description: str, timeout: float = TIMEOUT) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        result = check()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {description}")
        threading.Event().wait(0.005)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require_runtime() -> None:
    if sys.version_info < (3, 14) or not hasattr(sys, "remote_exec"):
        raise RuntimeError("This smoke requires CPython 3.14+ with sys.remote_exec")


def permission_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, PermissionError) or any(
        term in text for term in ("operation not permitted", "permission denied", "ptrace access")
    )


class Subject:
    def __init__(self, directory: Path, *, disabled: bool = False, process_label: str | None = None) -> None:
        directory.mkdir(parents=True)
        self.directory = directory
        self.batches: list[str] = []
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHON_DISABLE_REMOTE_DEBUG", None)
        environment.pop("TRACE_PROCESS_LABEL", None)
        if process_label is not None:
            environment["TRACE_PROCESS_LABEL"] = process_label
        environment["ATTACH_SMOKE_ENV"] = "business-original"
        if disabled:
            environment["PYTHON_DISABLE_REMOTE_DEBUG"] = "1"
        self.process = subprocess.Popen(
            [sys.executable, str(SUBJECT), str(directory)], cwd=directory,
            env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            self.baseline = wait_for(
                lambda: self._ready(), f"subject {self.process.pid} startup",
            )
            assert self.baseline["pid"] == self.process.pid
            assert self.baseline["hooks"] == {"current": False, "future": False}
            assert not self.baseline["tracing_imported"], self.baseline
        except BaseException:
            self.close(check=False)
            raise

    def _ready(self) -> dict[str, Any] | None:
        self.check_alive()
        return read_json(self.directory / "ready.json")

    def check_alive(self) -> None:
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise AssertionError(f"Subject exited {self.process.returncode}: {stdout}\n{stderr}")

    def batch(
        self, name: str, values: list[int] | None = None,
        *, process_label: str | None = None, process_name: str | None = None,
    ) -> dict[str, Any]:
        self.check_alive()
        response_path = self.directory / "response.json"
        response_path.unlink(missing_ok=True)
        request = {"action": "batch", "batch": name, "values": VALUES if values is None else values}
        if process_name is not None:
            request["process_name"] = process_name
        temporary = self.directory / "request.tmp"
        temporary.write_text(json.dumps(request), encoding="utf-8")
        temporary.replace(self.directory / "request.json")
        self.batches.append(name)
        response = wait_for(lambda: self._response(response_path), f"business batch {name}")
        assert response["batch"] == name, response
        expected_environment = self.baseline["environment"].copy()
        if process_label is not None:
            expected_environment["TRACE_PROCESS_LABEL"] = process_label
        assert response["environment"] == expected_environment, response
        for key in ("cwd", "sys_path"):
            assert response[key] == self.baseline[key], (key, response, self.baseline)
        expected = request["values"]
        assert response["values"] == [value * 2 for value in expected], response
        assert response["existing"]["values"] == [value * 3 for value in expected], response
        assert response["future"]["values"] == [value * 4 for value in expected], response
        assert response["count_only"] == expected, response
        return response

    def _response(self, path: Path) -> dict[str, Any] | None:
        self.check_alive()
        return read_json(path)

    def assert_detached(self) -> None:
        response = self.batch(f"detached-{len(self.batches)}", [])
        for state in (response, response["existing"], response["future"]):
            assert state["hooks"] == self.baseline["hooks"], response

    def close(self, *, check: bool = True) -> None:
        process = self.process
        try:
            if process.poll() is None and check:
                response = self.directory / "response.json"
                response.unlink(missing_ok=True)
                temporary = self.directory / "request.tmp"
                temporary.write_text(json.dumps({"action": "quit"}), encoding="utf-8")
                temporary.replace(self.directory / "request.json")
            elif process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=TIMEOUT)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=TIMEOUT)
            raise
        if check:
            assert process.returncode == 0, (process.returncode, stdout, stderr)
            assert stderr == "", stderr
            expected = ["business:ready", *(f"business:{name}" for name in self.batches)]
            assert stdout.splitlines() == expected, (stdout, expected)


def make_config(path: Path, *, invalid: bool = False, process_label: str | None = None) -> Path:
    function = "does_not_exist" if invalid else "main_work"
    specifications = [(function, 2), ("existing_work", 2), ("future_work", 2), ("count_only", 0)]
    targets = ",\n".join(
        f"    Target({SOURCE!r}, {name!r}, {('value',) if name == 'count_only' else ('value', 'batch')!r}, limit={limit}, caller_fields=('command',))"
        for name, limit in specifications
    )
    path.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "from trace_python.call_stack_analysis import Target, TraceConfig\n"
        f"assert os.getpid() == {os.getpid()}, 'config imported outside controller'\n"
        "def run(start):\n"
        "    raise AssertionError('attach must not call config.run')\n"
        f"config = TraceConfig(root=Path({str(REPOSITORY_ROOT)!r}), run=run, targets=(\n{targets},\n), process_label={process_label!r})\n",
        encoding="utf-8",
    )
    return path


def run_record(prefix: Path, pid: int) -> dict[str, Any] | None:
    for path in prefix.parent.glob(prefix.name + ".*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            first = stream.readline()
        if first.endswith("\n"):
            record = json.loads(first)
            if record["type"] == "run" and record["pid"] == pid:
                return record
    return None


def capture_batch(
    subjects: list[Subject], config: Path, prefix: Path, duration: float,
    *, cli: bool = False, process_labels: dict[int, str] | None = None,
    final_process_name: str | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    from trace_python.call_stack_analysis.attach import AttachError, capture_processes, main as attach_main

    abort = threading.Event()
    failures: list[BaseException] = []
    responses: dict[int, dict[str, Any]] = {}
    original_cwd, original_environment = Path.cwd(), dict(os.environ)

    def drive() -> None:
        try:
            for subject in subjects:
                pid = subject.process.pid
                record = wait_for(
                    lambda: abort.is_set() or run_record(prefix, pid),
                    f"run record for {pid}",
                )
                if abort.is_set():
                    return
                assert isinstance(record, dict), record
                responses[pid] = subject.batch(
                    f"capture-{prefix.name}", process_label=record["process_label"],
                )
                if final_process_name is not None:
                    # All target limits are exhausted; identity must still follow the rename.
                    subject.batch(
                        f"exhausted-{prefix.name}", process_label=record["process_label"],
                        process_name="EngineCore-matched",
                    )
                    # No target executes after this rename: only stop can refresh the final path.
                    subject.batch(
                        f"rename-{prefix.name}", [], process_label=record["process_label"],
                        process_name=final_process_name,
                    )
        except BaseException as exc:
            failures.append(exc)

    driver = threading.Thread(target=drive, name="attach-business-driver")
    driver.start()
    try:
        pids = [subject.process.pid for subject in subjects]
        if cli:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = attach_main([
                    "--pid", *(str(pid) for pid in pids), "--config", str(config),
                    "--output-prefix", str(prefix), "--duration", str(duration),
                    "--timeout", str(TIMEOUT),
                    *(argument for pid, label in (process_labels or {}).items()
                      for argument in ("--process-label", f"{pid}={label}")),
                ])
            report = json.loads(stdout.getvalue())
            if code != 0:
                raise AttachError(report)
            assert stderr.getvalue() == "", stderr.getvalue()
        else:
            report = capture_processes(
                pids, config, prefix, duration, TIMEOUT, process_labels=process_labels,
            )
    except BaseException as exc:
        if permission_error(exc):
            raise PermissionUnavailable(str(exc)) from exc
        raise
    finally:
        abort.set()
        driver.join(TIMEOUT + 2)
        if driver.is_alive():
            raise TimeoutError("request driver did not finish")
        assert Path.cwd() == original_cwd
        assert dict(os.environ) == original_environment
    if failures:
        raise failures[0]
    assert set(responses) == {subject.process.pid for subject in subjects}, responses
    return report, responses


def check_capture(
    capture: dict[str, Any], report: dict[str, Any], response: dict[str, Any],
    *, calls_per_target: int = len(VALUES),
) -> dict[str, Any]:
    prefix = Path(capture["output_prefix"])
    paths = [Path(str(prefix) + suffix)
             for suffix in (".jsonl", ".txt", ".tree.json", ".tree.txt")]
    assert all(path.is_file() for path in paths), paths
    records = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    run, *events, summary = records
    assert run["type"] == "run" and summary["type"] == "summary", records
    assert summary["status"] == "ok" and summary["error"] is None, summary
    assert summary["events"] == len(events) == 6, summary
    assert summary["elapsed_seconds"] >= 0, summary
    assert capture["stopped"] is True, capture
    for record in records:
        assert record["pid"] == capture["pid"], record
        assert record["run_id"] == report["run_id"], record
        assert record["process_instance_id"] == capture["process_instance_id"], record
        assert record["process_label"] == capture["process_label"], record
    assert run["process_name"] == response["process_name"], run
    assert all(event["process_name"] == run["process_name"] for event in events), events
    assert summary["process_name"] == capture["process_name"], summary
    assert summary["output_prefix"] == str(prefix), summary
    if run["output_prefix"] != str(prefix):
        assert not any(Path(run["output_prefix"] + suffix).exists()
                       for suffix in (".jsonl", ".txt", ".tree.json", ".tree.txt")), run
    assert [event["sequence"] for event in events] == list(range(1, 7)), events
    expected_counts = {f"{SOURCE}:{name}": calls_per_target for name in (*FUNCTIONS, "count_only")}
    assert summary["counts"] == expected_counts, summary
    for name in FUNCTIONS:
        selected = [event for event in events if event["target"] == f"{SOURCE}:{name}"]
        assert [event["hit"] for event in selected] == [1, 2], selected
        assert [event["values"]["value"] for event in selected] == VALUES[:2], selected
        assert all(event["values"]["batch"] == response["batch"] for event in selected), selected
        for event in selected:
            caller_command = dict(event["caller_values"]["command"]["items"])
            assert caller_command["batch"] == response["batch"], event
            assert caller_command["action"] == "batch", event
        state = response if name == "main_work" else response[name.split("_")[0]]
        assert state["hooks"] == {"current": True, "future": True}, state
        for event in selected:
            assert event["type"] == "event" and event["thread_id"] == state["thread_id"], event
            stack = event["stack"]
            assert stack[-1]["function"] == name, stack
            caller = {"main_work": "main", "existing_work": "existing_loop", "future_work": "run_future"}[name]
            assert any(frame["function"] == caller for frame in stack[:-1]), stack
            assert all(frame["file"] and frame["line"] > 0 for frame in stack), stack
            assert event["native_thread_id"] > 0 and event["monotonic_ns"] > 0, event
    assert len({event["thread_id"] for event in events}) == 3, events
    tree = json.loads(paths[2].read_text(encoding="utf-8"))
    assert tree["summary"] == summary and tree["tree"], tree
    # A detached subject must not append even after more business calls.
    return {"prefix": str(prefix), "sizes": {str(path): path.stat().st_size for path in paths}}


def successful_scenario(prefix: Path, duration: float = 0.5) -> dict[str, Any]:
    require_runtime()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=prefix.name + "-subjects-", dir=prefix.parent))
    subjects: list[Subject] = []
    reports: list[dict[str, Any]] = []
    try:
        config = make_config(work / "config.py", process_label="configured")
        subjects.append(Subject(work / "first"))
        process_labels = {subjects[0].process.pid: "P"}
        for label in ("first", "reattach"):
            report, responses = capture_batch(
                subjects, config, Path(str(prefix) + "-" + label), duration, cli=label == "first",
                process_labels=process_labels, final_process_name="EngineCore",
            )
            assert report["pids"] == [subjects[0].process.pid], report
            assert len(report["captures"]) == 1, report
            checked = check_capture(
                report["captures"][0], report, responses[subjects[0].process.pid],
                calls_per_target=2 * len(VALUES),
            )
            assert report["captures"][0]["process_label"] == "P", report
            assert report["captures"][0]["process_name"] == "EngineCore", report
            subjects[0].assert_detached()
            subjects[0].batch(f"after-{label}")
            assert all(Path(path).stat().st_size == size for path, size in checked["sizes"].items())
            reports.append(report)
        first, second = (report["captures"][0] for report in reports)
        assert first["process_instance_id"] != second["process_instance_id"], reports
        assert reports[0]["run_id"] != reports[1]["run_id"], reports
        subjects.append(Subject(work / "second"))
        subjects[1].batch("set-name", [], process_name="EngineCore")
        process_labels[subjects[1].process.pid] = "D"
        report, responses = capture_batch(
            subjects, config, Path(str(prefix) + "-multi"), duration, cli=True,
            process_labels=process_labels, final_process_name="EngineCore-final",
        )
        assert report["pids"] == [subject.process.pid for subject in subjects], report
        assert {capture["pid"] for capture in report["captures"]} == set(report["pids"]), report
        assert len({capture["process_instance_id"] for capture in report["captures"]}) == 2, report
        for capture in report["captures"]:
            check_capture(capture, report, responses[capture["pid"]], calls_per_target=2 * len(VALUES))
            assert capture["process_label"] == process_labels[capture["pid"]], capture
            assert capture["process_name"] == "EngineCore-final", capture
            assert f".{capture['process_label']}.EngineCore-final." in Path(capture["output_prefix"]).name, capture
        assert len({capture["output_prefix"] for capture in report["captures"]}) == 2, report
        reports.append(report)
        for subject in subjects:
            subject.assert_detached()
        return {"reports": reports, "work_directory": str(work)}
    finally:
        active_error = sys.exc_info()[0] is not None
        for subject in subjects:
            subject.close(check=not active_error)


def process_label_scenario(prefix: Path) -> dict[str, Any]:
    require_runtime()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=prefix.name + "-labels-", dir=prefix.parent))
    subject = Subject(work / "subject", process_label="target-environment")
    reports: list[dict[str, Any]] = []
    try:
        for configured in (None, "configured"):
            config = make_config(work / "config.py", process_label=configured)
            report, responses = capture_batch(
                [subject], config, Path(f"{prefix}-{len(reports)}"), 0.5,
            )
            capture, = report["captures"]
            assert capture["process_label"] == (configured or "target-environment"), capture
            check_capture(capture, report, responses[subject.process.pid])
            subject.assert_detached()
            reports.append(report)
        return {"reports": reports}
    finally:
        subject.close(check=sys.exc_info()[0] is None)


def invalid_target_scenario(prefix: Path) -> dict[str, Any]:
    require_runtime()
    from trace_python.call_stack_analysis.attach import AttachError, capture_processes

    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=prefix.name + "-invalid-", dir=prefix.parent))
    subject = Subject(work / "subject")
    try:
        config = make_config(work / "invalid.py", invalid=True)
        try:
            capture_processes([subject.process.pid], config, prefix, 0.01, TIMEOUT)
        except AttachError as exc:
            if permission_error(exc):
                raise PermissionUnavailable(str(exc)) from exc
            report = exc.report
            assert report["errors"], report
            assert "does_not_exist" in str(report["errors"]), report
        else:
            raise AssertionError("invalid target was accepted")
        subject.assert_detached()
        subject.batch("still-alive")
        return report
    finally:
        subject.close(check=sys.exc_info()[0] is None)


def timeout_scenario(prefix: Path) -> dict[str, Any]:
    require_runtime()
    from trace_python.call_stack_analysis.attach import AttachError, capture_processes

    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=prefix.name + "-timeout-", dir=prefix.parent))
    subject = Subject(work / "subject")
    try:
        config = make_config(work / "config.py")
        try:
            capture_processes([subject.process.pid], config, prefix, 0.01, 1e-9)
        except AttachError as exc:
            if permission_error(exc):
                raise PermissionUnavailable(str(exc)) from exc
            report = exc.report
            assert report["errors"], report
        else:
            raise AssertionError("nanosecond timeout unexpectedly acknowledged attachment")
        control = report.get("control_directory")
        if control is not None:
            directory = Path(control)
            assert directory.is_dir(), report
            assert (directory / f"start-{subject.process.pid}.cancel").exists(), report

            pending_start = not any(capture["pid"] == subject.process.pid for capture in report["captures"])

            def start_replied() -> dict[str, Any] | None:
                for path in directory.glob("*.response"):
                    message = read_json(path)
                    if not message or message.get("action") != "start":
                        continue
                    if pending_start and message.get("status") == "ok":
                        continue
                    if message.get("status") in ("ok", "cancelled", "error"):
                        return message
                return None

            acknowledgement = wait_for(start_replied, "late start cancellation acknowledgement")
            assert acknowledgement["status"] == ("cancelled" if pending_start else "ok"), acknowledgement
        # Once the late start has replied, no subsequent business thread may inherit a hook.
        wait_for(
            lambda: not subject.batch(f"timeout-probe-{len(subject.batches)}", [])["hooks"]["current"],
            "timed-out capture cleanup",
        )
        subject.assert_detached()
        subject.batch("after-timeout")
        return report
    finally:
        subject.close(check=sys.exc_info()[0] is None)


def disabled_scenario(prefix: Path) -> dict[str, Any]:
    require_runtime()
    from trace_python.call_stack_analysis.attach import AttachError, capture_processes

    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=prefix.name + "-disabled-", dir=prefix.parent))
    subject = Subject(work / "subject", disabled=True)
    try:
        config = make_config(work / "config.py")
        try:
            capture_processes([subject.process.pid], config, prefix, 0.01, 1.0)
        except AttachError as exc:
            report = exc.report
            assert report["errors"], report
        else:
            raise AssertionError("target disabled remote debugging but accepted attachment")
        subject.assert_detached()
        subject.batch("disabled-still-alive")
        assert not list(prefix.parent.glob(prefix.name + ".*.jsonl")), report
        return report
    finally:
        subject.close(check=sys.exc_info()[0] is None)


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, help="retain outputs under this prefix")
    parser.add_argument("--duration", type=positive_float, default=0.5)
    options = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="attach-smoke-") if options.output_prefix is None else None
    prefix = options.output_prefix.resolve() if options.output_prefix else Path(temporary.name) / "capture"
    try:
        result = successful_scenario(prefix, options.duration)
        result["process_labels"] = process_label_scenario(Path(str(prefix) + "-labels"))
        result["invalid_target"] = invalid_target_scenario(Path(str(prefix) + "-invalid"))
        result["timeout"] = timeout_scenario(Path(str(prefix) + "-timeout"))
        result["disabled"] = disabled_scenario(Path(str(prefix) + "-disabled"))
        print(json.dumps({"status": "ok", **result}))
        return 0
    except PermissionUnavailable as exc:
        print(f"SKIP: real sys.remote_exec unavailable: {exc}", file=sys.stderr)
        return 77
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
