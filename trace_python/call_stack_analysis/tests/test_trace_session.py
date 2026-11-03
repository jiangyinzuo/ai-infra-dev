"""Public behavior only: direct CLI, library lifetime, and real child processes."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trace_python.call_stack_analysis import Target, TraceSession
from trace_python.call_stack_analysis.tests import multiprocess_smoke as smoke

TOOL_DIRECTORY = Path(smoke.__file__).resolve().parents[1]
METHODS = [method for method in ("fork", "spawn") if method in multiprocessing.get_all_start_methods()]
pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="Python 3.12 thread hooks required")


def _run(command: list[str], cwd: Path, *, module: bool = False) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["TRACE_SMOKE_ENV"] = "original"
    if module:
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    process = subprocess.Popen(
        command, cwd=cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        # 即使场景本身失去响应，也回收整个子进程组，而不是仅杀死 API 主进程。
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"Subprocess timed out: {command!r}\nstdout={stdout}\nstderr={stderr}")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _cli_config(root: Path, outcome: str) -> tuple[Path, Path]:
    root.mkdir()
    config = root / "scenario.py"
    restored = root / "restored.json"
    ending = {
        "ok": "return",
        "error": "raise RuntimeError('cli expected failure')",
        "interrupt": "raise KeyboardInterrupt('cli expected interrupt')",
    }[outcome]
    config.write_text(
        "import atexit\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "from trace_python.call_stack_analysis import Target, TraceConfig\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "def report_restored():\n"
        "    (ROOT / 'restored.json').write_text(json.dumps({\n"
        "        'cwd': str(Path.cwd()), 'env': os.environ.get('TRACE_SMOKE_ENV')\n"
        "    }), encoding='utf-8')\n"
        "atexit.register(report_restored)\n"
        "def observe(value):\n"
        "    assert Path.cwd() == ROOT\n"
        "    assert os.environ['TRACE_SMOKE_ENV'] == 'configured'\n"
        "    print(f'business:{value}', flush=True)\n"
        "def run(start):\n"
        "    observe('inactive')\n"
        "    start()\n"
        "    for value in ('first', 'second', 'third'):\n"
        "        observe(value)\n"
        f"    {ending}\n"
        "config = TraceConfig(root=ROOT, run=run, targets=(\n"
        "    Target('scenario.py', 'observe', ('value',), limit=2),\n"
        "), env={'TRACE_SMOKE_ENV': 'configured'})\n",
        encoding="utf-8",
    )
    return config, restored


def _cli_command(entrypoint: str) -> list[str]:
    match entrypoint:
        case "directory":
            return [sys.executable, str(TOOL_DIRECTORY)]
        case "file":
            return [sys.executable, str(TOOL_DIRECTORY / "__main__.py")]
        case "module":
            return [sys.executable, "-m", "trace_python.call_stack_analysis"]
        case _:
            raise ValueError(entrypoint)


@pytest.mark.parametrize("entrypoint", ["directory", "file", "module"])
def test_direct_cli_preserves_outputs_and_target_context(tmp_path: Path, entrypoint: str) -> None:
    config, restored = _cli_config(tmp_path / "business", "ok")
    invocation = tmp_path / "unrelated-cwd"
    invocation.mkdir()
    output = tmp_path / "outputs"
    output.mkdir()
    prefix = output / "capture"
    command = _cli_command(entrypoint) + ["--config", str(config), "--output-prefix", str(prefix)]
    result = _run(command, invocation, module=entrypoint == "module")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", result.stdout
    assert all(f"business:{value}" in result.stderr for value in ("inactive", "first", "second", "third"))
    assert {path.name for path in output.iterdir()} == {
        "capture.jsonl", "capture.txt", "capture.tree.json", "capture.tree.txt",
    }
    capture = smoke.read_capture(Path(str(prefix) + ".jsonl"))
    assert capture["run"]["argv"] == command
    assert capture["run"]["cwd"] == str(invocation)
    assert capture["run"]["root"] == str(config.parent)
    assert capture["run"]["config"] == str(config)
    assert capture["summary"]["counts"] == {"scenario.py:observe": 3}
    assert [event["values"] for event in capture["events"]] == [
        {"value": "first"}, {"value": "second"},
    ]
    assert all("run" in {frame["function"] for frame in event["stack"]}
               for event in capture["events"])
    assert json.loads(restored.read_text(encoding="utf-8")) == {
        "cwd": str(invocation), "env": "original",
    }


@pytest.mark.parametrize("outcome,exit_code,error", [
    ("error", 1, "RuntimeError: cli expected failure"),
    ("interrupt", 130, "KeyboardInterrupt: cli expected interrupt"),
])
def test_cli_errors_close_outputs_and_restore_context(
    tmp_path: Path, outcome: str, exit_code: int, error: str,
) -> None:
    config, restored = _cli_config(tmp_path / "business", outcome)
    invocation = tmp_path / "elsewhere"
    invocation.mkdir()
    prefix = tmp_path / "failure"
    result = _run(
        _cli_command("directory") + ["--config", str(config), "--output-prefix", str(prefix)],
        invocation,
    )
    assert result.returncode == exit_code, result.stderr
    assert result.stdout == "", result.stdout
    assert error in result.stderr
    capture = smoke.read_capture(Path(str(prefix) + ".jsonl"), expected_status="error")
    assert error in capture["summary"]["error"]
    assert capture["summary"]["counts"] == {"scenario.py:observe": 3}
    assert json.loads(restored.read_text(encoding="utf-8")) == {
        "cwd": str(invocation), "env": "original",
    }


@pytest.fixture
def business(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module_path = tmp_path / "smoke_business_functions.py"
    module_path.write_text(
        "class BusinessFailure(RuntimeError):\n"
        "    pass\n"
        "def observe(value):\n"
        "    return value\n"
        "def fail(error):\n"
        "    raise error\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "smoke_business_functions", raising=False)
    module = importlib.import_module("smoke_business_functions")
    yield module
    sys.modules.pop("smoke_business_functions", None)


def _session(business: Any, prefix: Path) -> TraceSession:
    return TraceSession(
        root=Path(business.__file__).parent,
        targets=(
            Target("smoke_business_functions.py", "observe", ("value",), limit=1),
            Target("smoke_business_functions.py", "fail", ("error",), limit=1),
        ),
        output_prefix=prefix,
    )




def test_inactive_capture_explicit_prefix_and_per_target_limit(tmp_path: Path, business: Any) -> None:
    session = _session(business, tmp_path / "unused")
    prefix = tmp_path / "explicit"
    with session.capture(active=False, output_prefix=prefix) as tracer:
        assert business.observe("inactive") == "inactive"
        tracer.start()
        for value in ("first", "second", "third"):
            assert business.observe(value) == value
    assert not list(tmp_path.glob("unused*"))
    capture = smoke.read_capture(Path(str(prefix) + ".jsonl"))
    assert [event["values"] for event in capture["events"]] == [{"value": "first"}]
    assert capture["summary"]["counts"] == {"smoke_business_functions.py:observe": 3}


def test_errors_reraise_same_exception_and_restore_all_hook_defaults(tmp_path: Path, business: Any) -> None:
    session = _session(business, tmp_path / "restoration")
    previous = (sys.gettrace(), sys.getprofile(), threading.gettrace(), threading.getprofile())

    def old_trace(frame: Any, event: str, argument: Any) -> Any:
        return old_trace

    def old_profile(frame: Any, event: str, argument: Any) -> None:
        return None

    try:
        sys.settrace(old_trace)
        sys.setprofile(old_profile)
        threading.settrace(old_trace)
        threading.setprofile(old_profile)
        expected = (old_trace, old_profile, old_trace, old_profile)
        error = business.BusinessFailure("preserve this exception")
        with pytest.raises(business.BusinessFailure) as caught:
            session.run(business.fail, error)
        assert caught.value is error
        assert (sys.gettrace(), sys.getprofile(), threading.gettrace(), threading.getprofile()) == expected
        with session.capture() as tracer:
            assert business.observe("after error") == "after error"
        assert (sys.gettrace(), sys.getprofile(), threading.gettrace(), threading.getprofile()) == expected
    finally:
        sys.settrace(previous[0])
        sys.setprofile(previous[1])
        threading.settrace(previous[2])
        threading.setprofile(previous[3])
    paths = list(tmp_path.glob("restoration.*.jsonl"))
    assert len(paths) == 2
    summaries = []
    for path in paths:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        status = raw[-1]["status"]
        capture = smoke.read_capture(path, expected_status=status)
        summaries.append(capture["summary"])
        assert capture["events"][0]["hit"] == 1
    assert {summary["status"] for summary in summaries} == {"ok", "error"}
    failure = next(summary for summary in summaries if summary["status"] == "error")
    assert "BusinessFailure: preserve this exception" in failure["error"]


@pytest.mark.parametrize("start_method", METHODS)
def test_child_failure_is_local_and_parent_capture_survives(
    tmp_path: Path, business: Any, start_method: str,
) -> None:
    prefix = tmp_path / "child-failure"
    session = _session(business, prefix)
    context = multiprocessing.get_context(start_method)
    process = context.Process(
        name="FailingBusinessChild", target=session.run,
        args=(business.fail, business.BusinessFailure("expected child failure")),
    )
    child_pid = None
    try:
        with session.capture():
            business.observe("parent-before-child")
            process.start()
            child_pid = process.pid
            process.join(30)
            assert not process.is_alive(), "Failing child did not exit"
            assert process.exitcode == 1
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            assert not process.is_alive(), "Could not reap failing child"
        process.close()
    paths = list(tmp_path.glob("child-failure.*.jsonl"))
    assert len(paths) == 2
    captures = []
    for path in paths:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        captures.append(smoke.read_capture(path, expected_status=raw[-1]["status"]))
    parent = next(capture for capture in captures if capture["run"]["pid"] == os.getpid())
    child = next(capture for capture in captures if capture["run"]["pid"] == child_pid)
    assert parent["summary"]["status"] == "ok"
    assert parent["summary"]["counts"] == {"smoke_business_functions.py:observe": 1}
    assert parent["events"][0]["values"] == {"value": "parent-before-child"}
    assert child["run"]["ppid"] == os.getpid()
    assert child["run"]["process_name"] == "FailingBusinessChild"
    assert child["summary"]["counts"] == {"smoke_business_functions.py:fail": 1}
    assert "BusinessFailure: expected child failure" in child["summary"]["error"]
    assert child["events"][0]["sequence"] == child["events"][0]["hit"] == 1
    assert {capture["run"]["run_id"] for capture in captures} == {session.run_id}
    assert len({capture["run"]["process_instance_id"] for capture in captures}) == 2


@pytest.mark.parametrize("start_method", METHODS)
def test_real_api_engine_worker_topology(tmp_path: Path, start_method: str) -> None:
    invocation = tmp_path / "arbitrary-cwd"
    invocation.mkdir()
    prefix = tmp_path / "artifacts" / "topology"
    result = _run([
        sys.executable, str(Path(smoke.__file__).resolve()),
        "--start-method", start_method, "--output-prefix", str(prefix),
    ], invocation)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    report = json.loads(result.stdout)
    assert report["results"] == [26, 36, 56]
    assert report["total"] == 118
    assert report["thread_results"] == {"existing": 34, "future": 38}
    assert report["artifacts_retained"] is True
    assert len(report["processes"]) == 4
    assert len({process["pid"] for process in report["processes"]}) == 4
    assert len(report["files"]) == 16
    assert all(Path(path).is_file() for path in report["files"])
    for path in prefix.parent.glob(prefix.name + ".*.jsonl"):
        capture = smoke.read_capture(path)
        assert capture["run"]["run_id"] == report["run_id"]
        assert capture["run"]["cwd"] == str(invocation)
