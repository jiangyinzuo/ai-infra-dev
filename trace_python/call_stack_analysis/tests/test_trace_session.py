"""Public behavior only: direct CLI, library lifetime, and real child processes."""

from __future__ import annotations

import ast
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
        "), env={'TRACE_SMOKE_ENV': 'configured'}, process_label='CLI')\n",
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
    assert capture["run"]["process_label"] == "CLI"
    assert all(event["process_label"] == "CLI" for event in capture["events"])
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


def test_tree_values_stay_under_target_and_reset_between_captures(tmp_path: Path) -> None:
    """变量叶节点不能破坏兄弟树线，展示限次不应丢失原始事件。"""
    source = tmp_path / "tree_business.py"
    source.write_text(
        'def leaf(value):\n    return value\n'
        'def sibling():\n    return "sibling"\n'
        'def root(value):\n    leaf(value)\n    sibling()\n',
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("tree_business", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = TraceSession(
        root=tmp_path,
        targets=(
            Target(source.name, "root", ("value",), limit=5),
            Target(source.name, "leaf", ("value",), limit=5),
            Target(source.name, "sibling", limit=5),
        ),
        output_prefix=tmp_path / "unused",
    )
    for run in range(2):
        prefix = tmp_path / f"capture-{run}"
        with session.capture(output_prefix=prefix):
            for index in range(5):
                module.root(f"run-{run}:值-{index}\nnext")
        capture = smoke.read_capture(Path(str(prefix) + ".jsonl"))
        assert len(capture["events"]) == 15
        assert sum(event["values"] == {"value": f"run-{run}:值-4\nnext"}
                   for event in capture["events"]) == 2
        lines = Path(str(prefix) + ".tree.txt").read_text().splitlines()
        displayed = [line.split("value = ", 1)[1] for line in lines if "─value = " in line]
        assert [ast.literal_eval(value) for value in displayed] == [
            f"run-{run}:值-{index}\nnext"
            for _ in range(2) for index in range(2)
        ]
        assert sum("另有 3 次变量快照未展示" in line for line in lines) == 2
        for function, last_child in (("root", False), ("leaf", True)):
            position = next(i for i, line in enumerate(lines) if f"─{function} 🎯" in line)
            node_line = lines[position]
            branch_pos = node_line.index(f"─{function}") - 1
            child_prefix = node_line[:branch_pos] + (
                "  " if node_line[branch_pos] == "└" else "│ "
            )
            assert lines[position + 1].startswith(child_prefix + "├─🔎[")
            assert lines[position + 2].startswith(child_prefix + "│ └─value = ")
            assert lines[position + 3].startswith(child_prefix + "├─🔎[")
            assert lines[position + 4].startswith(child_prefix + "│ └─value = ")
            last_branch = "└─" if last_child else "├─"
            assert lines[position + 5].startswith(child_prefix + last_branch + "🔎 ")
        sibling = next(i for i, line in enumerate(lines) if "─sibling 🎯" in line)
        assert "🔎" not in lines[sibling + 1]


@pytest.mark.parametrize("fields", [(), ("value",)])
def test_caller_fields_use_direct_frame_and_share_tree_limit(tmp_path: Path, fields: tuple[str, ...]) -> None:
    """同名变量分属两层；不跳过 wrapper，也不能读取调用后才赋值的变量。"""
    source = tmp_path / "caller_business.py"
    source.write_text(
        'def observe(value):\n    return value\n'
        'class Caller:\n'
        '    role = "decode"\n'
        '    def call(self, index):\n'
        '        value = "outer"\n'
        '        return self.wrapper(index)\n'
        '    def wrapper(self, index):\n'
        '        value = f"caller-{index}"\n'
        '        result = observe(f"callee-{index}")\n'
        '        later = "after-call"\n'
        '        return result\n',
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("caller_business", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = TraceSession(
        root=tmp_path,
        targets=(Target(source.name, "observe", fields, limit=3,
                        caller_fields=("value", "index", "self.role", "later")),),
        output_prefix=tmp_path / "unused",
    )
    for run in range(2):
        prefix = tmp_path / f"caller-{run}"
        with session.capture(output_prefix=prefix):
            for index in range(4):
                assert module.Caller().call(index) == f"callee-{index}"
        capture = smoke.read_capture(Path(str(prefix) + ".jsonl"))
        assert capture["summary"]["counts"] == {f"{source.name}:observe": 4}
        for index, event in enumerate(capture["events"]):
            assert event["values"] == ({"value": f"callee-{index}"} if fields else {})
            assert event["caller_values"] == {
                "value": f"caller-{index}", "index": index, "self.role": "decode",
                "later": {"unavailable": "KeyError"},
            }
            assert event["stack"][-2]["function"] == "Caller.wrapper"
        assert len(capture["events"]) == 3
        text = Path(str(prefix) + ".txt").read_text()
        assert "👆 caller=Caller.wrapper" in text and "value = 'caller-2'" in text
        tree = Path(str(prefix) + ".tree.txt").read_text().splitlines()
        position = next(i for i, line in enumerate(tree) if "─observe 🎯" in line)
        branch = tree[position].index("─observe") - 1
        child_prefix = tree[position][:branch] + "  "
        caller_rows = [i for i, line in enumerate(tree) if line.startswith(child_prefix + "├─👆[")]
        assert len(caller_rows) == 2
        for row in caller_rows:
            assert tree[row].endswith("caller=Caller.wrapper")
            assert tree[row + 1].startswith(child_prefix + "│ ├─value = 'caller-")
            assert tree[row + 4] == child_prefix + "│ └─later = <unavailable: KeyError>"
        local_rows = [line for line in tree if line.startswith(child_prefix + "├─🔎[")]
        assert len(local_rows) == (2 if fields else 0)
        assert child_prefix + "└─🔎 另有 1 次变量快照未展示" in tree
        assert not any("value = 'caller-2'" in line for line in tree)


def test_invalid_caller_expression_is_rejected_before_business_call(tmp_path: Path, business: Any) -> None:
    session = TraceSession(
        root=tmp_path,
        targets=(Target("smoke_business_functions.py", "observe", caller_fields=("len(value)",)),),
        output_prefix=tmp_path / "invalid",
    )
    with pytest.raises(ValueError, match="字段路径"):
        with session.capture():
            pytest.fail("invalid caller expression reached business code")


@pytest.mark.parametrize("value", [
    None, True, False, "中文\nnext", [], (), (7,), {}, set(), frozenset(),
    [1, False, None], ("a", 2), {1, 2}, frozenset({"a", "b"}),
    {1: "integer", "1": "string", (2, 3): True, None: False},
    {"nested": [1, False], "type": "dict", "items": {}},
])
def test_text_values_preserve_python_container_semantics(tmp_path: Path, business: Any, value: Any) -> None:
    """人类输出可直接读作 Python 值，不丢失容器或字典 key 的类型。"""
    prefix = tmp_path / "python-values"
    with _session(business, prefix).capture(output_prefix=prefix):
        business.observe(value)
    for suffix in (".txt", ".tree.txt"):
        lines = Path(str(prefix) + suffix).read_text().splitlines()
        rendered, = [line.split("value = ", 1)[1] for line in lines if "value = " in line]
        parsed = ast.parse(rendered, mode="eval").body
        if isinstance(value, frozenset):
            assert isinstance(parsed, ast.Call) and parsed.func.id == "frozenset"
            actual = frozenset(ast.literal_eval(parsed.args[0])) if parsed.args else frozenset()
        else:
            actual = ast.literal_eval(parsed)
        assert actual == value and type(actual) is type(value)


@pytest.mark.parametrize("value", [list(range(20)), tuple(range(20)), set(range(20)),
                                   dict.fromkeys(range(20), True)])
def test_text_values_bound_large_containers(tmp_path: Path, business: Any, value: Any) -> None:
    """只展示有界前缀，并明确告知容器总长度；JSONL 保留结构化快照。"""
    prefix = tmp_path / "bounded-values"
    with _session(business, prefix).capture(output_prefix=prefix):
        business.observe(value)
    capture = smoke.read_capture(Path(str(prefix) + ".jsonl"))
    snapshot = capture["events"][0]["values"]["value"]
    assert snapshot["length"] == 20 and len(snapshot["items"]) == 16
    for suffix in (".txt", ".tree.txt"):
        lines = Path(str(prefix) + suffix).read_text().splitlines()
        rendered, = [line.split("value = ", 1)[1] for line in lines if "value = " in line]
        body, length = rendered.rsplit(" (length=", 1)
        assert length == "20)" and "..." in body
        if isinstance(value, dict):
            assert body.endswith(", ...}")
            actual = ast.literal_eval(body[:-6] + "}")
            assert actual == dict(list(value.items())[:16])
        else:
            actual = ast.literal_eval(body)
            assert Ellipsis in actual and len(actual) == 17
            assert set(actual) - {Ellipsis} <= set(value)


def test_text_values_bound_cycles_and_never_repr_business_objects(tmp_path: Path, business: Any) -> None:
    """循环容器必须有界；打印字典 key 不能执行业务对象的 repr。"""
    class Opaque:
        def __repr__(self):
            raise AssertionError("business repr must not execute")

    cycle = []
    cycle.append(cycle)
    prefix = tmp_path / "cycle"
    with _session(business, prefix).capture(output_prefix=prefix):
        business.observe({Opaque(): cycle})
    for suffix in (".txt", ".tree.txt"):
        text = Path(str(prefix) + suffix).read_text()
        assert "Opaque>" in text
        assert "[...] (length=1)" in text


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


@pytest.mark.parametrize("tensor_parallel_size", [None, "1"])
def test_multi_api_startup_keeps_two_frontends_without_data_parallelism(
    monkeypatch: pytest.MonkeyPatch, tensor_parallel_size: str | None,
) -> None:
    pytest.importorskip("vllm")
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from trace_python.call_stack_analysis.config.startup_common import _command
    from trace_python.call_stack_analysis.configuration import load_config

    monkeypatch.delenv("TRACE_VLLM_STARTUP_TENSOR_PARALLEL_SIZE", raising=False)
    if tensor_parallel_size is not None:
        monkeypatch.setenv("TRACE_VLLM_STARTUP_TENSOR_PARALLEL_SIZE", tensor_parallel_size)
    config = load_config(TOOL_DIRECTORY / "config/startup_multi_api.py")
    assert config.root == REPOSITORY_ROOT / "code/vllm-workspace/vllm"
    parser = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(parser.add_subparsers())
    args = parser.parse_args(_command("multi-api")[3:])
    assert args.api_server_count == 2
    assert args.data_parallel_size == 1
    assert args.tensor_parallel_size == int(tensor_parallel_size or "2")
    assert not args.headless


def test_same_name_children_keep_business_labels_and_separate_outputs(
    tmp_path: Path, business: Any,
) -> None:
    from dataclasses import replace

    context = multiprocessing.get_context("spawn")
    session = _session(business, tmp_path / "shared")
    processes = [
        context.Process(
            name="EngineCore", target=replace(session, process_label=label).run,
            args=(business.observe, label),
        )
        for label in ("P", "D")
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
    captures = [smoke.read_capture(path) for path in tmp_path.glob("shared.*.jsonl")]
    assert {capture["run"]["process_label"] for capture in captures} == {"P", "D"}
    for capture in captures:
        label = capture["run"]["process_label"]
        assert f".{label}.EngineCore." in capture["paths"][0].name
        assert capture["events"][0]["values"] == {"value": label}
        assert capture["summary"]["output_prefix"] == capture["run"]["output_prefix"]


def test_process_rename_moves_outputs_before_exit_even_after_hit_limit(
    tmp_path: Path, business: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    monkeypatch.setenv("TRACE_PROCESS_LABEL", "D")
    session = replace(_session(business, tmp_path / "capture"), process_label="P/节点")
    process = multiprocessing.current_process()
    previous_name = process.name
    suffixes = (".jsonl", ".txt", ".tree.json", ".tree.txt")
    try:
        with session.capture() as tracer:
            assert os.environ["TRACE_PROCESS_LABEL"] == "P/节点"
            business.observe("recorded")
            before = tracer.writer.prefix
            process.name = "Worker/TP:0"
            business.observe("over-limit")
            after = tracer.writer.prefix
            assert after != before
            assert after.parent == tmp_path
            assert ".P_节点.Worker_TP_0." in after.name
            assert all(Path(str(after) + suffix).is_file() for suffix in suffixes)
            assert not any(Path(str(before) + suffix).exists() for suffix in suffixes)
    finally:
        process.name = previous_name
    assert os.environ["TRACE_PROCESS_LABEL"] == "D"
    records = [json.loads(line) for line in Path(str(after) + ".jsonl").read_text().splitlines()]
    assert records[-1]["process_name"] == "Worker/TP:0"
    assert records[-1]["process_label"] == "P/节点"
    assert records[-1]["output_prefix"] == str(after)
    assert records[-1]["counts"] == {"smoke_business_functions.py:observe": 2}
    assert [record["values"] for record in records if record["type"] == "event"] == [
        {"value": "recorded"},
    ]


def test_subprocess_inherits_label_and_named_files_survive_sigkill(
    tmp_path: Path, business: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    monkeypatch.setenv("TRACE_PROCESS_LABEL", "original")
    subject = tmp_path / "named_subject.py"
    subject.write_text(
        "import multiprocessing, os, signal\n"
        "from pathlib import Path\n"
        "from trace_python.call_stack_analysis import Target, TraceSession\n"
        "def observe(value):\n    return value\n"
        "session = TraceSession(Path(__file__).parent,\n"
        "    (Target(Path(__file__).name, 'observe', ('value',)),),\n"
        "    Path(__file__).with_name('child'))\n"
        "with session.capture():\n"
        "    multiprocessing.current_process().name = 'EngineCore'\n"
        "    observe(42)\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n",
        encoding="utf-8",
    )
    session = replace(_session(business, tmp_path / "parent"), process_label="P")
    with session.capture():
        result = _run([sys.executable, str(subject)], tmp_path, module=True)
    assert result.returncode == -signal.SIGKILL, result.stderr
    assert os.environ["TRACE_PROCESS_LABEL"] == "original"
    path, = tmp_path.glob("child.*.jsonl")
    assert ".P.EngineCore." in path.name
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["type"] == "event"
    assert records[-1]["process_label"] == "P"
    assert records[-1]["process_name"] == "EngineCore"
    assert records[-1]["values"] == {"value": 42}
    assert Path(records[-1]["output_prefix"] + ".jsonl") == path
