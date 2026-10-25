"""用实际转换并执行的源码检查对象身份与原程序语义。"""
import ast
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from trace_python.object_flow_analysis.backend import transform
from trace_python.object_flow_analysis.runtime import Runtime


def trace_source(tmp_path, source, *, max_events=10000):
    path = tmp_path / "subject.py"
    path.write_text(source, encoding="utf-8")
    tree = ast.parse(source)
    line = next(node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(value, ast.Name) and value.id == "target"
                        for value in node.targets))
    runtime = Runtime(tmp_path, max_events=max_events)
    code = transform(compile(source, str(path), "exec"), runtime.dispatch,
                     (str(path), "run", line, "target"))
    module = ModuleType("flow_subject")
    module.__file__ = str(path)
    namespace = module.__dict__
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    runtime.start()
    try:
        exec(code, namespace)
        result = namespace["run"]()
    finally:
        runtime.stop()
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    return runtime, result


def test_actual_identity_alias_binding_and_arguments(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def consume(value=None):
    return value

def run():
    target = []
    alias = target
    target = None
    result = consume(*(), **{"value": alias})
    return result is alias
''')
    assert result is True
    events = runtime.records
    captured = next(event["object_id"] for event in events if event["kind"] == "capture")
    assert any(event["kind"] == "argument" and event["object_id"] == captured
               and event["site"]["function"] == "consume" for event in events)
    assert any(event["kind"] == "return" and event["object_id"] == captured
               and event["site"]["function"] == "consume" for event in events)


def test_write_order_and_failure_does_not_record_success(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def run():
    target = object()
    order = []
    mapping = {}
    def value():
        order.append("value")
        return target
    def container():
        order.append("container")
        return mapping
    def key():
        order.append("key")
        return "ok"
    container()[key()] = value()
    try:
        mapping[[]] = target
    except TypeError:
        order.append("failed")
    return order, mapping["ok"] is target
''')
    assert result == (["value", "container", "key", "failed"], True)
    writes = [event for event in runtime.records if event["kind"] == "observed_write"]
    assert len(writes) == 1


def test_copy_identity_and_carrier_history(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def consume(value):
    return value

def run():
    target = [1]
    mapping = {}
    mapping["x"] = target
    mapping.clear()
    consume(mapping)
    copied = target.copy()
    consume(copied)
    return copied == target and copied is not target
''')
    assert result is True
    events = runtime.records
    captured = next(event["object_id"] for event in events if event["kind"] == "capture")
    assert any(event["kind"] == "shallow_copy" and event["object_id"] != captured
               for event in events)
    assert any(event["kind"] == "historical_carrier_use" for event in events)


def test_limits_do_not_change_target_execution(tmp_path):
    runtime, result = trace_source(tmp_path, '''
def consume(value):
    return value

def run():
    target = []
    for i in range(10):
        consume(target).append(i)
    return target
''', max_events=2)
    assert result == list(range(10))
    assert runtime.summary()["incomplete"] is True
    assert len(runtime.records) <= 2


def test_cli_outputs_real_events_and_call_tree(tmp_path):
    root = Path(__file__).resolve().parents[3]
    config = Path(__file__).resolve().parents[1] / "examples" / "config.py"
    output = tmp_path / "events.jsonl"
    result = subprocess.run([
        sys.executable, "-m", "trace_python.object_flow_analysis",
        "--config", str(config), "--output", str(output),
    ], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    import json
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert any(record.get("kind") == "capture" for record in records)
    assert any(record.get("kind") == "observed_write" for record in records)
    tree = output.with_suffix(".tree.txt").read_text()
    assert "inspect_mapping" in tree
    assert "历史载体" in tree
    structured = json.loads(output.with_suffix(".tree.json").read_text())
    assert structured["summary"]["captures"] == records[-1]["captures"]
    assert any(label.startswith("run (") for label in structured["tree"])
    assert not Path(str(output) + ".tree.txt").exists()
