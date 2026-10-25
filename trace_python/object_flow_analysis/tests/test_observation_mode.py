"""关闭载体观察仍保留身份边界，原容器与帧敏感调用正常执行。"""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_container_mode_changes_evidence_not_program_result(tmp_path, enabled):
    project = tmp_path / "project"
    project.mkdir()
    (project / "scene.py").write_text('''
def consume(value):
    return value

def run():
    target = []
    mapping = {}
    mapping["key"] = target
    value = mapping.get("key")
    copied = target.copy()
    result = consume(value)
    assert result is target
    assert copied is not target
    assert locals()["value"] is target
    print("SCENE_OK")
''')
    config = tmp_path / "config.py"
    config.write_text('''from pathlib import Path
from trace_python.object_flow_analysis import FlowConfig, Origin

def run():
    import scene
    scene.run()

config = FlowConfig(Path(__file__).parent / "project", Origin("scene.py", "run", 6, "target"), run,
                    observe_containers=True)
''')
    output = tmp_path / "flow.jsonl"
    command = [sys.executable, "-m", "trace_python.object_flow_analysis",
               "--config", str(config), "--output", str(output),
               "--observe-containers" if enabled else "--no-observe-containers"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[3],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "SCENE_OK" in result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[-1]["observe_containers"] is enabled
    kinds = {event["kind"] for event in rows[:-1]}
    assert {"capture", "bind", "argument", "return"} <= kinds
    if enabled:
        assert {"observed_write", "builtin_result", "shallow_copy"} <= kinds
    else:
        assert kinds <= {"capture", "bind", "argument", "return"}
        assert len(rows[-1]["objects"]) == 1
