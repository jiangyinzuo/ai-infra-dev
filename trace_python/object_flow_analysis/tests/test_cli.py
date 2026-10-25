"""场景失败仍输出已有证据，配置错误在场景前报告。"""
import json
from pathlib import Path
import subprocess
import sys


def test_failed_scene_preserves_capture(tmp_path):
    root = Path(__file__).resolve().parents[3]
    project = tmp_path / "project"
    project.mkdir()
    subject = project / "subject_failure.py"
    subject.write_text('def run():\n    target = object()\n    raise ValueError("scene failed")\n')
    config = tmp_path / "config.py"
    config.write_text('''from pathlib import Path
from trace_python.object_flow_analysis import FlowConfig, Origin

def run():
    import subject_failure
    subject_failure.run()

config = FlowConfig(Path(__file__).parent / "project", Origin("subject_failure.py", "run", 2, "target"), run)
''')
    output = tmp_path / "events.jsonl"
    result = subprocess.run([sys.executable, "-m", "trace_python.object_flow_analysis",
                             "--config", str(config), "--output", str(output)],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert any(record.get("kind") == "capture" for record in records)
    assert records[-1]["error"] == {"type": "ValueError", "message": "scene failed"}
    assert Path(str(output) + ".tree.txt").is_file()
