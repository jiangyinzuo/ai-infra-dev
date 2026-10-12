import importlib.util
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "process_containers", Path(__file__).resolve().parents[1] / "npu_process_containers.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_device_process_uses_host_pid_and_queries_owner_once():
    text = """| NPU ID | Process id | Process name | Process memory(MB) | Process id in container |
| 0 | 123 | worker | 4096 | 7 |
| 1 | 123 | worker | 2048 | 7 |
"""
    with patch.object(MODULE, "process_owner", return_value="model [abc]") as owner:
        lines = MODULE.container_lines(text)
    owner.assert_called_once_with(123)
    assert "0 / — / 123 / worker / model [abc] / 4096" in lines
    assert "1 / — / 123 / worker / model [abc] / 2048" in lines


def test_chip_process_keeps_chip_association():
    rows = MODULE.parse_processes("""| NPU Chip | Process id | Process name | Process memory(MB) |
| 2 1 | 456 | inference | 8192 |
""")
    assert rows == [{"npu": 2, "chip": 1, "pid": 456, "name": "inference", "memory": "8192"}]


def test_exited_and_denied_process_are_distinct():
    with patch.object(Path, "read_text", side_effect=FileNotFoundError):
        assert MODULE.process_owner(123) == "进程已退出"
    with patch.object(Path, "read_text", side_effect=PermissionError):
        assert MODULE.process_owner(123) == "权限不足（/proc）"


def test_unknown_cgroup_is_not_claimed_as_host_process():
    with patch.object(Path, "read_text", return_value="0::/user.slice/session.scope\n"):
        assert MODULE.process_owner(123) == "未识别（无容器 ID）"
