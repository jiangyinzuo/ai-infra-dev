"""无需真实 NPU 的端到端冒烟测试。

运行：uv run --with pytest pytest -q smoketest/
依赖：ansible-playbook、Bash、SSH 客户端。
"""

from pathlib import Path
import socket
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(hosts_file, *args):
    result = subprocess.run(
        [str(ROOT / "npu-top.sh"), "-i", str(hosts_file),
         "--limit", "smoke-*", *args],
        cwd=hosts_file.parent,
        text=True,
        capture_output=True,
        timeout=30,
    )
    print(result.stdout, end="")
    print(result.stderr, end="")
    return result


@pytest.fixture
def hosts_file(tmp_path):
    path = tmp_path / "hosts.ini"
    path.write_text("[npu_servers]\nsmoke-local ansible_connection=local\n", encoding="utf-8")
    return path


def test_success_output(hosts_file):
    fake = hosts_file.parent / "npu-smi"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "cat <<'EOF'\n"
        "| NPU Name | Health | AICore(%) |\n"
        "| 0 Ascend | OK | - |\n"
        "| 0 | 0000:01:00.0 | 0 0 / 1024 0 / 32768 |\n"
        "| NPU Chip | Process id | Process name |\n"
        "EOF\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    result = run(hosts_file, "-e", f"npu_smi_bin={fake}")
    assert result.returncode == 0
    assert "NPU=0 Chip=0 AICore=0%" in result.stdout
    assert "smoke-local IdleCards=1/1 NPU=[0]" in result.stdout
    for marker in ("TASK [", "PLAY [", "PLAY RECAP", "ok:", "skipping:"):
        assert marker not in result.stdout


def test_command_failure(hosts_file):
    result = run(hosts_file, "-e", "npu_smi_bin=/bin/false")
    assert result.returncode == 2
    assert "FAILED!" in result.stdout
    assert "TOP10" in result.stdout
    assert "查询失败或未获得有效汇总：smoke-local" in result.stdout
    assert "PLAY RECAP" not in result.stdout


def test_unreachable(hosts_file):
    # 保持绑定但不监听，避免其他进程占用端口并意外连上真实 SSH。
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        hosts_file.write_text(
            "[npu_servers]\n"
            f"smoke-unreachable ansible_host=127.0.0.1 ansible_port={port} "
            "ansible_connection=ssh "
            'ansible_ssh_common_args="-o ConnectTimeout=1 -o ConnectionAttempts=1"\n',
            encoding="utf-8",
        )
        result = run(hosts_file)
    assert result.returncode == 2
    assert "UNREACHABLE!" in result.stdout
    assert "TOP10" in result.stdout
    assert "查询失败或未获得有效汇总：smoke-unreachable" in result.stdout
    assert "PLAY RECAP" not in result.stdout
