from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from containers.core import host
from containers.commands import create
from ssh_common import container


@pytest.fixture
def bare_host(monkeypatch):
    monkeypatch.setattr(host, "is_wsl", lambda: False)
    monkeypatch.setattr(container.socket, "gethostname", lambda: "server")
    monkeypatch.setattr(container.Path, "exists", lambda p: False)
    monkeypatch.setattr(container.Path, "read_text", lambda p: "0::/\n")


@pytest.mark.parametrize(
    "kind", ["wsl", "hostname", "docker", "podman", "systemd", "cgroup"]
)
def test_non_host_rejected_before_statvfs(bare_host, monkeypatch, kind):
    if kind == "wsl":
        monkeypatch.setattr(host, "is_wsl", lambda: True)
    elif kind == "hostname":
        monkeypatch.setattr(container.socket, "gethostname", lambda: "server-container")
    elif kind == "cgroup":
        monkeypatch.setattr(
            container.Path, "read_text", lambda p: "0::/system.slice/docker-123.scope"
        )
    else:
        marker = {
            "docker": "/.dockerenv",
            "podman": "/run/.containerenv",
            "systemd": "/run/systemd/container",
        }[kind]
        monkeypatch.setattr(container.Path, "exists", lambda p: str(p) == marker)
    monkeypatch.setattr(host.os, "statvfs", lambda p: pytest.fail("read non-host shm"))
    with pytest.raises(ValueError):
        host.shm_capacity()


def test_exact_total_capacity_not_available_space(bare_host, monkeypatch):
    monkeypatch.setattr(
        host.os,
        "statvfs",
        lambda p: SimpleNamespace(f_frsize=4096, f_blocks=123456, f_bavail=1),
    )
    command = create.docker_command("example", "/home/user")
    assert command[command.index("--shm-size") + 1] == str(4096 * 123456)
    assert "--ipc=host" not in command


def test_empty_capacity_and_read_failure(bare_host, monkeypatch):
    monkeypatch.setattr(
        host.os, "statvfs", lambda p: SimpleNamespace(f_frsize=4096, f_blocks=0)
    )
    with pytest.raises(ValueError):
        host.shm_capacity()

    def missing(p):
        raise FileNotFoundError(p)

    monkeypatch.setattr(host.os, "statvfs", missing)
    with pytest.raises(FileNotFoundError):
        host.shm_capacity()
