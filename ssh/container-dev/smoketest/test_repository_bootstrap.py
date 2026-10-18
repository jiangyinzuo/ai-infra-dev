import os
from pathlib import Path
import subprocess
import sys
import pytest
import shutil


@pytest.fixture(autouse=True)
def ncat_available(tmp_path, monkeypatch):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    ncat = binaries / "ncat"
    ncat.write_text("#!/bin/sh\nexit 0\n")
    ncat.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + ":" + os.environ["PATH"])


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ssh_common import remote


def test_remote_bootstrap_clone_then_update_without_deployed_scripts(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    origins = tmp_path / "origins"
    origins.mkdir()
    env = dict(
        os.environ,
        HOME=str(home),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )

    def git(path, *args):
        return subprocess.run(
            ["git", "-C", str(path), *args],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    for name in ("ai-infra-dev", "vimrc"):
        p = origins / name
        p.mkdir()
        git(p, "init")
        (p / "content").write_text("first")
        if name == "vimrc":
            (p / "install").mkdir()
            setup = p / "install/setup.sh"
            setup.write_text('#!/bin/sh\necho setup >> "$HOME/setup-runs"\n')
            setup.chmod(0o755)
        git(p, "add", ".")
        git(
            p,
            "-c",
            "user.name=Smoke",
            "-c",
            "user.email=smoke@example.invalid",
            "commit",
            "-m",
            "initial",
        )

    def execute(host, command):
        for name in ("ai-infra-dev", "vimrc"):
            command = command.replace(
                "git@github.com:jiangyinzuo/" + name + ".git", str(origins / name)
            )
        return subprocess.run(
            ["sh", "-c", command], env=env, capture_output=True, text=True, check=True
        ).returncode

    monkeypatch.setattr(remote, "ssh_command", execute)
    remote.update_remote_repositories("suzhou-test")
    remote.update_remote_repositories("suzhou-test")
    assert (home / "ai-infra-dev/content").read_text() == "first"
    assert (home / "setup-runs").read_text().splitlines() == ["setup"]


def test_remote_debug_preserves_incoming_proxy(tmp_path, monkeypatch):
    proxy = "ssh -o 'ProxyCommand=ncat --proxy 192.0.2.10:3128 --proxy-type http %h %p'"
    home = tmp_path / "home"
    home.mkdir()
    binary = tmp_path / "git"
    binary.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$GIT_SSH_COMMAND" >> "$HOME/ssh-command-log"\nexit 0\n'
    )
    binary.chmod(0o755)
    for name in ("ai-infra-dev", "vimrc"):
        (home / name / ".git").mkdir(parents=True)
    env = dict(
        os.environ,
        HOME=str(home),
        PATH=str(tmp_path) + ":" + os.environ["PATH"],
        GIT_SSH_COMMAND=proxy,
    )

    def execute(host, command):
        return subprocess.run(["sh", "-c", command], env=env, check=True).returncode

    monkeypatch.setattr(remote, "ssh_command", execute)
    remote.update_remote_repositories("suzhou-test", debug=True)
    assert set((home / "ssh-command-log").read_text().splitlines()) == {proxy + " -vvv"}


@pytest.mark.parametrize("available", [(), ("git",), ("ncat",)])
def test_missing_prerequisites_stop_before_repository_changes(
    tmp_path, monkeypatch, available
):
    binaries = tmp_path / "isolated-bin"
    binaries.mkdir()
    marker = tmp_path / "unexpected-git-run"
    for name in available:
        executable = binaries / name
        executable.write_text(f"#!/bin/sh\nprintf called > '{marker}'\n")
        executable.chmod(0o755)
    shell = shutil.which("sh")
    results = []

    def execute(host, command):
        result = subprocess.run(
            [shell, "-c", command],
            env={"HOME": str(tmp_path), "PATH": str(binaries)},
            capture_output=True,
            text=True,
        )
        results.append(result)
        return result.returncode

    monkeypatch.setattr(remote, "ssh_command", execute)
    assert remote.update_remote_repositories("suzhou-test") == 127
    for name in set(("git", "ncat")) - set(available):
        assert name in results[0].stderr
    assert not marker.exists()
    assert not (tmp_path / "ai-infra-dev").exists()
