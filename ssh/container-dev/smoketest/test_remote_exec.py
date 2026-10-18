from pathlib import Path
import shutil
import subprocess
import sys

import pytest

DEVTOOLS = Path(__file__).resolve().parents[1] / "devtools"
sys.path.insert(0, str(DEVTOOLS))
import remote_exec


@pytest.mark.parametrize(
    "host, expected",
    [(None, False), ("", False), ("localhost", False), ("suzhou-one", True)],
)
def test_target_classification(host, expected):
    assert remote_exec.is_remote_host(host) is expected


def test_remote_cli_adds_entrypoint_and_disables_hints(monkeypatch):
    calls = []

    def run_remote(host, script, args):
        calls.append((host, script, args))
        return 7

    monkeypatch.setattr(remote_exec, "run_remote", run_remote)
    arguments = ["containers", "create", "--image", "image with spaces"]
    assert remote_exec.run_remote_cli("suzhou-one", arguments) == 7
    assert calls == [("suzhou-one", "cli.py", ["--no-hints", *arguments])]
    assert arguments == ["containers", "create", "--image", "image with spaces"]


def test_remote_cli_quotes_arguments_and_preserves_exit_status(monkeypatch):
    import shlex

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 19)

    monkeypatch.setattr(remote_exec.subprocess, "run", run)
    arguments = ["containers", "create", "--name", "a b;$(touch unwanted)"]
    assert remote_exec.run_remote_cli("suzhou-one", arguments) == 19
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[1] == "jiangyinzuo@suzhou-one"
    assert shlex.split(command[2]) == [
        "python3",
        "$HOME/devtools/cli.py",
        "--no-hints",
        *arguments,
    ]
    assert kwargs == {"check": False}


def test_remote_cli_rejects_invalid_host_before_execution(monkeypatch):
    monkeypatch.setattr(
        remote_exec.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid host reached SSH"),
    )
    with pytest.raises(ValueError, match="--host"):
        remote_exec.run_remote_cli("user@host", ["containers", "ls"])


@pytest.mark.parametrize(
    "release, expected", [("Microsoft-WSL2", True), ("Linux", False)]
)
def test_wsl_detection(monkeypatch, release, expected):
    monkeypatch.setattr(remote_exec.platform, "release", lambda: release)
    assert remote_exec.is_wsl() is expected


@pytest.mark.parametrize(
    "command,args",
    [
        ("nvim-plugins", ["--jobs", "2"]),
        ("binaries", []),
    ],
)
def test_extract_dispatches_once_as_fixed_user(tmp_path, command, args):
    tools = tmp_path / "ssh/container-dev/devtools"
    shutil.copytree(
        DEVTOOLS,
        tools,
        ignore=shutil.ignore_patterns(
            "__pycache__", "nvim-plugins", "*.tar.gz", "*.tar.xz"
        ),
    )
    ssh = tools.parents[1] / "ssh2"
    ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    ssh.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            str(tools / "cli.py"),
            "--no-hints",
            command,
            "extract",
            "--host",
            "suzhou-one",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "jiangyinzuo@suzhou-one" in result.stdout
    assert (
        'python3 "$HOME"/devtools/cli.py --no-hints ' + command + " extract"
        in result.stdout
    )
    assert "No supported archives" not in result.stdout
    assert "No plugin archives" not in result.stdout
