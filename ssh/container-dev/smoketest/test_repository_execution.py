import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ssh_common import remote


def test_commit_guard_matches_mismatch_and_missing(tmp_path, monkeypatch):
    repo = tmp_path / "ai-infra-dev"
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], env=env, text=True
        ).strip()

    repo.mkdir()
    git("init")
    git(
        "-c",
        "user.name=Smoke",
        "-c",
        "user.email=smoke@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(remote, "local_commit", lambda: head)
    command = remote.checked_command("suzhou-test", "printf EXECUTED")
    result = subprocess.run(
        ["sh", "-c", command], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and result.stdout == "EXECUTED"
    monkeypatch.setattr(remote, "local_commit", lambda: "f" * 40)
    result = subprocess.run(
        ["sh", "-c", remote.checked_command("suzhou-test", "printf EXECUTED")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and not result.stdout
    assert "git-repo update --host suzhou-test" in result.stderr
    env["HOME"] = str(tmp_path / "missing")
    result = subprocess.run(
        ["sh", "-c", command], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2 and "remote=missing" in result.stderr


def test_bootstrap_uses_existing_checkout_without_guard(monkeypatch):
    monkeypatch.setattr(
        remote,
        "local_commit",
        lambda: (_ for _ in ()).throw(
            AssertionError("bootstrap must not require matching commit")
        ),
    )
    commands = []
    monkeypatch.setattr(
        remote, "ssh_command", lambda host, command: commands.append(command) or 0
    )
    assert remote.update_remote_repositories("suzhou-test") == 0
    assert "pull --ff-only" in commands[0]
    assert "git clone" in commands[0]
    assert "reset --hard" not in commands[0]
