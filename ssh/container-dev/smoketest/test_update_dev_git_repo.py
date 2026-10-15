import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "devtools/update-dev-git-repo.py"


def git(*args, cwd=None, env=None):
    return subprocess.run(
        [shutil.which("git"), *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def workspace(tmp_path):
    home = tmp_path / "home with spaces"
    home.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Smoke",
        "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
        "GIT_COMMITTER_NAME": "Smoke",
        "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
        "USG_IP": "192.0.2.1",
    }
    for name in ("ai-infra-dev", "vimrc"):
        source = sources / name
        git("init", str(source), env=env)
        (source / "content").write_text("initial\n")
        if name == "vimrc":
            install = source / "install"
            install.mkdir()
            setup = install / "setup.sh"
            setup.write_text('#!/bin/sh\nset -eu\npwd >> "$HOME/setup-runs"\n')
            setup.chmod(0o755)
            git("add", "install/setup.sh", cwd=source, env=env)
        git("add", "content", cwd=source, env=env)
        git("commit", "-m", "initial", cwd=source, env=env)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Only replace the network destination; clone/pull run through real Git.
    real_git = shutil.which("git")
    wrapper = binaries / "git"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        'printf "%s\\n" "$GIT_SSH_COMMAND" >> "$HOME/git-ssh-commands"\n'
        'printf "%s\\n" "${GIT_TRACE:-}" >> "$HOME/git-traces"\n'
        "if [[ $1 == clone ]]; then\n"
        "  repo=${2##*/}; repo=${repo%.git}\n"
        "  shift 2\n"
        '  exec "$REAL_GIT" clone "$SOURCES/$repo" "$@"\n'
        'fi\nexec "$REAL_GIT" "$@"\n'
    )
    wrapper.chmod(0o755)
    for name, body in {
        "nc": 'echo "${NC_HELP:-Ncat}"',
    }.items():
        stub = binaries / name
        stub.write_text(f"#!/bin/sh\n{body}\n")
        stub.chmod(0o755)
    env.update(
        PATH=f"{binaries}:{os.environ['PATH']}",
        REAL_GIT=real_git,
        SOURCES=str(sources),
    )
    # Exercise the real CLI in a child interpreter without relying on the host's
    # kernel (the smoke suite itself may run on WSL). This hook is test-only.
    hooks = tmp_path / "python-hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        "import os, platform\n"
        'platform.release = lambda: os.environ.get("TEST_KERNEL_RELEASE", "Linux")\n'
    )
    env["PYTHONPATH"] = str(hooks)
    return home, sources, env


def run_script(env, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("args", [(), ("--debug",)])
def test_clone_then_fast_forward(workspace, args):
    home, sources, env = workspace
    result = run_script(env, *args)
    assert result.returncode == 0, result.stderr
    assert (home / "setup-runs").read_text().splitlines() == [str(home / "vimrc")]
    for name in ("ai-infra-dev", "vimrc"):
        assert (home / name / "content").read_text() == "initial\n"
        source = sources / name
        (source / "content").write_text("updated\n")
        git("commit", "-am", "update", cwd=source, env=env)
    result = run_script(env, *args)
    assert result.returncode == 0, result.stderr
    assert (home / "setup-runs").read_text().splitlines() == [str(home / "vimrc")]
    for name in ("ai-infra-dev", "vimrc"):
        assert (home / name / "content").read_text() == "updated\n"
        assert git("rev-parse", "HEAD", cwd=home / name, env=env) == git(
            "rev-parse", "HEAD", cwd=sources / name, env=env
        )


def test_setup_failure_is_reported(workspace):
    home, sources, env = workspace
    source = sources / "vimrc"
    (source / "install/setup.sh").write_text("#!/bin/sh\nexit 23\n")
    git("commit", "-am", "failing setup", cwd=source, env=env)
    result = run_script(env)
    assert result.returncode == 23
    assert (home / "vimrc/.git").is_dir()


def test_divergence_preserves_local_commit(workspace):
    home, sources, env = workspace
    assert run_script(env).returncode == 0
    local = home / "ai-infra-dev"
    (local / "local-only").write_text("keep me\n")
    git("add", "local-only", cwd=local, env=env)
    git("commit", "-m", "local change", cwd=local, env=env)
    head = git("rev-parse", "HEAD", cwd=local, env=env)
    source = sources / "ai-infra-dev"
    (source / "content").write_text("upstream\n")
    git("commit", "-am", "upstream change", cwd=source, env=env)
    result = run_script(env)
    assert result.returncode != 0
    assert git("rev-parse", "HEAD", cwd=local, env=env) == head
    assert (local / "local-only").read_text() == "keep me\n"
    assert (local / "content").read_text() == "initial\n"


def test_existing_non_repository_is_preserved(workspace):
    home, _, env = workspace
    destination = home / "ai-infra-dev"
    destination.mkdir()
    marker = destination / "important"
    marker.write_text("do not overwrite\n")
    result = run_script(env)
    assert result.returncode != 0
    assert "not a Git repository" in result.stderr
    assert marker.read_text() == "do not overwrite\n"
    assert not (destination / ".git").exists()
    assert not (home / "vimrc").exists()


@pytest.mark.parametrize(
    "nc_help, expected",
    [
        ("Ncat 7.94", ["--proxy-type", "http", "--proxy", "192.0.2.1:3128"]),
        ("OpenBSD netcat", ["-X", "connect", "-x", "192.0.2.1:3128"]),
    ],
)
@pytest.mark.parametrize("debug", [False, True])
def test_proxy_commands_and_debug(workspace, nc_help, expected, debug):
    import shlex

    home, _, env = workspace
    env["NC_HELP"] = nc_help
    env.pop("GIT_TRACE", None)
    result = run_script(env, *(("--debug",) if debug else ()))
    assert result.returncode == 0, result.stderr
    commands = (home / "git-ssh-commands").read_text().splitlines()
    assert len(commands) == 2
    ssh = shlex.split(commands[0])
    assert ssh[0] == "ssh"
    for option in (
        "StrictHostKeyChecking=accept-new",
        "Hostname=ssh.github.com",
        "HostKeyAlias=github.com",
    ):
        assert option in ssh
    assert ssh[ssh.index("-p") + 1] == "443"
    proxy = next(arg for arg in ssh if arg.startswith("ProxyCommand="))
    nc = shlex.split(proxy.partition("=")[2])
    assert nc == ["nc", *(["-v"] if debug else []), *expected, "%h", "%p"]
    assert ("-vvv" in ssh) == debug
    assert (home / "git-traces").read_text().splitlines() == (
        ["1"] * 2 if debug else [""] * 2
    )
    assert "GIT_SSH_COMMAND=" in result.stdout
    assert "./install/setup.sh" in result.stdout
    assert "HTTP CONNECT proxy=192.0.2.1:3128" in result.stderr


@pytest.mark.parametrize("target", ["user@host", "-host", "host;id", "host/path", ""])
def test_rejects_non_alias_remote_target(workspace, target):
    home, _, env = workspace
    env["TEST_KERNEL_RELEASE"] = "6.1-microsoft-standard-WSL2"
    result = run_script(env, "--host=" + target)
    assert result.returncode == 2
    assert not (home / "ai-infra-dev").exists()


def test_wsl_requires_host(workspace):
    home, _, env = workspace
    env["TEST_KERNEL_RELEASE"] = "6.1-microsoft-standard-WSL2"
    result = run_script(env)
    assert result.returncode == 2
    assert "WSL requires --host" in result.stderr
    assert not (home / "ai-infra-dev").exists()


def test_server_rejects_host(workspace):
    home, _, env = workspace
    result = run_script(env, "--host", "test-host")
    assert result.returncode == 2
    assert "--host" in result.stderr
    assert not (home / "ai-infra-dev").exists()


def test_missing_proxy_is_reported(workspace):
    home, _, env = workspace
    env.pop("USG_IP", None)
    result = run_script(env)
    assert result.returncode == 1
    assert "USG_IP" in result.stderr
    assert not (home / "ai-infra-dev").exists()
