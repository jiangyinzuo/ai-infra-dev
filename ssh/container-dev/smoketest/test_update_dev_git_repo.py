import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "devtools/update-dev-git-repo.sh"


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
        git("add", "content", cwd=source, env=env)
        git("commit", "-m", "initial", cwd=source, env=env)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Only replace the network destination; clone/pull run through real Git.
    real_git = shutil.which("git")
    wrapper = binaries / "git"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        "if [[ $1 == clone ]]; then\n"
        "  repo=${2##*/}; repo=${repo%.git}\n"
        "  shift 2\n"
        '  exec "$REAL_GIT" clone "$SOURCES/$repo" "$@"\n'
        'fi\nexec "$REAL_GIT" "$@"\n'
    )
    wrapper.chmod(0o755)
    for name, body in {
        "uname": "echo Linux",
        "nc": "echo Ncat",
    }.items():
        stub = binaries / name
        stub.write_text(f"#!/bin/sh\n{body}\n")
        stub.chmod(0o755)
    env.update(
        PATH=f"{binaries}:{os.environ['PATH']}",
        REAL_GIT=real_git,
        SOURCES=str(sources),
    )
    return home, sources, env


def run_script(env, *args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
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
    for name in ("ai-infra-dev", "vimrc"):
        assert (home / name / "content").read_text() == "initial\n"
        source = sources / name
        (source / "content").write_text("updated\n")
        git("commit", "-am", "update", cwd=source, env=env)
    result = run_script(env, *args)
    assert result.returncode == 0, result.stderr
    for name in ("ai-infra-dev", "vimrc"):
        assert (home / name / "content").read_text() == "updated\n"
        assert git("rev-parse", "HEAD", cwd=home / name, env=env) == git(
            "rev-parse", "HEAD", cwd=sources / name, env=env
        )


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
