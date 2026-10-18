import gzip
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

DEVTOOLS = Path(__file__).resolve().parents[2] / "devtools"


@pytest.fixture
def workspace(tmp_path):
    home = tmp_path / "home with spaces"
    tools = home / "devtools/openeuler24.03-lts-aarch64"
    tools.mkdir(parents=True)
    for source in DEVTOOLS.glob("*.py"):
        shutil.copy2(source, tools.parent / source.name)
    for package in ("workflow", "containers"):
        shutil.copytree(
            DEVTOOLS / package,
            tools.parent / package,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.tar", "*.tar.gz", "*.tar.zst"
            ),
        )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for utility in (
        "dirname",
        "tar",
        "gzip",
        "mkdir",
        "rm",
        "rmdir",
        "chmod",
        "cp",
        "cat",
        "env",
        "printf",
        "bash",
        "grep",
    ):
        (binaries / utility).symlink_to(shutil.which(utility))
    uname = binaries / "uname"
    uname.write_text("#!/bin/sh\necho Linux\n")
    uname.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": str(binaries)}
    hooks = tmp_path / "python-hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        'import platform\nplatform.release = lambda: "Linux"\n'
    )
    env["PYTHONPATH"] = str(hooks)
    return home, tools, env


def extract(workspace):
    home, tools, env = workspace
    return subprocess.run(
        [
            sys.executable,
            str(tools.parent / "cli.py"),
            "--no-hints",
            "binaries",
            "extract",
        ],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def bindfs_archive(tools):
    with tarfile.open(tools / "bindfs-1.17.7.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("bindfs-1.17.7/configure")
        entry.size = 0
        archive.addfile(entry, io.BytesIO())


@pytest.mark.parametrize("location", ["path", "local"])
def test_bindfs_build_runs_only_when_bindfs_is_missing(workspace, location):
    home, tools, env = workspace
    bindfs_archive(tools)
    builder = home / "vimrc/install/build_bindfs.sh"
    builder.parent.mkdir(parents=True)
    builder.write_text(
        'set -eu\ntest -f "$1/configure"\nprintf built > "$HOME/bindfs-build"\n'
    )
    sudo = Path(env["PATH"]) / "sudo"
    sudo.write_text('#!/bin/sh\nprintf installed > "$HOME/packages-installed"\n')
    sudo.chmod(0o755)

    result = extract(workspace)
    assert result.returncode == 0, result.stderr
    assert (home / "bindfs-build").read_text() == "built"
    assert (home / "packages-installed").read_text() == "installed"

    (home / "bindfs-build").unlink()
    (home / "packages-installed").unlink()
    builder.unlink()
    directory = Path(env["PATH"]) if location == "path" else home / ".local/bin"
    directory.mkdir(parents=True, exist_ok=True)
    installed = directory / "bindfs"
    installed.write_text("#!/bin/sh\nexit 0\n")
    installed.chmod(0o755)

    result = extract(workspace)
    assert result.returncode == 0, result.stderr
    assert not (home / "bindfs-build").exists()
    assert not (home / "packages-installed").exists()


def test_bindfs_package_failure_prevents_build(workspace):
    home, tools, env = workspace
    bindfs_archive(tools)
    builder = home / "vimrc/install/build_bindfs.sh"
    builder.parent.mkdir(parents=True)
    builder.write_text('touch "$HOME/unexpected-build"\n')
    sudo = Path(env["PATH"]) / "sudo"
    sudo.write_text("#!/bin/sh\nexit 42\n")
    sudo.chmod(0o755)
    result = extract(workspace)
    assert result.returncode == 42
    assert not (home / "unexpected-build").exists()
    assert not (home / ".bashrc").exists()


def test_fzf_installs_all_three_tools_in_plugin_bin(workspace):
    home, tools, env = workspace
    support = tools / "fzf-0.74.5-support"
    support.mkdir()
    for name in ("fzf-preview.sh", "fzf-tmux"):
        (support / name).write_text(f"#!/bin/sh\nprintf '{name}'\n")
    binary = b"#!/bin/sh\nprintf fzf-smoke\n"
    with tarfile.open(tools / "fzf-0.74.5-linux_arm64.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("fzf")
        entry.size = len(binary)
        archive.addfile(entry, io.BytesIO(binary))
    destination = home / "plugged/fzf/bin"
    destination.mkdir(parents=True)
    (destination / "keep").write_text("preserved")
    for _ in range(2):
        result = extract(workspace)
        assert result.returncode == 0, result.stderr
    for name, expected in (
        ("fzf", "fzf-smoke"),
        ("fzf-preview.sh", "fzf-preview.sh"),
        ("fzf-tmux", "fzf-tmux"),
    ):
        result = subprocess.run(
            [str(destination / name)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout == expected
    assert (destination / "keep").read_text() == "preserved"
    assert not (tools / "fzf").exists()


def test_no_archives_configures_bashrc_once(workspace):
    home, tools, env = workspace
    bashrc = home / ".bashrc"
    bashrc.write_text("# existing configuration without final newline")
    for _ in range(3):
        result = extract(workspace)
        assert result.returncode == 0, result.stderr
    content = bashrc.read_text()
    assert content.startswith("# existing configuration without final newline\n")
    assert (
        content.count('source "$HOME/devtools/openeuler24.03-lts-aarch64/path.env"')
        == 1
    )
    # Missing path.env is safe; when present, the installed line loads it.
    command = ["bash", "-c", 'source "$HOME/.bashrc"']
    subprocess.run(command, env=env, check=True, timeout=30)
    (tools / "path.env").write_text("export EXTRACT_SMOKE=loaded\n")
    result = subprocess.run(
        ["bash", "-c", 'source "$HOME/.bashrc"; printf "%s" "$EXTRACT_SMOKE"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout == "loaded"


def test_tar_overwrites_files_preserving_extras_and_archive(workspace):
    home, tools, env = workspace
    directory = tools / "tool"
    directory.mkdir(mode=0o700)
    (directory / "binary").write_bytes(b"old")
    (directory / "extra").write_bytes(b"keep")
    archive = tools / "tool.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("tool/binary")
        entry.size = len(b"updated")
        entry.mode = 0o755
        tar.addfile(entry, io.BytesIO(b"updated"))
    original = archive.read_bytes()
    env["TAR_OPTIONS"] = "--keep-old-files"
    for _ in range(2):
        result = extract(workspace)
        assert result.returncode == 0, result.stderr
    assert (directory / "binary").read_bytes() == b"updated"
    assert (directory / "extra").read_bytes() == b"keep"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert archive.read_bytes() == original
    assert (home / ".bashrc").read_text().count(
        'source "$HOME/devtools/openeuler24.03-lts-aarch64/path.env"'
    ) == 1


def test_single_gzip_renames_tree_sitter_and_makes_it_executable(workspace):
    _, tools, env = workspace
    payload = b"#!/bin/sh\nprintf tree-sitter-smoke\n"
    archive = tools / "tree-sitter-linux-arm64.gz"
    archive.write_bytes(gzip.compress(payload))
    (tools / "notes.gz").write_bytes(gzip.compress(b"notes\n"))
    result = extract(workspace)
    assert result.returncode == 0, result.stderr
    assert (tools / "tree-sitter").read_bytes() == payload
    assert not (tools / "tree-sitter-linux-arm64").exists()
    assert (tools / "notes").read_bytes() == b"notes\n"
    assert archive.is_file()
    result = subprocess.run(
        [str(tools / "tree-sitter")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "tree-sitter-smoke"


def test_corrupt_archive_fails_before_bashrc_changes(workspace):
    home, tools, _ = workspace
    (tools / "broken.tar.gz").write_bytes(b"not an archive")
    result = extract(workspace)
    assert result.returncode != 0
    assert not (home / ".bashrc").exists()
