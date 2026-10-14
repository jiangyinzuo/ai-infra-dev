import gzip
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

DEVTOOLS = Path(__file__).resolve().parents[2] / "devtools"


@pytest.fixture
def workspace(tmp_path):
    home = tmp_path / "home with spaces"
    tools = home / "devtools/openeuler24.03-lts-aarch64"
    tools.mkdir(parents=True)
    shutil.copy2(
        DEVTOOLS / "openeuler24.03-lts-aarch64/extract.sh", tools / "extract.sh"
    )
    shutil.copy2(DEVTOOLS / "remote-exec.sh", tools.parent / "remote-exec.sh")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uname = binaries / "uname"
    uname.write_text("#!/bin/sh\necho Linux\n")
    uname.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{binaries}:{os.environ['PATH']}"}
    return home, tools, env


def extract(workspace):
    home, tools, env = workspace
    return subprocess.run(
        ["bash", str(tools / "extract.sh")],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


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
