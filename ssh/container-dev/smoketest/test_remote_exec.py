import os
from pathlib import Path
import shutil
import subprocess

import pytest

DEVTOOLS = Path(__file__).resolve().parents[1] / "devtools"


@pytest.mark.parametrize(
    "script,args",
    [
        ("extract-nvim-plugin.sh", ["--jobs", "2"]),
        ("openeuler24.03-lts-aarch64/extract.sh", []),
    ],
)
def test_extract_dispatches_once_as_fixed_user(tmp_path, script, args):
    tools = tmp_path / "ssh/container-dev/devtools"
    (tools / script).parent.mkdir(parents=True)
    shutil.copy2(DEVTOOLS / script, tools / script)
    for name in ("remote_exec.py", "config.py"):
        shutil.copy2(DEVTOOLS / name, tools / name)
    ssh = tools.parents[1] / "ssh2"
    ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    ssh.chmod(0o755)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uname = binaries / "uname"
    uname.write_text("#!/bin/sh\necho Microsoft\n")
    uname.chmod(0o755)
    result = subprocess.run(
        ["bash", str(tools / script), "--host", "suzhou-one", *args],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "jiangyinzuo@suzhou-one" in result.stdout
    assert 'bash "$HOME"/devtools/' + script in result.stdout
    assert "No supported archives" not in result.stdout
