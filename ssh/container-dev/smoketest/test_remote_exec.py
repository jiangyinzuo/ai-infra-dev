from pathlib import Path
import shutil
import subprocess
import sys

import pytest

DEVTOOLS = Path(__file__).resolve().parents[1] / "devtools"


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
