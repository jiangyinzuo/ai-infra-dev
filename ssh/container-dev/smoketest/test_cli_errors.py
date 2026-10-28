import sys
from pathlib import Path
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import cli


@pytest.mark.parametrize("diagnostic", ["远程仓库 commit 不一致\n请先更新仓库\n", None])
def test_subprocess_failure_reports_diagnostic_not_shell_payload(
    monkeypatch, capsys, diagnostic
):
    def fail(args):
        raise subprocess.CalledProcessError(
            2, ["/path/ssh2", "private-shell-payload"], stderr=diagnostic
        )

    class Parser:
        def parse_args(self):
            from argparse import Namespace

            return Namespace(handler=fail)

    monkeypatch.setattr(cli, "build_parser", Parser)
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private-shell-payload" not in captured.err
    if diagnostic:
        assert captured.err == diagnostic
    else:
        assert "ssh2" in captured.err and "2" in captured.err


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["binaries"],
        ["binaries", "sync"],
        ["git-repo"],
        ["containers"],
        ["sync-all"],
        ["models"],
        ["models", "fetch"],
        ["models", "upload"],
        ["images"],
        ["images", "download"],
        ["images", "upload"],
    ],
)
def test_empty_required_command_prints_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(command)
    assert error.value.code == 0
    assert capsys.readouterr().out.startswith("usage:")


@pytest.mark.parametrize(
    "command",
    [
        ["binaries", "download"],
        ["binaries", "extract"],
        ["git-repo", "update"],
        ["containers", "create"],
        ["containers", "delete"],
        ["containers", "ls"],
        ["models", "discovery"],
        ["checkhealth"],
        ["images", "ls"],
    ],
)
def test_empty_optional_command_preserves_execution(command, capsys):
    args = cli.build_parser().parse_args(command)
    assert args.command == command[0]
    assert capsys.readouterr().out == ""
