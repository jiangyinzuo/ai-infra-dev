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
