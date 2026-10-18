import shlex
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from cli import build_parser
from containers import interface
from containers.commands import create
from containers.core import remote
from ssh_common.config import REMOTE_USER


def test_wsl_remote_create_forwards_origin_and_actual_name(monkeypatch, capsys):
    monkeypatch.setattr(interface, "is_wsl", lambda: True)
    calls = []
    monkeypatch.setattr(
        remote, "dispatch", lambda args, action, options: calls.append(options) or 0
    )
    args = build_parser().parse_args(["containers", "create", "--host", "suzhou-one"])
    assert interface.execute(args) == 0
    assert calls[0][-2:] == ["--connection-host", "suzhou-one"]
    create.print_instructions("chosen-container", "/home/user", "suzhou-one")
    lines = capsys.readouterr().out.splitlines()
    command = next(line for line in lines if line.startswith("ssh2 "))
    tokens = shlex.split(command)
    assert tokens[:3] == ["ssh2", "-t", REMOTE_USER + "@suzhou-one"]
    assert shlex.split(tokens[3]) == [
        "docker",
        "exec",
        "-it",
        "chosen-container",
        "bash",
    ]
