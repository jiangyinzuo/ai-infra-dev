from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cli import build_parser
from workflow import hints


def test_host_reused_in_valid_followup_commands(monkeypatch):
    monkeypatch.setattr(hints, "is_wsl", lambda: True)
    args = build_parser().parse_args(["binaries", "sync", "--host", "suzhou-one"])
    for _, command in hints.suggestions(args):
        parsed = build_parser().parse_args(command)
        if command[:2] == ["binaries", "download"]:
            assert parsed.host is None
        else:
            assert parsed.host in ("suzhou-one", ["suzhou-one"])


def test_json_and_disabled_hints_emit_nothing(capsys):
    hints.show(SimpleNamespace(json=True))
    hints.show(SimpleNamespace(no_hints=True))
    assert capsys.readouterr() == ("", "")


def test_server_does_not_suggest_wsl_only_fetch(monkeypatch):
    monkeypatch.setattr(hints, "is_wsl", lambda: False)
    args = build_parser().parse_args(["models", "discovery"])
    commands = [command for _, command in hints.suggestions(args)]
    assert ["models", "discovery"] in commands
    assert not any(command[:2] == ["models", "fetch"] for command in commands)
