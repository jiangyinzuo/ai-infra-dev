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
    groups = hints.groups(args)
    assert [title for title, _ in groups] == ["Workflow", "Container"]
    workflow = groups[0][1]
    assert [command[:2] for _, command in workflow] == [
        ["git-repo", "update"],
        ["binaries", "download"],
        ["binaries", "sync"],
        ["binaries", "extract"],
    ]
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
