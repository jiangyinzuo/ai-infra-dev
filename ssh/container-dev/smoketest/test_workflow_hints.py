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


def test_dev_hints_only_for_containers_and_use_created_name(monkeypatch, capsys):
    monkeypatch.setattr(hints, "is_wsl", lambda: True)
    dev = Path(__file__).resolve().parents[3] / "code/dev"
    hints.show(
        build_parser().parse_args(["models", "discovery", "--host", "suzhou-one"])
    )
    assert f"python3 {dev}" not in capsys.readouterr().err
    hints.show(
        build_parser().parse_args(
            ["containers", "create", "--host", "suzhou-one", "--name", "my-container"]
        )
    )
    output = capsys.readouterr().err
    assert f"python3 {dev} vllm install --host suzhou-one" in output
    assert f"python3 {dev} vllm serve --script qwen3.5-0.8b.py" in output
    assert f"python3 {dev} devtools install" in output
    assert " vllm-ascend install" not in output
    assert " vllm-ascend serve" not in output
    assert "--container my-container" in output
    assert "<CONTAINER>" not in output
    hints.show(
        build_parser().parse_args(
            [
                "containers",
                "delete",
                "--host",
                "suzhou-one",
                "--name",
                "deleted-container",
            ]
        )
    )
    assert "--container deleted-container" not in capsys.readouterr().err
