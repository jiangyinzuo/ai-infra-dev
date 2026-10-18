from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from cli import build_parser
from workflow import sync_all, binaries


@pytest.mark.parametrize("fail", [None, 0, 1, 2, 3])
def test_order_local_download_and_stop_on_failure(monkeypatch, fail):
    monkeypatch.setattr(sync_all, "is_wsl", lambda: True)
    calls = []

    def step(index):
        def execute(args):
            calls.append((index, args))
            return 23 if index == fail else 0

        return execute

    monkeypatch.setattr(sync_all.update_git_repo, "execute", step(0))

    def binary(args):
        return step(1 if args.binaries_action == "download" else 3)(args)

    monkeypatch.setattr(binaries, "execute", binary)
    monkeypatch.setattr(sync_all.sync, "execute", step(2))
    args = build_parser().parse_args(
        ["binaries", "sync-all", "--host", "suzhou-one", "--jobs", "3"]
    )
    assert sync_all.execute(args) == (0 if fail is None else 23)
    assert [i for i, _ in calls] == list(range(4 if fail is None else fail + 1))
    if len(calls) > 1:
        assert calls[1][1].host is None
        assert calls[1][1].jobs == 3
    if len(calls) > 2:
        assert calls[2][1].apply and calls[2][1].host == ["suzhou-one"]


def test_sync_all_rejects_server_before_actions(monkeypatch):
    monkeypatch.setattr(sync_all, "is_wsl", lambda: False)
    args = build_parser().parse_args(["binaries", "sync-all", "--host", "suzhou-one"])
    with pytest.raises(ValueError, match="WSL"):
        sync_all.execute(args)
