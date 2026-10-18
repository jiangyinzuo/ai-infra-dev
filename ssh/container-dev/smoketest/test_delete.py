from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / "devtools/containers"
sys.path.insert(0, str(ROOT))
from commands import delete as module


def test_interactive_delete_keeps_mapping_and_targets_selected_id(
    monkeypatch, tmp_path
):
    own = {
        "Id": "own-id",
        "Name": "/jiangyinzuo-bindfs-dev1",
        "State": {"Status": "running", "Running": True},
        "Mounts": [],
    }
    foreign = dict(own, Id="other-id", Name="/someone-else")
    monkeypatch.setattr(module, "containers", lambda: [foreign, own])
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    store = module.MappingStore()
    store.root = tmp_path / "registry"
    monkeypatch.setattr(module, "MappingStore", lambda: store)
    monkeypatch.setattr(
        store, "remove_unused", lambda *a: pytest.fail("default must retain mappings")
    )
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    calls = []
    monkeypatch.setattr(
        module,
        "run",
        lambda *a, **kw: calls.append(a) or subprocess.CompletedProcess(a, 0),
    )
    module.execute(
        Namespace(
            name=None,
            keep_bindfs=False,
            remove_bindfs=False,
            cleanup_bindfs=False,
            yes=False,
        )
    )
    assert calls == [("docker", "stop", "own-id"), ("docker", "rm", "-v", "own-id")]
