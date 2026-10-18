import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / "devtools/containers"
sys.path.insert(0, str(ROOT))


def test_interactive_delete_keeps_mapping_and_targets_selected_id(
    monkeypatch, tmp_path
):
    spec = importlib.util.spec_from_file_location(
        "container_delete", ROOT / "delete.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    monkeypatch.setattr(sys, "argv", ["delete.py"])
    calls = []
    monkeypatch.setattr(
        module,
        "run",
        lambda *a, **kw: calls.append(a) or subprocess.CompletedProcess(a, 0),
    )
    module.main()
    assert calls == [("docker", "stop", "own-id"), ("docker", "rm", "-v", "own-id")]
