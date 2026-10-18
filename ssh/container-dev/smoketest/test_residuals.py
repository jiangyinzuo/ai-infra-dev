import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools/containers"))
from core import residuals


def test_deleted_source_and_stopped_reference(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    directory = root / "key"
    directory.mkdir(parents=True)
    target = directory / "view"
    target.mkdir()
    source = tmp_path / "deleted-home"
    (directory / "mapping.json").write_text(
        json.dumps({"source": str(source), "uid": os.getuid(), "gid": os.getgid()})
    )
    monkeypatch.setattr(
        residuals, "mount_table", lambda: {str(target): ("fuse", str(source))}
    )
    monkeypatch.setattr(
        residuals, "bindfs_processes", lambda: {str(target): str(source)}
    )
    store = SimpleNamespace(root=root)
    row = residuals.scan(store, [])[0]
    assert not row["source_exists"] and row["cleanup_allowed"]
    stopped = [{"Name": "/other", "Mounts": [{"Type": "bind", "Source": str(target)}]}]
    assert not residuals.scan(store, stopped)[0]["cleanup_allowed"]
    calls = []
    monkeypatch.setattr(residuals, "containers", lambda: stopped)
    monkeypatch.setattr(residuals, "run", lambda *a: calls.append(a))
    residuals.cleanup(store, [row])
    assert not calls


def test_unknown_fuse_is_not_cleanable(tmp_path, monkeypatch):
    d = tmp_path / "key"
    d.mkdir()
    (d / "mapping.json").write_text(
        json.dumps({"source": "/missing", "uid": os.getuid(), "gid": os.getgid()})
    )
    monkeypatch.setattr(
        residuals, "mount_table", lambda: {str(d / "view"): ("fuse", "/missing")}
    )
    monkeypatch.setattr(residuals, "bindfs_processes", lambda: {})
    assert not residuals.scan(SimpleNamespace(root=tmp_path), [])[0]["cleanup_allowed"]
