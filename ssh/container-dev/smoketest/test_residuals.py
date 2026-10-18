import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from containers.core import mappings, residuals


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
    monkeypatch.setattr(mappings, "run", lambda *a: calls.append(a))
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


def test_reference_overlap_names_are_sorted_and_unique():
    target = Path("/registry/key/view")
    snapshot = [
        {"Name": "/z", "Mounts": [{"Type": "bind", "Source": str(target)}] * 2},
        {"Name": "/a", "Mounts": [{"Type": "bind", "Source": str(target / "child")}]},
        {"Name": "/parent", "Mounts": [{"Type": "bind", "Source": "/registry"}]},
        {"Name": "/sibling", "Mounts": [{"Type": "bind", "Source": str(target) + "2"}]},
        {"Name": "/volume", "Mounts": [{"Type": "volume", "Source": str(target)}]},
    ]
    assert residuals.references(target, snapshot) == ["a", "parent", "z"]


@pytest.mark.parametrize(
    "has_process,kind,allowed",
    [
        (True, "legacy", True),
        (False, "unverified-legacy", False),
    ],
)
def test_legacy_discovery_is_preserved(
    tmp_path, monkeypatch, has_process, kind, allowed
):
    parent = Path("/var/tmp/root-mapped-regression")
    target = parent / "123"
    original_stat = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path, **kwargs: (
            SimpleNamespace(st_uid=os.getuid())
            if path == parent
            else original_stat(path, **kwargs)
        ),
    )
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: False if path == parent else original_is_symlink(path),
    )
    monkeypatch.setattr(
        residuals,
        "mount_table",
        lambda: {
            str(target): ("fuse.bindfs", "/missing"),
        },
    )
    monkeypatch.setattr(
        residuals,
        "bindfs_processes",
        lambda: (
            {
                str(target): "/missing",
            }
            if has_process
            else {}
        ),
    )
    row = residuals.scan(SimpleNamespace(root=tmp_path), [])[0]
    assert row["kind"] == kind
    assert row["cleanup_allowed"] is allowed


def test_cleanup_rechecks_mount_identity(tmp_path, monkeypatch):
    directory = tmp_path / "key"
    directory.mkdir()
    target = directory / "view"
    target.mkdir()
    (directory / "mapping.json").write_text(
        json.dumps(
            {
                "source": "/missing",
                "uid": os.getuid(),
                "gid": os.getgid(),
            }
        )
    )
    mounts = {str(target): ("fuse.bindfs", "/missing")}
    monkeypatch.setattr(residuals, "mount_table", lambda: mounts)
    monkeypatch.setattr(residuals, "bindfs_processes", lambda: {})
    monkeypatch.setattr(residuals, "containers", lambda: [])
    calls = []
    monkeypatch.setattr(mappings, "run", lambda *args: calls.append(args))
    store = SimpleNamespace(root=tmp_path)
    rows = residuals.scan(store, [])
    assert rows[0]["cleanup_allowed"]
    mounts[str(target)] = ("ext4", "/missing")
    residuals.cleanup(store, rows)
    assert calls == []
    assert target.exists()
