import json
from pathlib import Path
import sys
import os

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from containers.core import mappings


def test_reuse_restore_and_referenced_mapping(tmp_path, monkeypatch):
    store = mappings.MappingStore()
    store.root = tmp_path / "registry"
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep").write_text("original")
    mounted = set()
    mounts = []
    monkeypatch.setattr(Path, "is_mount", lambda path: path in mounted)

    def run(*command, **kwargs):
        from subprocess import CompletedProcess

        if command[0] == "findmnt":
            return CompletedProcess(
                command,
                0,
                json.dumps(
                    {"filesystems": [{"source": str(source), "fstype": "fuse.bindfs"}]}
                ),
            )
        if command[:3] == ("sudo", "-n", "prlimit"):
            mounted.add(Path(command[-1]))
            mounts.append(command)
        elif command[:3] == ("sudo", "-n", "umount"):
            mounted.remove(Path(command[-1]))
        elif command[0] == "rmdir":
            Path(command[-1]).rmdir()
        elif command[0] == "rm":
            Path(command[-1]).unlink()
        return CompletedProcess(command, 0)

    monkeypatch.setattr(mappings, "run", run)
    monkeypatch.setattr(mappings, "containers", lambda: [])
    with store.locked():
        first = store.acquire(source, "bindfs")
        assert store.acquire(source, "bindfs") == first
        assert len(mounts) == 1
        # A stopped container still owns its mount reference.
        references = [
            {
                "State": {"Running": False},
                "Mounts": [{"Type": "bind", "Source": str(first)}],
            }
        ]
        store.remove_unused([str(first)], references)
        assert first in mounted
        mounted.clear()  # simulate host restart
        assert store.acquire(source, "bindfs") == first
        assert len(mounts) == 2
        store.remove_unused([str(first)], [])
        assert not first.parent.exists()
        assert (source / "keep").read_text() == "original"


@pytest.mark.parametrize("reference", ["view", "view/child", "."])
def test_remove_unused_rechecks_live_references(tmp_path, monkeypatch, reference):
    store = mappings.MappingStore()
    store.root = tmp_path / "registry"
    directory = store.root / "key"
    directory.mkdir(parents=True)
    target = directory / "view"
    target.mkdir()
    metadata = directory / "mapping.json"
    metadata.write_text(
        json.dumps(
            {
                "source": "/source",
                "uid": os.getuid(),
                "gid": os.getgid(),
            }
        )
    )
    calls = []
    monkeypatch.setattr(Path, "is_mount", lambda path: path == target)
    monkeypatch.setattr(store, "validate_mount", lambda *args: calls.append("validate"))

    def snapshot():
        assert calls == ["validate"]
        return [{"Mounts": [{"Type": "bind", "Source": str(directory / reference)}]}]

    monkeypatch.setattr(mappings, "containers", snapshot)
    monkeypatch.setattr(mappings, "run", lambda *args: calls.append(args))
    with store.locked():
        store.remove_unused([str(target)], [])
    assert calls == ["validate"]
    assert metadata.exists() and target.exists()


@pytest.mark.parametrize(
    "fstype,mounted_source,process_source,expected",
    [
        ("fuse.bindfs", "/source", None, True),
        ("fuse.bindfs", "/other", "/source", False),
        ("fuse", "/source", "/source", True),
        ("fuse", "/source", "/other", False),
        ("fuse", "/source", None, False),
        ("ext4", "/source", "/source", False),
    ],
)
def test_bindfs_mount_identity(fstype, mounted_source, process_source, expected):
    processes = {} if process_source is None else {"/view": process_source}
    assert (
        mappings.is_bindfs_mount(
            Path("/view"), Path("/source"), (fstype, mounted_source), processes
        )
        is expected
    )


def test_entry_removal_preserves_unrelated_files(tmp_path, monkeypatch):
    directory = tmp_path / "entry"
    directory.mkdir()
    target = directory / "view"
    target.mkdir()
    metadata = directory / "mapping.json"
    metadata.write_text("{}")
    keep = directory / "keep"
    keep.write_text("unrelated")
    calls = []

    def run(*args):
        calls.append(args)
        if args[0] == "rmdir":
            Path(args[-1]).rmdir()
        elif args[0] == "rm":
            Path(args[-1]).unlink()

    monkeypatch.setattr(mappings, "run", run)
    mappings.remove_mapping_entry(target, True, managed=True)
    assert calls == [
        ("sudo", "-n", "umount", str(target)),
        ("rmdir", str(target)),
        ("rm", "--", str(metadata)),
    ]
    assert keep.read_text() == "unrelated"


@pytest.mark.parametrize("second_running", [False, True])
@pytest.mark.parametrize("remove_last", [False, True])
def test_two_containers_share_mapping_until_last_deletion(
    tmp_path, monkeypatch, second_running, remove_last
):
    from argparse import Namespace
    from subprocess import CompletedProcess
    from containers.commands import delete

    store = mappings.MappingStore()
    store.root = tmp_path / "registry"
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep").write_text("user data")
    mounted = set()
    calls = []
    snapshot = []
    monkeypatch.setattr(Path, "is_mount", lambda path: path in mounted)

    def run(*command, **kwargs):
        calls.append(command)
        if command[:3] == ("sudo", "-n", "prlimit"):
            mounted.add(Path(command[-1]))
        elif command[0] == "findmnt":
            return CompletedProcess(
                command,
                0,
                json.dumps(
                    {"filesystems": [{"source": str(source), "fstype": "fuse.bindfs"}]}
                ),
            )
        elif command[:3] == ("sudo", "-n", "umount"):
            mounted.remove(Path(command[-1]))
        elif command[:2] == ("docker", "stop"):
            next(c for c in snapshot if c["Id"] == command[-1])["State"]["Running"] = (
                False
            )
        elif command[:2] == ("docker", "rm"):
            snapshot[:] = [c for c in snapshot if c["Id"] != command[-1]]
        elif command[0] == "rmdir":
            Path(command[-1]).rmdir()
        elif command[0] == "rm":
            Path(command[-1]).unlink()
        elif command[:3] != ("sudo", "-n", "python3"):
            pytest.fail(f"Unexpected external command: {command}")
        return CompletedProcess(command, 0)

    monkeypatch.setattr(mappings, "run", run)
    monkeypatch.setattr(delete, "run", run)
    monkeypatch.setattr(mappings, "containers", lambda: snapshot.copy())
    monkeypatch.setattr(delete, "containers", lambda: snapshot.copy())
    monkeypatch.setattr(delete, "MappingStore", lambda: store)
    monkeypatch.setattr(delete.os, "getuid", lambda: 1000)
    with store.locked():
        target = store.acquire(source, "bindfs")
        assert store.acquire(source, "bindfs") == target
    for index, running in ((1, True), (2, second_running)):
        snapshot.append(
            {
                "Id": str(index),
                "Name": f"/jiangyinzuo-bindfs-dev{index}",
                "State": {"Running": running},
                "Mounts": [{"Type": "bind", "Source": str(target)}],
            }
        )

    def remove(index, cleanup):
        delete.execute(
            Namespace(
                name=f"jiangyinzuo-bindfs-dev{index}",
                cleanup_bindfs=False,
                yes=False,
                keep_bindfs=not cleanup,
                remove_bindfs=cleanup,
            )
        )

    remove(1, True)
    assert [c["Id"] for c in snapshot] == ["2"]
    assert target in mounted and (target.parent / "mapping.json").is_file()
    assert not any(c[:3] == ("sudo", "-n", "umount") for c in calls)
    remove(2, remove_last)
    assert snapshot == []
    assert (target in mounted) == (not remove_last)
    assert target.parent.exists() == (not remove_last)
    assert sum(c[:3] == ("sudo", "-n", "prlimit") for c in calls) == 1
    assert sum(c[:3] == ("sudo", "-n", "umount") for c in calls) == int(remove_last)
    assert (source / "keep").read_text() == "user data"
