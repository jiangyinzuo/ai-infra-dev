import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
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
        if command[:3] == ("sudo", "-n", "bindfs"):
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
