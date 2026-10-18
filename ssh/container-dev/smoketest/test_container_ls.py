import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
from cli import build_parser
from containers.commands import ls
from containers.core import mappings, process, residuals


def test_ls_json_filters_display_but_checks_all_references(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "registry"
    directory = root / "key"
    directory.mkdir(parents=True)
    target = directory / "view"
    target.mkdir()
    source = tmp_path / "deleted-source"
    metadata = directory / "mapping.json"
    metadata.write_text(
        json.dumps(
            {
                "source": str(source),
                "uid": os.getuid(),
                "gid": os.getgid(),
            }
        )
    )
    original = metadata.read_bytes()
    own = {
        "Id": "own-id",
        "Name": "/jiangyinzuo-bindfs-dev1",
        "Config": {"Image": "vllm-ascend:test"},
        "State": {"Status": "running", "Running": True},
        "Mounts": [
            {"Type": "bind", "Source": str(target), "Destination": "/home/user"}
        ],
    }
    foreign = {
        "Id": "foreign-id",
        "Name": "/someone-else",
        "Config": {"Image": "other:image"},
        "State": {"Status": "exited", "Running": False},
        "Mounts": [{"Type": "bind", "Source": str(target), "Destination": "/data"}],
    }
    monkeypatch.setattr(ls, "containers", lambda: [foreign, own])
    monkeypatch.setattr(ls, "MappingStore", lambda: SimpleNamespace(root=root))
    monkeypatch.setattr(residuals, "mount_table", lambda: {})
    monkeypatch.setattr(residuals, "bindfs_processes", lambda: {})

    def forbidden(*args, **kwargs):
        pytest.fail("ls must not create, mount, remove, or prompt")

    for module in (process, mappings, residuals):
        monkeypatch.setattr(module, "run", forbidden)
    monkeypatch.setattr(residuals, "cleanup", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    ls.execute(build_parser().parse_args(["containers", "ls", "--json"]))

    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"containers", "mappings"}
    assert len(result["containers"]) == 1
    container = result["containers"][0]
    assert set(container) == {"name", "status", "image", "mappings"}
    assert container["name"] == "jiangyinzuo-bindfs-dev1"
    assert container["status"] == "running"
    assert container["image"] == "vllm-ascend:test"
    assert container["mappings"]
    assert result["mappings"] == [
        {
            "target": str(target),
            "source": str(source),
            "kind": "managed",
            "mounted": False,
            "source_exists": False,
            "containers": ["jiangyinzuo-bindfs-dev1", "someone-else"],
            "cleanup_allowed": False,
        }
    ]
    assert metadata.read_bytes() == original
    assert target.is_dir()
    assert not source.exists()


def test_ls_json_empty_registry_is_read_only(tmp_path, monkeypatch, capsys):
    root = tmp_path / "absent-registry"
    monkeypatch.setattr(ls, "containers", lambda: [])
    monkeypatch.setattr(ls, "MappingStore", lambda: SimpleNamespace(root=root))
    monkeypatch.setattr(residuals, "mount_table", lambda: {})
    monkeypatch.setattr(residuals, "bindfs_processes", lambda: {})

    def forbidden(*args, **kwargs):
        pytest.fail("empty ls must not initialize or modify the registry")

    for module in (process, mappings, residuals):
        monkeypatch.setattr(module, "run", forbidden)
    monkeypatch.setattr(residuals, "cleanup", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    ls.execute(build_parser().parse_args(["containers", "ls", "--json"]))

    assert json.loads(capsys.readouterr().out) == {"containers": [], "mappings": []}
    assert not root.exists()
