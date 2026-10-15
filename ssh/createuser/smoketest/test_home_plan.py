import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "home_plan", Path(__file__).resolve().parents[1] / "home_plan.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def disks(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("home", "data", "data3")]
    for path in paths:
        path.mkdir()
    monkeypatch.setattr(MODULE, "CANDIDATES", tuple(map(str, paths)))
    monkeypatch.setattr(
        MODULE.pwd, "getpwnam", lambda name: (_ for _ in ()).throw(KeyError(name))
    )
    free = {path: (index + 1) * MODULE.MIN_FREE for index, path in enumerate(paths)}
    monkeypatch.setattr(MODULE, "available", lambda path: free[path])
    return paths, free


def test_new_user_selects_largest_existing_directory(disks):
    paths, _ = disks
    assert MODULE.plan_home("jiangyinzuo") == {
        "home": str(paths[2] / "jiangyinzuo"),
        "move": False,
    }


def test_low_space_and_existing_destination_rejected(disks):
    paths, free = disks
    (paths[2] / "jiangyinzuo").mkdir()
    with pytest.raises(ValueError, match="已存在"):
        MODULE.plan_home("jiangyinzuo")
    for path in paths:
        free[path] = MODULE.MIN_FREE - 1
    with pytest.raises(ValueError, match="不足 10 GiB"):
        MODULE.plan_home("jiangyinzuo")


def test_existing_home_with_space_stays_put(disks, monkeypatch):
    paths, free = disks
    current = paths[0] / "jiangyinzuo"
    current.mkdir()
    free[current] = MODULE.MIN_FREE
    monkeypatch.setattr(
        MODULE.pwd, "getpwnam", lambda name: SimpleNamespace(pw_dir=str(current))
    )
    assert MODULE.plan_home("jiangyinzuo") == {"home": str(current), "move": False}


def test_same_filesystem_cannot_resolve_low_space(disks, monkeypatch):
    paths, free = disks
    current = paths[0] / "jiangyinzuo"
    current.mkdir()
    free[current] = MODULE.MIN_FREE - 1
    monkeypatch.setattr(
        MODULE.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(pw_dir=str(current), pw_uid=current.stat().st_uid),
    )
    with pytest.raises(ValueError, match="其他.*文件系统"):
        MODULE.plan_home("jiangyinzuo")


def test_cross_disk_migration_requires_room_for_home(disks, monkeypatch):
    paths, free = disks
    current = paths[0] / "jiangyinzuo"
    current.mkdir()
    free[current] = 1
    monkeypatch.setattr(
        MODULE.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(pw_dir=str(current), pw_uid=999999),
    )
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        value = original_stat(path, *args, **kwargs)
        return SimpleNamespace(st_uid=999999, st_dev=-1) if path == current else value

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    monkeypatch.setattr(
        MODULE.subprocess, "check_output", lambda *a, **k: str(MODULE.MIN_FREE)
    )
    assert MODULE.plan_home("jiangyinzuo") == {
        "home": str(paths[2] / "jiangyinzuo"),
        "move": True,
    }
    free[paths[2]] = MODULE.MIN_FREE
    free[paths[1]] = MODULE.MIN_FREE
    free[paths[0]] = MODULE.MIN_FREE
    with pytest.raises(ValueError, match="保留 10 GiB"):
        MODULE.plan_home("jiangyinzuo")
