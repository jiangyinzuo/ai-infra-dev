import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
SPEC = importlib.util.spec_from_file_location(
    "podman_storage", Path(__file__).resolve().parents[1] / "devtools/podman_storage.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_largest_local_writable_disk(tmp_path, monkeypatch):
    home = tmp_path / "home/user"
    home.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setattr(MODULE, "CANDIDATES", (str(data), str(remote)))
    monkeypatch.setattr(MODULE, "local_directory", lambda path: path != remote)
    sizes = {
        home: MODULE.PODMAN_MIN_FREE,
        data: MODULE.PODMAN_MIN_FREE * 2,
        remote: MODULE.PODMAN_MIN_FREE * 3,
    }
    monkeypatch.setattr(MODULE, "available", sizes.__getitem__)
    assert MODULE.select_storage(home, "user") == data / "user/podman-storage"
    monkeypatch.setattr(MODULE, "available", lambda path: MODULE.PODMAN_MIN_FREE - 1)
    with pytest.raises(ValueError, match="200 GiB"):
        MODULE.select_storage(home, "user")


def test_existing_storage_and_configuration_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("CONTAINERS_STORAGE_CONF", raising=False)
    monkeypatch.setattr(
        MODULE, "select_storage", lambda *args: pytest.fail("existing store relocated")
    )
    store = tmp_path / ".local/share/containers/storage"
    store.mkdir(parents=True)
    (store / "db.sql").write_text("existing store")
    MODULE.configure()
    config = tmp_path / ".config/containers/storage.conf"
    assert not config.exists()
    config.parent.mkdir(parents=True)
    config.write_text('[storage]\ngraphroot = "/existing"\n')
    MODULE.configure()
    assert config.read_text() == '[storage]\ngraphroot = "/existing"\n'
