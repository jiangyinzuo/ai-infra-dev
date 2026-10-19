"""离线冒烟测试：不连接 SSH，不读取真实模型目录。"""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import models as discovery
import cli


def run_cli(arguments):
    with patch.object(
        sys, "argv", ["cli.py", "--no-hints", "models", "discovery", *arguments]
    ):
        return cli.main()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(discovery, "is_wsl", lambda: False)


def make_model(path, weight="model.safetensors"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / weight).write_bytes(b"smoke fixture")
    return str(path)


def test_discovers_models_at_every_data_level(server, tmp_path):
    expected = []
    roots = []
    for suffix in ("", "1", "2", "3"):
        root = tmp_path / f"data{suffix}"
        roots.append(root)
        expected.append(make_model(root))
        for name in ("weight", "weights", "model", "models"):
            child = root / name
            roots.append(child)
            expected.append(make_model(child / "模型 with spaces"))
        expected.append(
            make_model(root / "other" / "nested", "pytorch_model-00001.bin")
        )
    roots.append(tmp_path / "missing")
    assert discovery.discover_models(roots=roots) == sorted(expected)


def test_ignores_incomplete_candidates(server, tmp_path):
    config_only = tmp_path / "config-only"
    config_only.mkdir()
    (config_only / "config.json").write_text("{}", encoding="utf-8")
    weight_only = tmp_path / "weight-only"
    weight_only.mkdir()
    (weight_only / "model.safetensors").write_bytes(b"fixture")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{}", encoding="utf-8")
    (broken / "model.safetensors").symlink_to(tmp_path / "absent")
    assert discovery.discover_models(roots=[tmp_path]) == []


def test_supports_weight_links_without_following_directory_links(server, tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"fixture")
    root = tmp_path / "search"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "pytorch_model.bin").symlink_to(source)
    outside = tmp_path / "outside"
    make_model(outside)
    (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    assert discovery.discover_models(roots=[root]) == [str(root)]


@pytest.mark.parametrize("host", [None, "", "localhost"])
def test_wsl_rejects_local_scan_before_accessing_roots(monkeypatch, host):
    monkeypatch.setattr(discovery, "is_wsl", lambda: True)

    def roots():
        pytest.fail("WSL local scan accessed roots")
        yield

    with pytest.raises(ValueError):
        discovery.discover_models(host=host, roots=roots())


def test_cli_requires_remote_host_on_wsl(monkeypatch, capsys):
    monkeypatch.setattr(discovery, "is_wsl", lambda: True)
    assert run_cli([]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "--host" in output.err


def test_server_cli_outputs_json_without_host(server, tmp_path, capsys):
    expected = make_model(tmp_path / "model")
    assert run_cli(["--root", str(tmp_path), "--json"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == [expected]
    assert output.err == ""


def test_file_root_is_an_error(server, tmp_path, capsys):
    root = tmp_path / "not-a-directory"
    root.write_text("fixture", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        discovery.discover_models(roots=[root])
    assert run_cli(["--root", str(root)]) == 1
    assert capsys.readouterr().out == ""


def test_empty_roots_do_not_use_defaults(server):
    assert discovery.discover_models(roots=[]) == []


def test_skips_permission_denied_directories(server, tmp_path, capsys):
    import os

    allowed = make_model(tmp_path / "allowed")
    denied = tmp_path / "lost+found"
    make_model(denied / "hidden")
    denied.chmod(0)
    try:
        if os.access(denied, os.R_OK | os.X_OK):
            pytest.skip("当前用户可绕过目录权限")
        assert discovery.discover_models(roots=[tmp_path, denied]) == [allowed]
        assert run_cli(["--root", str(tmp_path), "--json"]) == 0
        output = capsys.readouterr()
        assert json.loads(output.out) == [allowed]
        assert output.err == ""
    finally:
        denied.chmod(0o700)
