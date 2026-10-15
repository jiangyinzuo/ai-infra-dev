from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import bootstrap


@pytest.fixture
def target(tmp_path):
    (tmp_path / "id_ed25519").write_text("smoke-private")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA smoke")
    config = tmp_path / "config"
    config.write_text(
        'Host xiaoqiaoling-one\n User other\n IgnoreUnknown Password\n Password "test-password"\n'
    )
    return {
        "host": "xiaoqiaoling-one",
        "public_key": str(tmp_path / "id_ed25519.pub"),
        "config": str(config),
        "ssh_config": str(config),
        "ssh": "/usr/bin/ssh",
        "password_method": "sshpass",
    }


def response(code, stderr=""):
    return subprocess.CompletedProcess([], code, "", stderr)


def test_existing_key_skips_password_and_upload(target, monkeypatch):
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: response(0))
    monkeypatch.setattr(
        bootstrap, "copy_key", lambda *a: pytest.fail("uploaded existing key")
    )
    monkeypatch.setattr(
        bootstrap,
        "host_passwords",
        lambda *a: pytest.fail("read password unnecessarily"),
    )
    bootstrap.ensure_root_access(target)


def test_password_install_then_verify_and_skip_on_repeat(target, monkeypatch):
    results = iter(
        [
            response(255, "Permission denied (publickey,password)."),
            response(0),
            response(0),
        ]
    )
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: next(results))
    uploads = []
    monkeypatch.setattr(bootstrap, "copy_key", lambda *args: uploads.append(args) or 0)
    bootstrap.ensure_root_access(target)
    bootstrap.ensure_root_access(target)
    assert len(uploads) == 1
    assert uploads[0][0][-1] == "root@xiaoqiaoling-one"
    assert uploads[0][1] == "test-password"
    assert target["public_key"] in uploads[0][0]


@pytest.mark.parametrize(
    "check,message",
    [(True, "Permission denied"), (False, "Host key verification failed")],
)
def test_check_and_transport_errors_never_upload(target, monkeypatch, check, message):
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *a, **k: response(255, message)
    )
    monkeypatch.setattr(
        bootstrap, "copy_key", lambda *a: pytest.fail("unexpected upload")
    )
    with pytest.raises(ValueError):
        bootstrap.ensure_root_access(target, check=check)
