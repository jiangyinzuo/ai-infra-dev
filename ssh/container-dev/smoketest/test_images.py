from pathlib import Path
import subprocess
import sys
import shutil

import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[1] / "devtools"),
]
from workflow import images
from cli import build_parser


def test_upload_real_rsync(monkeypatch, tmp_path):
    if not shutil.which("rsync"):
        pytest.skip("rsync not installed")
    source = tmp_path / "image with spaces.tar.gz"
    source.write_bytes(b"image archive fixture")
    destination = tmp_path / "remote with spaces"
    ssh = tmp_path / "ssh2"
    ssh.write_text(
        '#!/bin/sh\nif [ "$1" = -l ]; then shift 2; fi\nshift\nexec sh -c "$*"\n'
    )
    ssh.chmod(0o755)
    monkeypatch.setattr(images, "SSH_DIR", tmp_path)
    monkeypatch.setattr(images, "REMOTE_DIR", str(destination))
    monkeypatch.setattr(images, "checked_command", lambda host, command: command)
    monkeypatch.setattr(images, "is_wsl", lambda: True)
    args = build_parser().parse_args(
        ["images", "upload", "--host", "suzhou-one", "--archive", str(source)]
    )
    assert images.upload(args) == 0
    assert (
        destination / "containers" / source.name
    ).read_bytes() == source.read_bytes()


def test_upload_rejects_server(monkeypatch):
    monkeypatch.setattr(images, "is_wsl", lambda: False)
    args = build_parser().parse_args(
        ["images", "upload", "--host", "suzhou-one", "--archive", "missing"]
    )
    with pytest.raises(ValueError, match="WSL"):
        images.upload(args)
