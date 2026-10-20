import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import models
import model_paths
from cli import build_parser


def test_path_inspection(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "existing").touch()
    link = tmp_path / "linked"
    link.symlink_to(directory, target_is_directory=True)
    row = model_paths.inspect_path(link)
    assert row["symlink"] and row["contents"] == "非空"
    assert row["read"] and row["enter"]
    missing = model_paths.inspect_path(directory / "new")
    assert not missing["exists"] and missing["usable"]
    directory.chmod(0o700)
    assert not model_paths.public_access(directory)


def test_upload_real_rsync_public_permissions(monkeypatch, tmp_path):
    if not shutil.which("rsync"):
        pytest.skip("rsync not installed")
    source = tmp_path / "local" / "model with spaces"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "config.json").chmod(0o600)
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    (source / "model.safetensors").symlink_to(blob)
    parent = tmp_path / "remote"
    parent.mkdir(mode=0o755)
    ssh = tmp_path / "ssh2"
    ssh.write_text(
        '#!/bin/sh\nif [ "$1" = -l ]; then shift 2; fi\nshift\nexec sh -c "$*"\n'
    )
    ssh.chmod(0o755)
    rows = model_paths.upload_paths([str(parent)], source.name)
    monkeypatch.setattr(models, "SSH_DIR", tmp_path)
    monkeypatch.setattr(models, "is_wsl", lambda: True)
    monkeypatch.setattr(models, "checked_command", lambda host, command: command)
    from ssh_common import transfer

    monkeypatch.setattr(transfer, "checked_command", lambda host, command: command)
    original = subprocess.run

    def run(command, **kwargs):
        if command[0] == str(ssh):
            return subprocess.CompletedProcess(command, 0, json.dumps(rows))
        return original(command, **kwargs)

    monkeypatch.setattr(models.subprocess, "run", run)
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    args = build_parser().parse_args(
        ["models", "upload", "--host", "suzhou-one", "--model", str(source)]
    )
    assert models.upload_model(args) == 0
    destination = parent / source.name
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert not (destination / "model.safetensors").is_symlink()
    assert destination.stat().st_mode & 0o777 == 0o755
    assert (destination / "config.json").stat().st_mode & 0o777 == 0o644
    assert parent.stat().st_mode & 0o777 == 0o755
