import sys
from pathlib import Path
import shutil

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import models
from cli import build_parser


def args(tmp_path):
    return build_parser().parse_args(
        [
            "models",
            "fetch",
            "--host",
            "suzhou-one",
            "--output",
            str(tmp_path / "downloads"),
        ]
    )


def test_fetch_selected_model_with_real_rsync(monkeypatch, tmp_path):
    if not shutil.which("rsync"):
        pytest.skip("rsync not installed")
    source = tmp_path / "remote" / "model with spaces"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    weight = tmp_path / "cache-blob"
    weight.write_bytes(b"weight fixture")
    (source / "model.safetensors").symlink_to(weight)
    # A local transport stand-in runs rsync's actual remote server command.
    ssh = tmp_path / "ssh2"
    ssh.write_text('#!/bin/sh\nif [ "$1" = -l ]; then shift 2; fi\nshift\nexec "$@"\n')
    ssh.chmod(0o755)
    monkeypatch.setattr(models, "SSH_DIR", tmp_path)
    monkeypatch.setattr(models, "checked_command", lambda host, command: "rsync")
    from ssh_common import transfer

    monkeypatch.setattr(transfer, "checked_command", lambda host, command: "rsync")
    monkeypatch.setattr(models, "is_wsl", lambda: True)
    monkeypatch.setattr(
        models, "discover_models", lambda *a, **kw: ["/not-selected", str(source)]
    )
    answers = iter(["bad", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    options = args(tmp_path)
    assert models.fetch_model(options) == 0
    target = Path(options.output) / source.name
    assert (target / "config.json").read_text() == "{}"
    assert not (target / "model.safetensors").is_symlink()
    assert (target / "model.safetensors").read_bytes() == b"weight fixture"
    (target / "keep").write_text("local data")
    options.model = str(source)
    monkeypatch.setattr(
        models, "discover_models", lambda *a, **kw: pytest.fail("direct mode scanned")
    )
    monkeypatch.setattr(
        "builtins.input", lambda prompt: pytest.fail("direct mode prompted")
    )
    assert models.fetch_model(options) == 0
    assert (target / "keep").read_text() == "local data"


def test_cancel_does_not_create_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "is_wsl", lambda: True)
    monkeypatch.setattr(models.shutil, "which", lambda cmd: "/usr/bin/rsync")
    monkeypatch.setattr(models, "discover_models", lambda *a, **kw: ["/data/model"])
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    assert models.fetch_model(args(tmp_path)) == 0
    assert not (tmp_path / "downloads").exists()


def test_non_wsl_refused_before_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "is_wsl", lambda: False)
    monkeypatch.setattr(
        models, "discover_models", lambda *a, **kw: pytest.fail("unexpected discovery")
    )
    with pytest.raises(ValueError, match="WSL"):
        models.fetch_model(args(tmp_path))
