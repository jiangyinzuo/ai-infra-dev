import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[1] / "devtools"),
]
from workflow import images


def test_server_download_reuses_archive_and_dated_image(monkeypatch, tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    state = tmp_path / "state"
    docker = commands / "docker"
    docker.write_text(
        "#!"
        + sys.executable
        + """
import sys,os,pathlib
p=pathlib.Path(os.environ['DOCKER_STATE'])
if sys.argv[1]=='images': print(p.read_text() if p.exists() else '')
elif sys.argv[1]=='load':
 assert pathlib.Path(sys.argv[3]).read_bytes()==b'fixture'
 p.write_text('quay.io/ascend/vllm-ascend:nightly-main-a3-openeuler')
elif sys.argv[1]=='tag':
 p.write_text(p.read_text()+'\\n'+sys.argv[3])
"""
    )
    docker.chmod(0o755)
    wget = commands / "wget"
    wget.write_text('#!/bin/sh\nprintf fixture > "$3"\n')
    wget.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands) + ":" + os.environ["PATH"])
    monkeypatch.setenv("DOCKER_STATE", str(state))
    monkeypatch.setattr(images, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(images, "is_wsl", lambda: False)
    args = SimpleNamespace(archive="nightly-main-a3-openeuler.tar.gz", source="quay")
    assert images.download(args) == 0
    _, saved = images.download_location(args.archive, args.source)
    expected = images.image_from_archive(saved)
    assert expected in state.read_text().splitlines()
    archive = tmp_path / "containers" / saved
    archive.unlink()
    wget.write_text("#!/bin/sh\nexit 89\n")
    assert images.download(args) == 0  # Docker hit: no download needed.
    state.unlink()
    archive.write_bytes(b"fixture")
    assert images.download(args) == 0  # Archive hit: load without wget.
    assert expected in state.read_text().splitlines()
