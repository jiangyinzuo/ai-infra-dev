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


@pytest.mark.parametrize("home_relative", [False, True])
def test_upload_real_rsync(monkeypatch, tmp_path, home_relative):
    if not shutil.which("rsync"):
        pytest.skip("rsync not installed")
    source = tmp_path / "vllm-ascend_test.tar.gz"
    source.write_bytes(b"image archive fixture")
    destination = tmp_path / "remote with spaces"
    home = tmp_path / "remote-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    if home_relative:
        destination = home / "ai-infra-dev/ssh/container-dev/devtools"
    ssh = tmp_path / "ssh2"
    ssh.write_text(
        '#!/bin/sh\nif [ "$1" = -l ]; then shift 2; fi\nshift\nexec sh -c "$*"\n'
    )
    ssh.chmod(0o755)
    monkeypatch.setattr(images, "SSH_DIR", tmp_path)
    monkeypatch.setattr(
        images,
        "REMOTE_DIR",
        "~/ai-infra-dev/ssh/container-dev/devtools"
        if home_relative
        else str(destination),
    )
    monkeypatch.setattr(images, "checked_command", lambda host, command: command)
    from ssh_common import transfer

    monkeypatch.setattr(transfer, "checked_command", lambda host, command: command)
    monkeypatch.setattr(images, "is_wsl", lambda: True)
    statuses = iter(["missing", "loaded"])
    monkeypatch.setattr(images, "remote_load", lambda host, image: next(statuses))
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


@pytest.mark.parametrize(
    "filename",
    [
        "vllm-ascend:dev-26.2.0.B109-day20261019-A3.tar.gz",
        "vllm-ascend_dev-26.2.0.B109-day20261019-A3.tar.gz",
    ],
)
def test_image_name_formats(filename):
    assert (
        images.image_from_archive(filename)
        == "vllm-ascend:dev-26.2.0.B109-day20261019-A3"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "other_tag.tar.gz",
        "vllm-ascend_.tar.gz",
        "vllm-ascend:bad tag.tar.gz",
        "vllm-ascend:tag.tar",
    ],
)
def test_invalid_archive_rejected(filename):
    with pytest.raises(ValueError):
        images.image_from_archive(filename)


@pytest.mark.parametrize("status", ["image", "loaded"])
def test_remote_existing_image_or_archive_needs_no_local_file(monkeypatch, status):
    monkeypatch.setattr(images, "is_wsl", lambda: True)
    monkeypatch.setattr(images, "remote_load", lambda host, image: status)
    args = build_parser().parse_args(
        [
            "images",
            "upload",
            "--host",
            "suzhou-one",
            "--archive",
            "/missing/vllm-ascend_tag.tar.gz",
        ]
    )
    assert images.upload(args) == 0


def test_remote_probe_loads_alternate_archive_and_verifies_tag(tmp_path):
    import os
    import json

    archive = tmp_path / "vllm-ascend_tag.tar.gz"
    archive.write_bytes(b"fixture")
    state = tmp_path / "docker-state"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!"
        + sys.executable
        + '\nimport os,sys,pathlib\ns=pathlib.Path(os.environ["DOCKER_STATE"])\n'
        + 'if sys.argv[1]=="images":\n print(s.read_text() if s.exists() else "")\n'
        + 'elif sys.argv[1]=="load":\n assert pathlib.Path(sys.argv[3]).name=="vllm-ascend_tag.tar.gz"\n s.write_text("vllm-ascend:tag")\n'
    )
    docker.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + ":" + os.environ["PATH"],
        DOCKER_STATE=str(state),
        PYTHONPATH=str(Path(images.__file__).resolve().parents[1]),
    )

    def probe():
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                images.REMOTE_LOAD,
                str(tmp_path),
                "vllm-ascend:tag",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        events = [json.loads(line) for line in result.stdout.splitlines()]
        if events[-1]["status"] == "loaded":
            assert events[0]["command"].startswith("docker load --input ")
        return events[-1]["status"]

    assert probe() == "loaded"
    archive.unlink()
    assert probe() == "image"
    state.unlink()
    assert probe() == "missing"


def test_quay_nightly_url_and_local_date():
    from datetime import date

    name = "nightly-main-a3-openeuler.tar.gz"
    url, saved = images.download_location(name, "quay")
    assert (
        url
        == "https://cmc-lfg-artifactory.cmc.tools.huawei.com/artifactory/generic-8166/hw-cr/quay.io/ascend/vllm-ascend/"
        + name
    )
    assert (
        saved
        == "nightly-main-a3-openeuler-" + date.today().strftime("%Y%m%d") + ".tar.gz"
    )
    assert (
        images.image_from_archive(saved)
        == "quay.io/ascend/vllm-ascend:nightly-main-a3-openeuler-"
        + date.today().strftime("%Y%m%d")
    )
