import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from containers.commands import create


@pytest.mark.parametrize(
    "reference,tags,expected",
    [
        ("image-id", ["vllm:openEuler24.03"], True),
        ("vllm:ubuntu", ["vllm:ubuntu", "vllm:openEuler"], False),
        ("image-id", [], False),
    ],
)
def test_inspected_tags_select_distribution(monkeypatch, reference, tags, expected):
    monkeypatch.setattr(
        create,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0, json.dumps([{"Id": "sha256:fixed", "RepoTags": tags}])
        ),
    )
    image = create.Image.inspect(reference)
    assert image.id == "sha256:fixed" and image.openeuler == expected


def test_identity_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(
        create,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0, json.dumps([0, 0, "/root", "/root", "/root"])
        ),
    )
    with pytest.raises(ValueError, match="校验失败"):
        create.verify_identity("container", "/home/user")


def test_no_devices_refused_before_mount_or_create(monkeypatch, tmp_path):
    from argparse import Namespace

    monkeypatch.setattr(
        create.Image, "inspect", lambda ref: create.Image("sha256:fixed", False)
    )
    monkeypatch.setattr(create, "discover_devices", lambda: [])
    monkeypatch.setattr(
        create, "run", lambda *a, **kw: pytest.fail("must not create resources")
    )
    with pytest.raises(ValueError, match="没有发现"):
        create._create_container(
            Namespace(image="image", name="name"),
            str(tmp_path),
            [tmp_path],
            "/bin/bindfs",
            None,
        )


def test_create_local_platform_validation_precedes_handler(monkeypatch):
    from cli import build_parser
    from containers import interface
    from containers.core import remote

    monkeypatch.setattr(interface, "is_wsl", lambda: False)
    monkeypatch.setattr(remote, "is_wsl", lambda: False)
    monkeypatch.setattr(remote, "detect_local_platform", lambda: "a3")
    args = build_parser().parse_args(["containers", "create"])
    seen = []
    args.container_handler = lambda options: seen.append(options.npu_platform)
    interface.execute(args)
    assert seen == ["a3"]
    monkeypatch.setattr(
        remote,
        "detect_local_platform",
        lambda: pytest.fail("explicit platform should skip probe"),
    )
    args.npu_platform = "a3"
    interface.execute(args)
    assert seen == ["a3", "a3"]
