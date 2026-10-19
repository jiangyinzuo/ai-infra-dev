import importlib
import csv
from pathlib import Path
import subprocess
import sys

import pytest

CONTAINERS = Path(__file__).resolve().parents[1] / "devtools/containers"
sys.path.insert(0, str(CONTAINERS.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from containers.core import remote as common
from cli import build_parser
from ssh_common import remote as shared_remote


def load(name):
    return importlib.import_module(f"containers.commands.{name}")


def test_create_interactive_defaults_and_image_selection(monkeypatch, capsys):
    module = load("create")
    rows = (
        "2026-10-16 10:00:00 +0000 UTC\tvllm-ascend:newest",
        "2026-10-15 10:00:00 +0000 UTC\tvllm-ascend:older",
    )

    def run(*command, **kwargs):
        assert command[:3] == ("docker", "images", "vllm-ascend")
        return subprocess.CompletedProcess(command, 0, "\n".join(rows))

    monkeypatch.setattr(module, "run", run)
    answers = iter(("invalid", "2"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert module.select_image() == "vllm-ascend:older"
    assert "1) vllm-ascend:newest" in capsys.readouterr().out

    assert (
        module.next_container_name(
            ["jiangyinzuo-dev2", "other", "jiangyinzuo-dev10"], mapped=False
        )
        == "jiangyinzuo-dev11"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert (
        module.select_container_name(["jiangyinzuo-bindfs-dev2"])
        == "jiangyinzuo-bindfs-dev3"
    )


def test_create_image_selection_rejects_empty_image_list(monkeypatch):
    module = load("create")
    monkeypatch.setattr(
        module,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, ""),
    )
    with pytest.raises(ValueError, match="没有可用的 vllm-ascend 镜像"):
        module.select_image()


def test_create_prepare_uses_interactive_defaults(monkeypatch, tmp_path):
    module = load("create")
    monkeypatch.setattr(module, "require_host", lambda: None)
    calls = []

    def run(*command, **kwargs):
        calls.append(command)
        if command[:2] == ("docker", "images"):
            return subprocess.CompletedProcess(
                command, 0, "2026-10-16 10:00:00 +0000 UTC\tvllm-ascend:newest"
            )
        if command[:3] == ("docker", "container", "ls"):
            return subprocess.CompletedProcess(
                command, 0, "jiangyinzuo-dev2\njiangyinzuo-dev5\n"
            )
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/bindfs")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    answers = iter(("", ""))
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    args, home, sources, bindfs = module.prepare(
        build_parser().parse_args(["containers", "create"])
    )

    assert args.image == "vllm-ascend:newest"
    assert args.name == "jiangyinzuo-bindfs-dev1"
    assert home == str(tmp_path)
    assert sources[0] == tmp_path
    assert Path("/tmp") in sources and Path("/mnt") in sources
    assert bindfs == "/usr/bin/bindfs"
    assert calls[-1] == ("sudo", "-n", "true")


def test_direct_create_needs_neither_bindfs_nor_sudo(monkeypatch, tmp_path):
    module = load("create")
    monkeypatch.setattr(module, "require_host", lambda: None)
    monkeypatch.setattr(module, "shm_capacity", lambda: 1024**3)
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    def run(*command, **kwargs):
        if command[0] != "docker":
            pytest.fail("direct mode requires a non-Docker command")
        if command[:3] == ("docker", "image", "inspect"):
            return subprocess.CompletedProcess(
                command, 0, '[{"Id":"sha256:test","RepoTags":[]}]'
            )
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(
        module, "discover_devices", lambda: [{"path": "/dev/davinci0", "gid": 1000}]
    )
    options = build_parser().parse_args(
        ["containers", "create", "--image", "image", "--name", "test", "--no-bindfs"]
    )
    options, home, sources, bindfs = module.prepare(options)
    assert bindfs is None
    original = tmp_path / "keep"
    original.write_text("user data")

    def fail_start(*command, **kwargs):
        if command[:2] == ("docker", "start"):
            raise subprocess.CalledProcessError(1, command)
        return run(*command, **kwargs)

    monkeypatch.setattr(module, "run", fail_start)
    with pytest.raises(subprocess.CalledProcessError):
        module.create_container(options, home, sources, bindfs)
    assert original.read_text() == "user data"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "ssh/container-dev/devtools/containers"
    root.mkdir(parents=True)
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "is_wsl", lambda: True)
    from workflow import images

    monkeypatch.setattr(images, "is_wsl", lambda: True)
    return root


def args(host=None, npu=None):
    from argparse import Namespace

    return Namespace(host=host, npu_platform=npu)


def test_wsl_requires_explicit_target_and_local_platform(setup):
    with pytest.raises(ValueError, match="--host"):
        common.dispatch(args(), "download", [])
    with pytest.raises(ValueError, match="--npu-platform"):
        common.dispatch(args("localhost"), "download", [])
    assert common.dispatch(args("localhost", "a3"), "download", []) is None


def test_server_local_platform_uses_real_machine_probe(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(common, "is_wsl", lambda: False)
    import os

    binary = tmp_path / "npu-smi"
    sudo = tmp_path / "sudo"
    sudo.write_text('#!/bin/sh\n[ "$1" = -n ] || exit 2\nshift\nexec "$@"\n')
    sudo.chmod(0o755)
    fixture = CONTAINERS.parents[2] / "machine-info/smoketest/fixtures/npu-board-a3.txt"
    import shlex

    binary.write_text(
        '#!/bin/sh\n[ "$*" = "info -t board -i 0" ] || exit 2\ncat '
        + shlex.quote(str(fixture))
        + "\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    options = args()
    assert common.dispatch(options, "download", []) is None
    assert options.npu_platform == "a3"
    binary.write_text("#!/bin/sh\nprintf 'PCI Device ID : 0xD802\\n'\n")
    with pytest.raises(ValueError, match="无法识别本地"):
        common.dispatch(args(), "download", [])


def test_remote_uses_inventory_platform_and_safe_arguments(setup, monkeypatch):
    directory = setup.parents[2] / "machine-info"
    directory.mkdir()
    with (directory / "machines-output.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Host", "OS", "Arch", "NPUPlatform"])
        writer.writerow(["suzhou-one", "openEuler", "aarch64", "a3"])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shared_remote, "local_commit", lambda: "a" * 40)
    assert common.dispatch(args("suzhou-one"), "download", ["--date", "20261012"]) == 7
    assert calls[0][0].endswith("ssh2")
    assert "--npu-platform a3" in calls[0][2]
    import shlex

    remote_command = shlex.split(calls[0][2].rsplit("; fi; ", 1)[1])
    entrypoint = next(
        index for index, token in enumerate(remote_command) if token.endswith("/cli.py")
    )
    assert remote_command[entrypoint + 1 :] == [
        "--no-hints",
        "containers",
        "download",
        "--npu-platform",
        "a3",
        "--date",
        "20261012",
    ]
    with pytest.raises(ValueError, match="没有主机"):
        common.dispatch(args("missing"), "download", [])


@pytest.mark.parametrize(
    "subcommand, options",
    [
        ("create", ["--image", "image with spaces", "--name", "jiangyinzuo-dev1"]),
        ("delete", ["--name", "jiangyinzuo-dev1", "--yes"]),
        ("download", ["--date", "20261012"]),
        ("ls", ["--json"]),
    ],
)
def test_remote_dispatch_places_subcommand_before_options(
    setup, monkeypatch, subcommand, options
):
    directory = setup.parents[2] / "machine-info"
    directory.mkdir()
    with (directory / "machines-output.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Host", "OS", "Arch", "NPUPlatform"])
        writer.writerow(["suzhou-one", "openEuler", "aarch64", "a3"])
    calls = []

    def run_remote_cli(host, arguments):
        calls.append((host, arguments))
        return 7

    monkeypatch.setattr(common, "run_remote_cli", run_remote_cli)
    assert common.dispatch(args("suzhou-one"), subcommand, options) == 7
    assert calls == [
        (
            "suzhou-one",
            ["containers", subcommand, "--npu-platform", "a3", *options],
        )
    ]


def test_local_download_publishes_only_successful_archive(setup, monkeypatch):
    from workflow import images as module

    monkeypatch.setattr(module, "DATA_ROOT", setup)
    options = args("localhost", "a3")
    options.archive = "vllm-ascend:dev-26.2.0.B109-day20261013-A3.tar.gz"

    def run(command, **kwargs):
        assert command[0] == "wget"
        assert "/202610/20261013/" in command[-1]
        Path(command[3]).write_bytes(b"archive")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.download(options) == 0
    archives = list((setup / "containers").glob("*.tar.gz"))
    assert len(archives) == 1 and archives[0].read_bytes() == b"archive"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: pytest.fail("existing archive downloaded again"),
    )
    assert module.download(options) == 0
