import importlib.util
import csv
from pathlib import Path
import subprocess
import sys

import pytest

CONTAINERS = Path(__file__).resolve().parents[1] / "devtools/containers"
sys.path.insert(0, str(CONTAINERS))
import common


def load(name):
    spec = importlib.util.spec_from_file_location(name, CONTAINERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        module.next_container_name(["jiangyinzuo-dev2", "other", "jiangyinzuo-dev10"])
        == "jiangyinzuo-dev11"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert module.select_container_name(["jiangyinzuo-dev2"]) == "jiangyinzuo-dev3"


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
    monkeypatch.setattr(sys, "argv", ["create.py"])
    answers = iter(("", ""))
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    args, home, sources, bindfs = module.prepare()

    assert args.image == "vllm-ascend:newest"
    assert args.name == "jiangyinzuo-dev6"
    assert home == str(tmp_path)
    assert sources == [tmp_path]
    assert bindfs == "/usr/bin/bindfs"
    assert calls[-1] == ("sudo", "-n", "true")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "ssh/container-dev/devtools/containers"
    root.mkdir(parents=True)
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common.platform, "release", lambda: "Microsoft-WSL2")
    return root


def args(host=None, npu=None):
    from argparse import Namespace

    return Namespace(host=host, npu_platform=npu)


def test_wsl_requires_explicit_target_and_local_platform(setup):
    with pytest.raises(ValueError, match="--host"):
        common.dispatch(args(), "download.py", [])
    with pytest.raises(ValueError, match="--npu-platform"):
        common.dispatch(args("localhost"), "download.py", [])
    assert common.dispatch(args("localhost", "a3"), "download.py", []) is None


def test_server_local_platform_uses_real_machine_probe(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(common.platform, "release", lambda: "Linux")
    probe_dir = setup.parent
    import shutil
    import os

    source = CONTAINERS.parent / "machine_probe.py"
    shutil.copy2(source, probe_dir / "machine_probe.py")
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
    assert common.dispatch(options, "download.py", []) is None
    assert options.npu_platform == "a3"
    binary.write_text("#!/bin/sh\nprintf 'PCI Device ID : 0xD802\\n'\n")
    with pytest.raises(ValueError, match="无法识别本地"):
        common.dispatch(args(), "download.py", [])


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
    assert (
        common.dispatch(args("suzhou-one"), "download.py", ["--date", "20261012"]) == 7
    )
    assert calls[0][0].endswith("ssh2")
    assert "--npu-platform a3" in calls[0][2]
    with pytest.raises(ValueError, match="没有主机"):
        common.dispatch(args("missing"), "download.py", [])


def test_local_download_publishes_only_successful_archive(setup, monkeypatch):
    module = load("download")
    monkeypatch.setattr(module, "ROOT", setup)
    options = args("localhost", "a3")
    options.date = "20261013"
    options.version = "26.2.0"

    def run(command, **kwargs):
        assert command[0] == "wget"
        assert "/202610/20261013/" in command[-1]
        Path(command[3]).write_bytes(b"archive")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.download(options) == 0
    archives = list(setup.glob("*.tar.gz"))
    assert len(archives) == 1 and archives[0].read_bytes() == b"archive"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: pytest.fail("existing archive downloaded again"),
    )
    assert module.download(options) == 0
