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


def test_create_rejects_wsl_before_docker(setup):
    module = load("create")
    with pytest.raises(ValueError, match="禁止"):
        module.create(args("localhost", "a3"))


@pytest.mark.parametrize(
    "image,tags,copy_repo",
    [
        ("vllm-ascend:openEuler24.03", ["vllm-ascend:openEuler24.03"], True),
        ("ubuntu:24.04", ["ubuntu:24.04"], False),
        ("9824ebc97c4d", ["vllm-ascend:openEuler24.03"], True),
        ("ddcd7eeb5e02", ["vllm-ascend:Ubuntu24.04"], False),
        ("untagged-id", [], False),
        ("ubuntu:24.04", ["ubuntu:24.04", "custom:openEuler"], False),
    ],
)
def test_create_copies_repo_only_for_openeuler(
    setup, monkeypatch, image, tags, copy_repo
):
    module = load("create")
    monkeypatch.setattr(module.platform, "release", lambda: "Linux")
    monkeypatch.setattr(module, "initialize_user", lambda *args: None)
    monkeypatch.setattr(module, "configure_sudo", lambda *args: None)
    monkeypatch.setattr(
        module,
        "discover_devices",
        lambda: [
            {"path": "/dev/davinci0", "gid": 4321},
            {"path": "/dev/davinci_manager", "gid": 7654},
            {"path": "/dev/devmm_svm", "gid": 4321},
        ],
    )
    import json

    monkeypatch.setattr(
        module,
        "docker_output",
        lambda *args: (
            json.dumps([{"Id": "sha256:resolved", "RepoTags": tags}])
            if args[0] == "image"
            else module.pwd.getpwuid(module.os.getuid()).pw_name
        ),
    )
    options = args("localhost", "a3")
    options.image = image
    options.name = "smoke-container"
    monkeypatch.setattr(module.Path, "read_text", lambda path, **kwargs: "[smoke]\n")
    inherited = (
        "SUDOCODE_API_KEY",
        "OPENAI_API_KEY",
        "SOC_VERSION",
        "USG_IP",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    )
    for key in inherited:
        monkeypatch.setenv(key, f"smoke-value-{key}")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.create(options) == 0
    assert calls[0][:2] == ["docker", "create"]
    assert [
        calls[0][i + 1]
        for i, value in enumerate(calls[0][:-1])
        if value == "--group-add"
    ] == ["0", "4321", "7654"]
    assert calls[0][-2] == "sha256:resolved"
    environment = [
        calls[0][i + 1] for i, value in enumerate(calls[0][:-1]) if value == "-e"
    ]
    assert set(inherited).issubset(environment)
    assert not any("smoke-value-" in value for value in calls[0])
    assert calls[1] == ["docker", "start", "smoke-container"]
    paths = [str(module.Path.home() / ".local" / part) for part in ("bin", "lib")]
    mounts = [
        calls[0][i + 1] for i, value in enumerate(calls[0][:-1]) if value == "--mount"
    ]
    assert mounts == [f"type=volume,dst={path},volume-nocopy" for path in paths]
    assert calls[-1] == [
        "docker",
        "exec",
        "-u",
        "0",
        "smoke-container",
        "chown",
        f"{module.os.getuid()}:{module.os.getgid()}",
        *paths,
    ]
    assert len(calls) == (5 if copy_repo else 3)
    if copy_repo:
        assert calls[2][:6] == ["docker", "exec", "-i", "-u", "0", "smoke-container"]


def test_container_account_files_preserve_image_users(setup, monkeypatch, tmp_path):

    module = load("create")
    etc = tmp_path / "etc"
    etc.mkdir()
    (etc / "passwd").write_text("root:x:0:0:root:/root:/bin/bash\n")
    (etc / "group").write_text("root:x:0:\n")

    def docker_exec(command, **kwargs):
        assert command[:2] == ["docker", "exec"]
        path = etc / command[-1].rsplit("/", 1)[-1]
        if "cat" in command:
            return subprocess.CompletedProcess(command, 0, stdout=path.read_text())
        path.write_text(kwargs["input"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", docker_exec)
    module.initialize_user("smoke", "jiangyinzuo", 9011, 9012, "/data3/jiangyinzuo")
    expected = "root:x:0:0:root:/root:/bin/bash\njiangyinzuo:!:9011:9012::/data3/jiangyinzuo:/bin/bash\n"
    assert (etc / "passwd").read_text() == expected
    assert "jiangyinzuo:x:9012:" in (etc / "group").read_text()
    module.initialize_user("smoke", "jiangyinzuo", 9011, 9012, "/data3/jiangyinzuo")
    assert (etc / "passwd").read_text() == expected
    with pytest.raises(ValueError, match="已属于"):
        module.initialize_user("smoke", "other", 9011, 9012, "/data3/other")
    assert (etc / "passwd").read_text() == expected
