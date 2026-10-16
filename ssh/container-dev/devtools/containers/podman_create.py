#!/usr/bin/env python3
"""创建 rootless Podman NPU 开发容器：宿主机当前用户映射为容器 root。"""

import argparse
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shlex
import socket
import subprocess
import sys
import tempfile

from common import add_target_args, dispatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONTAINER_ENV_KEYS
from npu_devices import discover_devices


def run(*args, **kwargs):
    command = ["podman", *args]
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def output(*args):
    return run(*args, capture_output=True, text=True).stdout


def verify(container, home):
    if output("exec", container, "id", "-u").strip() != "0":
        raise ValueError("容器默认用户不是 root")
    # 容器创建可执行文件；宿主机验证所有权、读写及执行，始终清理探针。
    with tempfile.TemporaryDirectory(prefix=".podman-home-", dir=home) as directory:
        probe = Path(directory) / "probe"
        run(
            "exec",
            container,
            "sh",
            "-c",
            'umask 077; printf "#!/bin/sh\\nexit 0\\n" > "$1"; chmod 700 "$1"',
            "sh",
            str(probe),
        )
        if probe.stat().st_uid != os.getuid():
            raise ValueError("容器 root 创建文件未映射为宿主机当前 UID")
        with probe.open("a") as stream:
            stream.write("# host write verified\n")
        subprocess.run([str(probe)], check=True)
    result = output("exec", container, "npu-smi", "info")
    print(result, flush=True)
    if "NPU" not in result or "failed" in result.lower():
        raise ValueError(
            "npu-smi 未成功；rootless root 不具备宿主机 sudo 权限。容器保留供诊断，不自动提权。"
        )


def create(args):
    if "microsoft" in platform.release().lower() or os.getuid() == 0:
        raise ValueError("必须在服务器以普通用户执行，不要 sudo；WSL 请指定远程 --host")
    info = json.loads(output("info", "--format", "json"))
    if not info["host"]["security"]["rootless"]:
        raise ValueError("需要本机 rootless Podman")
    image = args.image or os.environ.get("IMAGE_NAME")
    if not image:
        rows = sorted(
            output(
                "images", "--format", "{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}"
            ).splitlines(),
            reverse=True,
        )
        images = [
            row.split("\t", 1)[1]
            for row in rows
            if row.split("\t", 1)[1].split(":")[0].endswith("/vllm-ascend")
            or row.split("\t", 1)[1].startswith("vllm-ascend:")
        ]
        if not images:
            raise ValueError("没有可用的 vllm-ascend 镜像，请先 podman load")
        for index, item in enumerate(images, 1):
            print(f"{index}) {item}")
        while True:
            choice = input("选择镜像编号 [1，最新]: ") or "1"
            if choice.isdigit() and 1 <= int(choice) <= len(images):
                image = images[int(choice) - 1]
                break
    image_data = json.loads(output("image", "inspect", image))[0]
    username = pwd.getpwuid(os.getuid()).pw_name
    name = args.name or os.environ.get("CONTAINER_NAME")
    if not name:
        names = output("ps", "-a", "--format", "{{.Names}}")
        numbers = [
            int(match[1])
            for line in names.splitlines()
            if (match := re.fullmatch(re.escape(username) + r"-dev([0-9]{1,9})", line))
        ]
        default = f"{username}-dev{max(numbers, default=0) + 1}"
        name = input(f"容器名 [{default}]: ") or default
    home = str(Path.home())
    devices = discover_devices()
    if not devices:
        raise ValueError("没有发现昇腾 NPU 字符设备")
    command = [
        "create",
        "--name",
        name,
        "--network=host",
        "--cgroups=disabled",
        "--userns=host",
        "--user=0:0",
        "--group-add=keep-groups",
        "--security-opt=label=disable",
        "--hostname",
        socket.gethostname() + "-container",
        "--shm-size=500g",
        "--workdir",
        home,
    ]
    for device in devices:
        command += ["--device", device["path"]]
    for value in [
        f"HOME={home}",
        "USER=root",
        "LOGNAME=root",
        "PIP_INDEX_URL=https://mirrors.tools.huawei.com/pypi/simple",
        "PIP_EXTRA_INDEX_URL=",
        "PIP_TRUSTED_HOST=mirrors.tools.huawei.com",
    ]:
        command += ["--env", value]
    for key in CONTAINER_ENV_KEYS:
        if key in os.environ:
            command += ["--env", key]
    for path in [
        home,
        "/data",
        "/data1",
        "/data2",
        "/data3",
        "/mnt",
        "/tmp",
        "/usr/local/dcmi",
        "/usr/local/Ascend/driver",
        "/usr/local/bin/npu-smi",
        "/etc/ascend_install.info",
        "/etc/hccn.conf",
    ]:
        if Path(path).exists():
            command += ["--volume", f"{path}:{path}"]
    for part in ("bin", "lib"):
        command += ["--mount", f"type=volume,dst={home}/.local/{part},volume-nocopy"]
    run(*command, "--entrypoint", "/bin/bash", "-it", image_data["Id"])
    run("start", name)
    if "openeuler" in image.lower():
        repo = Path(home) / "devtools/openeuler24.03-lts-aarch64/openEuler.repo"
        run(
            "exec",
            "-i",
            name,
            "sh",
            "-c",
            "mkdir -p /etc/yum.repos.d && cat > /etc/yum.repos.d/openEuler.repo",
            input=repo.read_text(),
            text=True,
        )
        run(
            "exec",
            name,
            "yum",
            "--disablerepo=debuginfo,source,update-source",
            "install",
            "-y",
            "nmap-ncat",
        )
    verify(name, home)
    print(
        f"容器 {name} 已启动；root 映射为宿主机 {username}，共享 HOME 文件读写/执行及 npu-smi 已验证。"
    )
    for command in (
        ["podman", "exec", "-it", name, "bash"],
        ["podman", "stop", name],
        ["podman", "rm", "-v", name],
    ):
        print(shlex.join(command))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_args(parser)
    parser.add_argument("--image", help="镜像名，默认交互选择")
    parser.add_argument("--name", help="容器名，默认交互输入")
    args = parser.parse_args()
    try:
        options = []
        for key, env in (("image", "IMAGE_NAME"), ("name", "CONTAINER_NAME")):
            value = getattr(args, key) or os.environ.get(env)
            if value:
                options += ["--" + key, value]
        remote = dispatch(args, "podman_create.py", options)
        return create(args) if remote is None else remote
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
