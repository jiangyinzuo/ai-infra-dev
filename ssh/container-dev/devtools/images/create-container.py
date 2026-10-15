#!/usr/bin/env python3
"""在昇腾服务器创建开发容器，不启动容器。"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys

from container_common import add_target_args, dispatch


def docker_output(*args):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=True
    ).stdout


@dataclass(frozen=True)
class Image:
    id: str
    tags: tuple[str, ...]

    @classmethod
    def inspect(cls, reference):
        data = json.loads(docker_output("image", "inspect", reference))[0]
        return cls(data["Id"], tuple(data.get("RepoTags") or ()))

    def is_openeuler(self, reference):
        names = (reference,) if reference in self.tags else self.tags
        return any("openeuler" in name.lower() for name in names)


def create(args):
    if "microsoft" in platform.release().lower():
        raise ValueError("禁止在 WSL 本机创建容器，请指定远程 --host")
    image = args.image or os.environ.get("IMAGE_NAME")
    if not image:
        rows = sorted(
            docker_output(
                "images",
                "vllm-ascend",
                "--format",
                "{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}",
            ).splitlines(),
            reverse=True,
        )
        images = [row.split("\t", 1)[1] for row in rows]
        if not images:
            raise ValueError("没有可用的 vllm-ascend 镜像")
        for index, item in enumerate(images, 1):
            print(f"{index}) {item}")
        while True:
            choice = input("选择镜像编号 [1，最新]: ") or "1"
            if choice.isdigit() and 1 <= int(choice) <= len(images):
                image = images[int(choice) - 1]
                break
            print("请输入有效编号")
    reference = image
    image = Image.inspect(reference)
    name = args.name or os.environ.get("CONTAINER_NAME")
    if not name:
        names = docker_output("ps", "-a", "--format", "{{.Names}}")
        numbers = [
            int(match[1])
            for line in names.splitlines()
            if (match := re.fullmatch(r"jiangyinzuo-dev([0-9]{1,9})", line))
        ]
        default = f"jiangyinzuo-dev{max(numbers, default=0) + 1}"
        name = input(f"容器名 [{default}]: ") or default
    home = str(Path.home())
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--net=host",
        "--shm-size=500g",
        "--privileged=true",
    ]
    for device in [
        *(f"/dev/davinci{i}" for i in range(16)),
        "/dev/davinci_manager",
        "/dev/devmm_svm",
        "/dev/hisi_hdc",
    ]:
        command += ["--device", device]
    command += [
        "--hostname",
        socket.gethostname() + "-container",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-w",
        home,
    ]
    for value in [
        "PIP_INDEX_URL=https://mirrors.tools.huawei.com/pypi/simple",
        "PIP_EXTRA_INDEX_URL=",
        "PIP_TRUSTED_HOST=mirrors.tools.huawei.com",
        f"HOME={home}",
        f"USER={os.environ.get('USER', '')}",
    ]:
        command += ["-e", value]
    # 仅传变量名，避免密钥值出现在 Docker 命令行中。
    for key in (
        "SUDOCODE_API_KEY",
        "OPENAI_API_KEY",
        "USG_IP",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        if key in os.environ:
            command += ["-e", key]
    volumes = [
        home,
        "/data",
        "/mnt",
        "/tmp",
        "/usr/local/dcmi",
        "/usr/local/Ascend/driver",
        "/usr/local/bin/npu-smi",
        "/etc/ascend_install.info",
        "/etc/hccn.conf",
    ]
    volumes += [path for path in ("/data1", "/data2", "/data3") if Path(path).is_dir()]
    for path in volumes:
        command += ["-v", f"{path}:{path}"]
    print(f"创建容器 {name}，镜像 {reference} ({image.id})", flush=True)
    subprocess.run([*command, "-it", image.id, "bash"], check=True)
    if image.is_openeuler(reference):
        subprocess.run(
            [
                "docker",
                "cp",
                f"{home}/devtools/openeuler24.03-lts-aarch64/openEuler.repo",
                f"{name}:/etc/yum.repos.d/openEuler.repo",
            ],
            check=True,
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_args(parser)
    parser.add_argument("--image", help="镜像名，默认交互选择")
    parser.add_argument("--name", help="容器名，默认交互输入")
    args = parser.parse_args()
    try:
        options = []
        for key in ("image", "name"):
            value = getattr(args, key) or os.environ.get(
                {"image": "IMAGE_NAME", "name": "CONTAINER_NAME"}[key]
            )
            if value:
                options += ["--" + key, value]
        remote = dispatch(args, "create-container.py", options)
        return create(args) if remote is None else remote
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    sys.exit(main())
