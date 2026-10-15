#!/usr/bin/env python3
"""在昇腾服务器创建并启动开发容器。"""

import argparse
from dataclasses import dataclass
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
from config import ENV_KEYS


def green(text):
    return f"\033[32m{text}\033[0m"


def run_docker(command, **kwargs):
    print(shlex.join(command), flush=True)
    return subprocess.run(command, **kwargs)


def docker_output(*args):
    return run_docker(
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


def initialize_user(container, username: str, uid, gid, home):
    """只更新容器账号文件，不启动容器或修改挂载目录。"""
    with tempfile.TemporaryDirectory(prefix="container-user-") as directory:
        directory = Path(directory)
        for filename in ("passwd", "group"):
            run_docker(
                [
                    "docker",
                    "cp",
                    f"{container}:/etc/{filename}",
                    str(directory / filename),
                ],
                check=True,
            )
        passwd = directory / "passwd"
        group = directory / "group"
        users = [
            line.split(":")
            for line in passwd.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        groups = [
            line.split(":")
            for line in group.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        named = next((entry for entry in users if entry[0] == username), None)
        owner = next((entry for entry in users if entry[2] == str(uid)), None)
        if named and named[2] != str(uid):
            raise ValueError(f"容器用户名 {username} 已被其他 UID 占用")
        if owner and owner[0] != username:
            raise ValueError(f"容器 UID {uid} 已属于 {owner[0]}，不能覆盖")
        if named and (named[3] != str(gid) or named[5] != home):
            raise ValueError(f"容器账号 {username} 的 GID/home 与宿主机不一致")
        changed = []
        if not any(entry[2] == str(gid) for entry in groups):
            if any(entry[0] == username for entry in groups):
                raise ValueError(f"容器组名 {username} 已被其他 GID 占用")
            group.write_text(
                group.read_text().rstrip("\n") + f"\n{username}:x:{gid}:\n"
            )
            changed.append(group)
        if named is None:
            # 禁止密码登录；不依赖镜像是否提供 useradd，也不复制宿主机账号库。
            passwd.write_text(
                passwd.read_text().rstrip("\n")
                + f"\n{username}:!:{uid}:{gid}::{home}:/bin/bash\n"
            )
            changed.append(passwd)
        for path in changed:
            path.chmod(0o644)
            run_docker(
                ["docker", "cp", str(path), f"{container}:/etc/{path.name}"], check=True
            )


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
            choice = input(green("选择镜像编号 [1，最新]: ")) or "1"
            if choice.isdigit() and 1 <= int(choice) <= len(images):
                image = images[int(choice) - 1]
                break
            print(green("请输入有效编号"))
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
        name = input(green(f"容器名 [{default}]: ")) or default
    home = str(Path.home())
    username = pwd.getpwuid(os.getuid()).pw_name
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
        f"USER={username}",
    ]:
        command += ["-e", value]
    # 仅传变量名，避免密钥值出现在 Docker 命令行中。
    for key in ENV_KEYS:
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
    print(green(f"创建容器 {name}，镜像 {reference} ({image.id})"), flush=True)
    run_docker([*command, "-it", image.id, "bash"], check=True)
    initialize_user(name, username, os.getuid(), os.getgid(), home)
    if image.is_openeuler(reference):
        run_docker(
            [
                "docker",
                "cp",
                f"{home}/devtools/openeuler24.03-lts-aarch64/openEuler.repo",
                f"{name}:/etc/yum.repos.d/openEuler.repo",
            ],
            check=True,
        )
    run_docker(["docker", "start", name], check=True)
    actual_user = docker_output("exec", name, "id", "-un").strip()
    if actual_user != username:
        raise ValueError(
            f"容器 {name} 用户校验失败：期望 {username}，实际 {actual_user!r}"
        )
    print(green(f"容器 {name} 已启动，用户 {username}"), flush=True)
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
        remote = dispatch(args, "create.py", options)
        return create(args) if remote is None else remote
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    sys.exit(main())
