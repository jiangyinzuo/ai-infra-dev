#!/usr/bin/env python3
"""Rootful Docker with bindfs-mapped writable directories (Python 3.9+)."""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def run(*args: str, **kwargs: Any) -> subprocess.CompletedProcess:
    print(shlex.join(args), flush=True)
    return subprocess.run(list(args), check=True, **kwargs)


def prepare() -> tuple[argparse.Namespace, str, list[Path], str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Docker 镜像名或 ID")
    parser.add_argument("--name", required=True, help="新容器名称")
    parser.add_argument(
        "--share",
        action="append",
        default=[],
        help="额外可写共享目录，容器内保持相同路径，可重复指定",
    )
    parser.add_argument(
        "--test-home",
        help="使用隔离目录代替真实 HOME，供验证使用",
    )
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error("Run as the owning ordinary user, not sudo")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", args.name):
        parser.error("Invalid container name")
    bindfs = shutil.which("bindfs")
    if not bindfs:
        parser.error("bindfs is required on the host")
    home = str(Path.home())
    sources = [Path(args.test_home or home).resolve()] + [
        Path(p).resolve() for p in args.share
    ]
    for source in sources:
        if not source.is_dir() or not os.access(source, os.W_OK):
            parser.error(
                "Source must be an existing writable directory: " + str(source)
            )
    if (
        run(
            "docker",
            "container",
            "ls",
            "-a",
            "--format",
            "{{.Names}}",
            capture_output=True,
            text=True,
        )
        .stdout.splitlines()
        .count(args.name)
    ):
        parser.error("Container name already exists")
    run("sudo", "-n", "true")
    return args, home, sources, bindfs


def mount_directory(bindfs: str, source: Path, target: Path, destination: str) -> None:
    target.mkdir()
    print(yellow(f"映射目录：{source} → {target} → 容器 {destination}"), flush=True)
    run(
        "sudo",
        "-n",
        bindfs,
        "--map={}/0:@{}/@0".format(os.getuid(), os.getgid()),
        "--create-for-user=" + str(os.getuid()),
        "--create-for-group=" + str(os.getgid()),
        "--chown-ignore",
        "--chgrp-ignore",
        "-o",
        "allow_other,nosuid,nodev",
        str(source),
        str(target),
    )


def docker_command(name: str, home: str) -> list[str]:
    return [
        "docker",
        "create",
        "--name",
        name,
        "--user",
        "0:0",
        "--userns=host",
        "--network=host",
        "--privileged",
        "--security-opt",
        "label=disable",
        "--workdir",
        home,
        "--env",
        "HOME=" + home,
        "--env",
        "USER=root",
        "--env",
        "LOGNAME=root",
    ]


def runtime_mounts(home: str) -> list[str]:
    options: list[str] = []
    for device in sorted(Path("/dev").iterdir()):
        if (
            re.fullmatch(r"davinci(?:[0-9]+|_manager)|devmm_svm|hisi_hdc", device.name)
            and device.is_char_device()
        ):
            options += ["--device", str(device)]
    for path in [
        "/usr/local/Ascend/driver",
        "/usr/local/dcmi",
        "/usr/local/bin/npu-smi",
        "/etc/ascend_install.info",
        "/etc/hccn.conf",
    ]:
        if Path(path).exists():
            options += ["--volume", path + ":" + path + ":ro"]
    # Python tools remain container-private, not shared with host Python.
    for part in ["bin", "lib"]:
        options += [
            "--mount",
            "type=volume,dst=" + home + "/.local/" + part + ",volume-nocopy",
        ]
    return options


def initialize_home(name: str, home: str) -> None:
    script = """import pathlib
import sys

path = pathlib.Path('/etc/passwd')
rows = [line.split(':') for line in path.read_text().splitlines()]
for row in rows:
    if row[0] == 'root':
        row[5] = sys.argv[1]
path.write_text(''.join(':'.join(row) + '\\n' for row in rows))
"""
    run("docker", "exec", name, "python3", "-c", script, home)
    run("docker", "exec", name, "mkdir", "-p", home + "/ascend/log")


def print_instructions(name: str, home: str, root: Path, mounted: list[Path]) -> None:
    print(green(f"容器 {name} 已启动，用户 root，HOME={home}"), flush=True)
    print(f"bindfs 挂载目录：{root}")
    print(yellow("进入容器："))
    print(shlex.join(["docker", "exec", "-it", name, "bash"]))
    print(yellow("删除容器及独立 volume："))
    print(shlex.join(["docker", "stop", name]))
    print(shlex.join(["docker", "rm", "-v", name]))
    print(yellow("删除容器后，卸载映射并清理空挂载目录："))
    for target in reversed(mounted):
        print(shlex.join(["sudo", "umount", str(target)]))
    print(shlex.join(["rmdir", *map(str, mounted), str(root)]))


def create_container(
    args: argparse.Namespace, home: str, sources: list[Path], bindfs: str
) -> None:
    # Outside HOME: prevent recursively mounting the mirror inside its source.
    root = Path(tempfile.mkdtemp(prefix="root-mapped-", dir="/var/tmp"))
    mounted: list[Path] = []
    created = False
    try:
        print(green(f"创建容器 {args.name}，镜像 {args.image}"), flush=True)
        print(
            f"容器 root → 宿主机 UID/GID {os.getuid()}:{os.getgid()}；HOME={home}",
            flush=True,
        )
        command = docker_command(args.name, home)
        for index, source in enumerate(sources):
            target = root / str(index)
            destination = home if index == 0 else str(source)
            mount_directory(bindfs, source, target, destination)
            mounted.append(target)
            command += ["--volume", "{}:{}".format(target, destination)]
        command += runtime_mounts(home)
        run(*command, "--entrypoint", "/bin/bash", "-it", args.image)
        created = True
        run("docker", "start", args.name)
        initialize_home(args.name, home)
        print_instructions(args.name, home, root, mounted)
    except BaseException:
        print(yellow("创建失败或中断，正在清理本次容器和 bindfs 挂载。"), flush=True)
        if created:
            run("docker", "rm", "-f", "-v", args.name)
        for target in reversed(mounted):
            run("sudo", "-n", "umount", str(target))
        shutil.rmtree(root)
        raise


def main() -> None:
    create_container(*prepare())


if __name__ == "__main__":
    main()
