#!/usr/bin/env python3
"""Rootful Docker with optional bindfs ownership mapping (Python 3.9+)."""

import argparse
from dataclasses import dataclass
import json
from contextlib import nullcontext
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
from typing import Optional

from ..core.remote import ROOT
from ssh_common.config import HOST_ENV_DEFAULTS, CONTAINER_ENV_KEYS, REMOTE_USER
from ..core.mappings import MappingStore
from ssh_common.npu_devices import discover_devices
from ..core.process import run
from ssh_common.terminal import green, yellow
from ..core.host import require_host, shm_capacity


@dataclass(frozen=True)
class Image:
    id: str
    openeuler: bool

    @classmethod
    def inspect(cls, reference: str):
        data = json.loads(
            run(
                "docker", "image", "inspect", reference, capture_output=True, text=True
            ).stdout
        )[0]
        tags = data.get("RepoTags") or []
        names = [reference] if reference in tags else tags
        return cls(data["Id"], any("openeuler" in tag.lower() for tag in names))


def initialize_image(name: str, image: Image) -> None:
    if not image.openeuler:
        return
    repo = ROOT.parent / "openeuler24.03-lts-aarch64/openEuler.repo"
    run(
        "docker",
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
        "docker",
        "exec",
        name,
        "yum",
        "--disablerepo=debuginfo,source,update-source",
        "install",
        "-y",
        "nmap-ncat",
    )


def verify_identity(name: str, home: str) -> None:
    script = "import os,pwd,json; print(json.dumps([os.getuid(),os.getgid(),os.environ.get('HOME'),pwd.getpwuid(0).pw_dir,os.getcwd()]))"
    actual = json.loads(
        run(
            "docker",
            "exec",
            name,
            "python3",
            "-c",
            script,
            capture_output=True,
            text=True,
        ).stdout
    )
    if actual != [0, 0, home, home, home]:
        raise ValueError(f"容器身份/HOME 校验失败：{actual!r}")


def select_image() -> str:
    rows = sorted(
        run(
            "docker",
            "images",
            "vllm-ascend",
            "--format",
            "{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}",
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        reverse=True,
    )
    images = [row.split("\t", 1)[1] for row in rows if "\t" in row]
    if not images:
        raise ValueError("没有可用的 vllm-ascend 镜像")
    for index, item in enumerate(images, 1):
        print(f"{index}) {item}")
    while True:
        choice = input(green("选择镜像编号 [1，最新]: ")) or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(images):
            return images[int(choice) - 1]
        print(green("请输入有效编号"))


def next_container_name(names: list[str], mapped: bool = True) -> str:
    """生成下一个容器名；mapped=True 使用 bindfs 前缀，否则使用直接挂载前缀。"""
    prefix = f"{REMOTE_USER}-bindfs-dev" if mapped else f"{REMOTE_USER}-dev"
    numbers = [
        int(match.group(1))
        for name in names
        if (match := re.fullmatch(re.escape(prefix) + r"([0-9]{1,9})", name))
    ]
    return f"{prefix}{max(numbers, default=0) + 1}"


def select_container_name(names: list[str], mapped: bool = True) -> str:
    default = next_container_name(names, mapped)
    return input(green(f"容器名 [{default}]: ")) or default


def prepare(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, list[Path], Optional[str]]:
    require_host()
    if os.getuid() == 0:
        raise ValueError("Run as the owning ordinary user, not sudo")
    if not args.image:
        args.image = select_image()
    existing_names = run(
        "docker",
        "container",
        "ls",
        "-a",
        "--format",
        "{{.Names}}",
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not args.name:
        args.name = select_container_name(existing_names, not args.no_bindfs)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", args.name):
        raise ValueError("Invalid container name")
    bindfs = None if args.no_bindfs else shutil.which("bindfs")
    if not args.no_bindfs and not bindfs:
        raise ValueError("bindfs is required on the host")
    home = str(Path.home())
    sources = [Path(args.test_home or home).resolve()] + [
        Path(p).resolve() for p in args.share
    ]
    for source in sources:
        if not source.is_dir() or not os.access(source, os.W_OK):
            raise ValueError(
                "Source must be an existing writable directory: " + str(source)
            )
    # 自动共享目录通常属于管理员；容器 root/挂载 sudo 负责访问，不要求用户可写目录根。
    sources += [
        Path(path)
        for path in ("/data", "/data1", "/data2", "/data3", "/mnt", "/tmp")
        if Path(path).is_dir() and Path(path) not in sources
    ]
    if args.name in existing_names:
        raise ValueError("Container name already exists")
    if bindfs is not None:
        run("sudo", "-n", "true")
    return args, home, sources, bindfs


def docker_command(name: str, home: str) -> list[str]:
    shm_size = shm_capacity()
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--user",
        "0:0",
        "--userns=host",
        "--network=host",
        "--shm-size",
        str(shm_size),
        "--hostname",
        socket.gethostname() + "-container",
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
        "--env",
        "PIP_INDEX_URL=https://mirrors.tools.huawei.com/pypi/simple",
        "--env",
        "PIP_EXTRA_INDEX_URL=",
        "--env",
        "PIP_TRUSTED_HOST=mirrors.tools.huawei.com",
        "--env",
        "NODE_TLS_REJECT_UNAUTHORIZED="
        + os.environ.get(
            "NODE_TLS_REJECT_UNAUTHORIZED",
            HOST_ENV_DEFAULTS["NODE_TLS_REJECT_UNAUTHORIZED"],
        ),
    ]
    # Docker inherits values from this process; logs contain names, not secrets.
    for key in CONTAINER_ENV_KEYS:
        if key not in HOST_ENV_DEFAULTS and key in os.environ:
            command += ["--env", key]
    return command


def runtime_mounts(home: str) -> list[str]:
    options: list[str] = []
    devices = discover_devices()
    if not devices:
        raise ValueError("没有发现可挂载的昇腾 NPU 字符设备")
    for device in devices:
        options += ["--device", device["path"]]
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
    # bindfs 实测中 import torch 成功，但运行库创建 ~/ascend/log 报 EPERM；
    # 预建后警告消失。仅处理日志目录，不修复 NPU 权限，原始失败原因尚未确认。
    run("docker", "exec", name, "mkdir", "-p", home + "/ascend/log")


def print_instructions(name: str, home: str) -> None:
    print(green(f"容器 {name} 已启动，用户 root，HOME={home}"), flush=True)
    print(shlex.join(["docker", "exec", "-it", name, "bash"]))
    print(yellow("删除容器（保留映射以便复用）："))
    print(
        shlex.join(
            [
                "python3",
                str(ROOT.parent / "cli.py"),
                "containers",
                "delete",
                "--name",
                name,
                "--keep-bindfs",
            ]
        )
    )
    print(yellow("删除容器及未被其他容器引用的映射："))
    print(
        shlex.join(
            [
                "python3",
                str(ROOT.parent / "cli.py"),
                "containers",
                "delete",
                "--name",
                name,
                "--remove-bindfs",
            ]
        )
    )


def create_container(
    args: argparse.Namespace, home: str, sources: list[Path], bindfs: Optional[str]
) -> None:
    store = MappingStore()
    # Serialize registry operations through container creation so deletion cannot
    # unmount an acquired mapping before Docker records its reference.
    with store.locked() if bindfs is not None else nullcontext():
        _create_container(args, home, sources, bindfs, store)


def _create_container(
    args: argparse.Namespace,
    home: str,
    sources: list[Path],
    bindfs: Optional[str],
    store: MappingStore,
) -> None:
    created = False
    image = Image.inspect(args.image)
    mounts = runtime_mounts(home)
    try:
        print(
            green(f"创建容器 {args.name}，镜像 {args.image} ({image.id})"), flush=True
        )
        if bindfs is not None:
            print(
                f"容器 root → 宿主机 UID/GID {os.getuid()}:{os.getgid()}；HOME={home}",
                flush=True,
            )
        else:
            print(
                yellow(
                    "直接挂载（无 bindfs）：容器新文件可能归宿主机 root 所有，普通用户可能无法修改。"
                ),
                flush=True,
            )
        command = docker_command(args.name, home)
        for index, source in enumerate(sources):
            destination = home if index == 0 else str(source)
            if bindfs is not None:
                target = store.acquire(source, bindfs)
            else:
                target = source
                print(yellow(f"直接挂载：{source} → 容器 {destination}"), flush=True)
            command += ["--volume", "{}:{}".format(target, destination)]
        command += mounts
        run(*command, "--entrypoint", "/bin/bash", "-it", image.id)
        created = True
        run("docker", "start", args.name)
        initialize_home(args.name, home)
        initialize_image(args.name, image)
        verify_identity(args.name, home)
        print_instructions(args.name, home)
    except BaseException:
        print(yellow("创建失败或中断，正在清理本次创建的资源。"), flush=True)
        if created:
            run("docker", "rm", "-f", "-v", args.name)
        if bindfs is not None:
            print(yellow("bindfs 映射已保留，可在下次创建时复用。"))
        raise


def execute(args: argparse.Namespace) -> None:
    create_container(*prepare(args))
