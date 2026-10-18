#!/usr/bin/env python3
"""Rootful Docker with optional bindfs ownership mapping (Python 3.9+)."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import re
import shlex
import shutil
from typing import Optional

from ..core.remote import ROOT
from config import HOST_ENV_DEFAULTS, CONTAINER_ENV_KEYS
from ..core.mappings import MappingStore
from ..core.process import run


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


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
    prefix = "jiangyinzuo-bindfs-dev" if mapped else "jiangyinzuo-dev"
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
    if args.name in existing_names:
        raise ValueError("Container name already exists")
    if bindfs is not None:
        run("sudo", "-n", "true")
    return args, home, sources, bindfs


def docker_command(name: str, home: str) -> list[str]:
    command = [
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
    try:
        print(green(f"创建容器 {args.name}，镜像 {args.image}"), flush=True)
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
        command += runtime_mounts(home)
        run(*command, "--entrypoint", "/bin/bash", "-it", args.image)
        created = True
        run("docker", "start", args.name)
        initialize_home(args.name, home)
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
