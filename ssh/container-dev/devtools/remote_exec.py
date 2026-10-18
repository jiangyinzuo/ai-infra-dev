#!/usr/bin/env python3
"""WSL 通过 ssh2 执行已同步的 devtools 脚本，不负责传文件。"""

import argparse
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import subprocess

from config import REMOTE_USER


def host_alias(value: str) -> str:
    """校验 SSH Host 别名；登录用户由共享配置决定，不接受 user@host。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("--host 仅接受 SSH Host 别名，用户固定为 " + REMOTE_USER)
    return value


def add_host_argument(
    parser: argparse.ArgumentParser, *, required: bool = False, multiple: bool = False
) -> None:
    """统一主机参数；multiple=True 时重复 --host 收集多个目标。"""
    help_text = f"SSH Host 别名，用户固定为 {REMOTE_USER}"
    help_text += (
        "；必填" if required else "；省略或 localhost 表示本地，受命令运行环境限制"
    )
    if multiple:
        help_text += "；可重复指定"
    parser.add_argument(
        "--host",
        type=host_alias,
        metavar="HOST",
        required=required,
        action="append" if multiple else "store",
        help=help_text,
    )


def is_wsl():
    return "microsoft" in platform.release().lower()


def run_remote(host, script, args=()):
    host_alias(host)
    path = PurePosixPath(script)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("远程脚本必须位于 ~/devtools/")
    interpreter = "bash" if path.suffix == ".sh" else "python3"
    command = interpreter + ' "$HOME"/devtools/' + shlex.quote(script)
    if args:
        command += " " + shlex.join(list(args))
    ssh = Path(__file__).resolve().parents[2] / "ssh2"
    print(f"run on {REMOTE_USER}@{host} using ~/devtools/{script}", flush=True)
    return subprocess.run(
        [str(ssh), f"{REMOTE_USER}@{host}", command], check=False
    ).returncode
