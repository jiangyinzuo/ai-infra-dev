#!/usr/bin/env python3
"""Container command registration for devtools/cli.py."""

from ssh_common.config import REMOTE_USER

import argparse
import sys
from contextlib import redirect_stdout
from .commands import create, delete, ls
from .core.remote import add_target_args
from ssh_common.remote import add_host_argument, is_remote_host, is_wsl, run_remote_cli


def register(subparsers) -> None:
    parser = subparsers.add_parser("containers", help="容器管理")
    commands = parser.add_subparsers(dest="container_action", required=True)
    creator = commands.add_parser("create", help="创建 root 容器，默认 bindfs 映射")
    creator.add_argument("--image", help="镜像名或 ID；默认交互选择")
    creator.add_argument("--name", help="容器名；默认递增并交互确认")
    creator.add_argument(
        "--no-bindfs",
        action="store_true",
        help="直接挂载，避免 FUSE 开销；新文件可能归宿主机 root 所有",
    )
    creator.add_argument(
        "--share", action="append", default=[], help="额外共享目录，可重复指定"
    )
    creator.add_argument("--test-home", help="以隔离目录代替真实 HOME")
    add_target_args(creator)
    creator.add_argument("--connection-host", help=argparse.SUPPRESS)
    creator.set_defaults(handler=execute, container_handler=create.execute)
    remover = commands.add_parser("delete", help="删除容器或清理未引用映射")
    remover.add_argument("--name", help="容器名称；默认交互选择")
    mode = remover.add_mutually_exclusive_group()
    mode.add_argument(
        "--keep-bindfs", action="store_true", help="删除容器及匿名 volume，保留映射"
    )
    mode.add_argument(
        "--remove-bindfs", action="store_true", help="同时清理未引用的托管映射"
    )
    remover.add_argument(
        "--cleanup-bindfs", action="store_true", help="仅清理未引用的残留映射"
    )
    remover.add_argument(
        "--yes", action="store_true", help="确认残留清理，适用于自动化"
    )
    add_host_argument(remover)
    remover.set_defaults(handler=execute, container_handler=delete.execute)
    listing = commands.add_parser(
        "ls", help=f"列出 {REMOTE_USER} 容器及 bindfs 映射，不作修改"
    )
    listing.add_argument(
        "--json", action="store_true", help="JSON 输出；命令日志写 stderr"
    )
    add_host_argument(listing)
    listing.set_defaults(handler=execute, container_handler=ls.execute)


def execute(args: argparse.Namespace):
    if args.container_action == "create" and is_wsl() and not is_remote_host(args.host):
        raise ValueError("请通过 --host 指定服务器执行容器管理")
    if is_remote_host(args.host):
        options = ["containers", args.container_action]
        for key in ("image", "name", "test_home"):
            value = getattr(args, key, None)
            if value is not None:
                options += ["--" + key.replace("_", "-"), value]
        for value in getattr(args, "share", []):
            options += ["--share", value]
        for key in (
            "no_bindfs",
            "keep_bindfs",
            "remove_bindfs",
            "cleanup_bindfs",
            "yes",
            "json",
        ):
            if getattr(args, key, False):
                options.append("--" + key.replace("_", "-"))
        if args.container_action == "create":
            from .core.remote import dispatch

            if is_wsl():
                options += ["--connection-host", args.host]

            return dispatch(args, "create", options[2:])
        with (
            redirect_stdout(sys.stderr)
            if getattr(args, "json", False)
            else redirect_stdout(sys.stdout)
        ):
            return run_remote_cli(args.host, options)
    if is_wsl():
        raise ValueError("请通过 --host 指定服务器执行容器管理")
    if args.container_action == "create":
        from .core.remote import dispatch

        dispatch(args, "create", [])
    return args.container_handler(args)
