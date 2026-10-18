#!/usr/bin/env python3
"""Update development repositories through the SSH HTTP CONNECT proxy."""

import os
from pathlib import Path
import shutil

from ssh_common.config import REMOTE_USER
from ssh_common.remote import (
    add_host_argument,
    is_remote_host,
    is_wsl,
    update_remote_repositories,
)
from ssh_common.terminal import run


def update_repositories(debug):
    missing = [name for name in ("git", "ncat") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "缺少必需命令：" + ", ".join(missing) + "；请安装后重试（不会自动安装）"
        )
    env = dict(os.environ)
    if debug:
        env.update(
            GIT_TRACE="1",
            GIT_SSH_COMMAND=(env.get("GIT_SSH_COMMAND") or "ssh") + " -vvv",
        )
    for repo in ("ai-infra-dev", "vimrc"):
        destination = Path.home() / repo
        cloning = not (destination.exists() or destination.is_symlink())
        if cloning:
            command = [
                "git",
                "clone",
                f"git@github.com:{REMOTE_USER}/{repo}.git",
                str(destination),
            ]
        else:
            git_dir = destination / ".git"
            if not (git_dir.is_dir() or git_dir.is_file()):
                raise RuntimeError(f"{destination} exists but is not a Git repository")
            command = ["git", "-C", str(destination), "pull", "--ff-only"]
        run(*command, env=env)
        if repo == "vimrc" and cloning:
            run("./install/setup.sh", cwd=destination)
    return 0


def register(subparsers):
    group = subparsers.add_parser("git-repo", help="开发 Git 仓库")
    actions = group.add_subparsers(dest="repo_action", required=True)
    parser = actions.add_parser(
        "update",
        description=(
            f"WSL 必须指定 --host，仅传 SSH Host 别名；远端用户固定为 {REMOTE_USER}。"
            "服务器本地执行不传 --host。"
        ),
    )
    add_host_argument(parser)
    parser.add_argument(
        "--debug", action="store_true", help="enable Git and SSH tracing"
    )
    parser.set_defaults(handler=execute, command_parser=parser)


def execute(args):
    parser = args.command_parser
    if is_wsl() and not is_remote_host(args.host):
        parser.error(f"WSL requires --host HOST (user: {REMOTE_USER})")
    if is_remote_host(args.host):
        return update_remote_repositories(args.host, args.debug)
    return update_repositories(args.debug)
