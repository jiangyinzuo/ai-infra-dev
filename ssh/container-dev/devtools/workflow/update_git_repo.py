#!/usr/bin/env python3
"""Update development repositories through the SSH HTTP CONNECT proxy."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from config import REMOTE_USER
from remote_exec import add_host_argument, is_wsl, run_remote


def update_repositories(debug):
    if shutil.which("git") is None:
        raise RuntimeError("git is required")
    env = dict(os.environ)
    if debug:
        env.update(GIT_TRACE="1", GIT_SSH_COMMAND="ssh -vvv")
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
        print(shlex.join(command), flush=True)
        subprocess.run(command, env=env, check=True)
        if repo == "vimrc" and cloning:
            print(
                f"(cd {shlex.quote(str(destination))} && ./install/setup.sh)",
                flush=True,
            )
            subprocess.run(["./install/setup.sh"], cwd=destination, check=True)
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
    if is_wsl() and (not args.host or args.host == "localhost"):
        parser.error(f"WSL requires --host HOST (user: {REMOTE_USER})")
    if args.host and args.host != "localhost":
        remote_args = ["--no-hints", "git-repo", "update"]
        if args.debug:
            remote_args.append("--debug")
        return run_remote(args.host, "cli.py", remote_args)
    return update_repositories(args.debug)
