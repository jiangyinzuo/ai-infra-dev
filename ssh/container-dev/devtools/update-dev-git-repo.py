#!/usr/bin/env python3
"""Update development repositories through the SSH HTTP CONNECT proxy."""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from config import REMOTE_USER
from remote_exec import is_wsl, run_remote


def host_alias(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            f"--host requires a Host alias only; user is fixed to {REMOTE_USER}"
        )
    return value


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            f"WSL 必须指定 --host，仅传 SSH Host 别名；远端用户固定为 {REMOTE_USER}。"
            "服务器本地执行不传 --host。"
        )
    )
    parser.add_argument("--host", type=host_alias, help="SSH Host alias")
    parser.add_argument(
        "--debug", action="store_true", help="enable Git and SSH tracing"
    )
    args = parser.parse_args()
    if is_wsl():
        if not args.host:
            parser.error(f"WSL requires --host HOST (user: {REMOTE_USER})")
        return run_remote(
            args.host, "update-dev-git-repo.py", ("--debug",) if args.debug else ()
        )
    if args.host:
        parser.error("--host is only supported on WSL; omit it on the server")
    return update_repositories(args.debug)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode if exc.returncode >= 0 else 128 - exc.returncode)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
