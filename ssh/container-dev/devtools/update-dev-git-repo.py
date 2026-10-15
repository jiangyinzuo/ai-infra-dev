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
    for command in ("git", "ssh", "nc"):
        if shutil.which(command) is None:
            requirement = "nc (Ncat or OpenBSD netcat)" if command == "nc" else command
            raise RuntimeError(f"{requirement} is required")
    proxy_ip = os.environ.get("USG_IP")
    if not proxy_ip:
        raise RuntimeError("未设置USG_IP环境变量，请检查 SSH wrapper")
    proxy_address = f"{proxy_ip}:3128"
    nc_help = subprocess.run(
        ["nc", "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ).stdout
    nc_command = ["nc"]
    if debug:
        nc_command.append("-v")
    if "Ncat" in nc_help:
        nc_command.extend(["--proxy-type", "http", "--proxy", proxy_address])
    else:
        nc_command.extend(["-X", "connect", "-x", proxy_address])
    nc_command.extend(["%h", "%p"])
    ssh_command = ["ssh"]
    if debug:
        ssh_command.append("-vvv")
    ssh_command.extend(
        [
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "Hostname=ssh.github.com",
            "-p",
            "443",
            "-o",
            "HostKeyAlias=github.com",
            "-o",
            "ProxyCommand=" + shlex.join(nc_command),
        ]
    )
    git_ssh_command = shlex.join(ssh_command)
    env = dict(os.environ, GIT_SSH_COMMAND=git_ssh_command)
    if debug:
        env["GIT_TRACE"] = "1"
    print(
        f"HTTP CONNECT proxy={proxy_address}; SSH endpoint=ssh.github.com:443",
        file=sys.stderr,
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
        print(
            f"GIT_SSH_COMMAND={shlex.quote(git_ssh_command)} {shlex.join(command)}",
            flush=True,
        )
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
        "--debug", action="store_true", help="enable Git, SSH and nc tracing"
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
