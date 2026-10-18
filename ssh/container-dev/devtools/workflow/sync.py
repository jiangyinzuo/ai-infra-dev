#!/usr/bin/env python3
"""按机器 OS/Arch 同步 devtools，默认 dry-run，不传输 Docker 镜像归档。"""

import shlex
import shutil

from .paths import DATA_ROOT, DEVTOOLS

SSH_DIR = DEVTOOLS.parent.parent
from ssh_common.config import (
    REMOTE_USER,
    REMOTE_DIR,
    TOOL_PLATFORMS,
    NVIM_PLUGIN_EXCLUDES,
)
from ssh_common.remote import add_host_argument, checked_command
from ssh_common.inventory import read_machines, machine_record
from ssh_common.terminal import green, run


def register(subparsers):
    parser = subparsers.add_parser("sync", help=__doc__)
    add_host_argument(parser, required=True, multiple=True)
    parser.add_argument(
        "--apply", dest="apply", action="store_true", help="执行同步并创建远端目录"
    )
    parser.add_argument(
        "--dry-run", dest="apply", action="store_false", help="仅预览（默认）"
    )
    parser.set_defaults(apply=False, handler=execute)


def build_plan(hosts):
    path = SSH_DIR / "machine-info/machines-output.csv"
    machines = read_machines(path, ("OS", "Arch"))
    plans = []
    for host in hosts:
        row = machine_record(machines, host, path)
        os_name = row["OS"].strip().lower().split(maxsplit=1)
        key = (os_name[0] if os_name else "", row["Arch"].strip().lower())
        try:
            source = DATA_ROOT / TOOL_PLATFORMS[key]
        except KeyError:
            raise ValueError(
                f"{host}: unsupported OS/Arch: {row['OS']} / {row['Arch']}"
            ) from None
        if not source.is_dir():
            raise ValueError(f"source directory not found: {source}")
        ssh = str(SSH_DIR / "ssh2")
        plans.append((host, source, ssh))
    return plans


def remote_path(path):
    """只在远端展开开头的 ~，其他字符按字面引用。"""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def synchronize(plans, args):
    plugins = DATA_ROOT / "nvim-plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    destination = REMOTE_DIR
    for host, source, ssh in plans:
        remote_command = "exec rsync"
        if args.apply:
            remote_command = f"mkdir -p -- {remote_path(destination)} && exec rsync"
        command = [
            "rsync",
            "-av",
            "--info=progress2",
            "-e",
            shlex.quote(ssh),
            "--rsync-path",
            checked_command(host, remote_command),
        ]
        if not args.apply:
            command.append("--dry-run")
        command += [
            f"--exclude=/nvim-plugins/{name}.tar.gz"
            for name in sorted(NVIM_PLUGIN_EXCLUDES)
        ]
        # Only resource archives, plugin metadata, and fzf runtime helpers cross SSH.
        command += [
            f"--include=/{source.name}/",
            f"--include=/{source.name}/omp",
            f"--include=/{source.name}/fzf-*-support/",
            f"--include=/{source.name}/fzf-*-support/*.sh",
            f"--include=/{source.name}/fzf-*-support/fzf-tmux",
        ]
        command += [
            f"--include=/{source.name}/*{suffix}"
            for suffix in (".tar", ".gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar.zst")
        ]
        command += [
            "--include=/nvim-plugins/",
            "--include=/nvim-plugins/*.tar.gz",
            "--include=/nvim-plugins/archive-hashes.json",
            "--exclude=*",
            str(source),
            str(plugins),
            f"{REMOTE_USER}@{host}:{destination}/",
        ]
        mode = "sync" if args.apply else "dry-run"
        print(
            green(f"{mode} {source} + plugins -> {REMOTE_USER}@{host}:{destination}"),
            flush=True,
        )
        result = run(*command, check=False)
        if result.returncode:
            return result.returncode
    return 0


def execute(args):
    plans = build_plan(args.host)
    if not shutil.which("rsync"):
        raise ValueError("rsync is required")
    return synchronize(plans, args)
