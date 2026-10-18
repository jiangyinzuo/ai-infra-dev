#!/usr/bin/env python3
"""按机器 OS/Arch 同步 devtools，默认 dry-run，不传输 Docker 镜像归档。"""

from pathlib import Path
import shlex
import shutil
import subprocess

DEVTOOLS = Path(__file__).resolve().parents[1]
SSH_DIR = DEVTOOLS.parent.parent
from config import REMOTE_USER, REMOTE_DIR, TOOL_PLATFORMS, NVIM_PLUGIN_EXCLUDES
from remote_exec import add_host_argument
from inventory import read_machines, machine_record


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
            source = DEVTOOLS / TOOL_PLATFORMS[key]
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
    (DEVTOOLS / "nvim-plugins").mkdir(exist_ok=True)
    common = [
        DEVTOOLS / "nvim-plugins",
        *sorted(DEVTOOLS.glob("*.py")),
        *sorted(DEVTOOLS.glob("*.sh")),
        DEVTOOLS / "proxy.env",
        DEVTOOLS / "containers",
        DEVTOOLS / "workflow",
    ]
    for host, source, ssh in plans:
        command = ["rsync", "-av", "--info=progress2", "-e", shlex.quote(ssh)]
        if args.apply:
            command += [
                "--rsync-path",
                f"mkdir -p -- {remote_path(REMOTE_DIR)} && exec rsync",
            ]
        else:
            command.append("--dry-run")
        # 镜像目录只同步目录和脚本，不传输归档。
        command += [
            "--exclude=*.md",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            "--include=/containers/**/",
            "--include=/containers/**.sh",
            "--include=/containers/**.py",
            "--exclude=/containers/**",
        ]
        command += [
            f"--exclude=/nvim-plugins/{name}.tar.gz"
            for name in sorted(NVIM_PLUGIN_EXCLUDES)
        ]
        command += [
            str(source),
            *(str(path) for path in common),
            f"{REMOTE_USER}@{host}:{REMOTE_DIR}/",
        ]
        mode = "sync" if args.apply else "dry-run"
        print(
            f"\033[32m{mode} {source} + common -> {REMOTE_USER}@{host}:{REMOTE_DIR}\033[0m",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


def execute(args):
    plans = build_plan(args.host)
    if not shutil.which("rsync"):
        raise ValueError("rsync is required")
    return synchronize(plans, args)
