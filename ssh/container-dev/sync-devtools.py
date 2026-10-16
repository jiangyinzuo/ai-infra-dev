#!/usr/bin/env python3
"""按机器 OS/Arch 同步 devtools，默认 dry-run，不传输 Docker 镜像归档。"""

import argparse
import csv
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SSH_DIR = ROOT.parent
DEVTOOLS = ROOT / "devtools"
sys.path.insert(0, str(DEVTOOLS))
from config import REMOTE_USER, REMOTE_DIR, TOOL_PLATFORMS


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"{__doc__} 固定同步到 {REMOTE_USER}@HOST:{REMOTE_DIR}/。"
    )
    parser.add_argument("hosts", nargs="+", metavar="HOST")
    parser.add_argument(
        "--apply", dest="apply", action="store_true", help="执行同步并创建远端目录"
    )
    parser.add_argument(
        "--dry-run", dest="apply", action="store_false", help="仅预览（默认）"
    )
    parser.set_defaults(apply=False)
    return parser.parse_args()


def build_plan(hosts):
    path = SSH_DIR / "machine-info/machines-output.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"Host", "OS", "Arch"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected Host,OS,Arch columns")
        machines = {row["Host"]: row for row in reader}
    plans = []
    for host in hosts:
        if host not in machines:
            raise ValueError(f"{host}: missing machine record in {path}")
        row = machines[host]
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


def main():
    args = parse_args()
    try:
        plans = build_plan(args.hosts)
        if not shutil.which("rsync"):
            raise ValueError("rsync is required")
        return synchronize(plans, args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
