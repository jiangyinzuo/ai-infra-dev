"""Quick read-only development environment summary; no network or deep scans."""

import json
from pathlib import Path
import shutil
import subprocess

from .paths import DATA_ROOT
from ssh_common.remote import add_host_argument, is_remote_host, is_wsl, run_remote_cli
from ssh_common.config import NVIM_PLUGIN_EXCLUDES


def register(subparsers):
    parser = subparsers.add_parser("checkhealth", help="快速检查服务器开发环境")
    add_host_argument(parser)
    parser.set_defaults(handler=execute)


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def repository(path):
    head = command(["git", "-C", str(path), "rev-parse", "--short", "HEAD"])
    if head is None:
        return "不可用", False
    status = command(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]
    )
    if status is None:
        return head + " 状态查询失败/超时", False
    return head + (" 有已跟踪改动" if status else " 干净（不检查未跟踪文件）"), True


def plugins(home):
    lock = home / ".config/nvim/lazy-lock.json"
    try:
        names = (set(json.loads(lock.read_text())) - NVIM_PLUGIN_EXCLUDES) | {
            "lazy.nvim"
        }
    except (OSError, ValueError, TypeError):
        return "锁文件缺失或无效", False
    installed = 0
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            return "锁文件包含无效插件名", False
        path = home / (
            ".local/share/nvim/lazy/lazy.nvim"
            if name == "lazy.nvim"
            else "plugged/" + name
        )
        try:
            if path.is_dir() and next(path.iterdir(), None) is not None:
                installed += 1
        except OSError:
            pass
    return f"{installed}/{len(names)} 目录存在且非空（未校验内容）", installed == len(
        names
    )


def local_report(home):
    rows = []
    for name in ("ai-infra-dev", "vimrc"):
        text, ok = repository(home / name)
        rows.append((name, text, ok))
    bindfs = shutil.which("bindfs") or str(home / ".local/bin/bindfs")
    version = command([bindfs, "--version"])
    rows.append(
        (
            "bindfs",
            version.splitlines()[0] if version else "缺失或不可运行",
            bool(version),
        )
    )
    text, ok = plugins(home)
    rows.append(("nvim-plugins", text, ok))
    nvim = shutil.which("nvim") or str(
        DATA_ROOT / "openeuler24.03-lts-aarch64/nvim-linux-arm64/bin/nvim"
    )
    version = command([nvim, "--version"])
    rows.append(
        (
            "nvim",
            version.splitlines()[0] if version else "缺失或不可运行",
            bool(version),
        )
    )
    version = command([str(home / "plugged/fzf/bin/fzf"), "--version"])
    rows.append(
        ("fzf", version.splitlines()[0] if version else "缺失或不可运行", bool(version))
    )
    return rows


def execute(args):
    if is_remote_host(args.host):
        return run_remote_cli(args.host, ["checkhealth"])
    if is_wsl():
        raise ValueError("WSL 请通过 --host 指定服务器")
    rows = local_report(Path.home())
    for name, text, ok in rows:
        print(f"{'OK' if ok else 'WARN'} {name}: {text}")
    return 0 if all(ok for _, _, ok in rows) else 1
