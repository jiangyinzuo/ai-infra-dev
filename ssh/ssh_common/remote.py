#!/usr/bin/env python3
"""通过 ssh2 执行远程仓库代码；普通操作要求两端 HEAD 一致。"""

import argparse
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import subprocess
import sys

from .config import REMOTE_USER

SSH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SSH_DIR.parent


def local_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def update_command(host: str) -> str:
    return shlex.join(
        [
            "python3",
            str(SSH_DIR / "container-dev/devtools/cli.py"),
            "git-repo",
            "update",
            "--host",
            host,
        ]
    )


def checked_command(host: str, command: str) -> str:
    expected = local_commit()
    if not re.fullmatch(r"[0-9a-f]{40,64}", expected):
        raise ValueError("无法识别本地 Git commit")
    hint = "请先提交并推送本地改动，再执行：" + update_command(host)
    color_warning = "1" if sys.stderr.isatty() and "NO_COLOR" not in os.environ else "0"
    # Check and execute in one SSH session; preserve stdin for interactive commands.
    return (
        'repo="$HOME/ai-infra-dev"; '
        'actual=$(git -C "$repo" rev-parse HEAD 2>/dev/null) || actual=missing; '
        f'if [ "$actual" != {shlex.quote(expected)} ]; then '
        "warning_start=; warning_end=; "
        f'if {{ [ -t 2 ] || [ {color_warning} = 1 ]; }} && [ "${{NO_COLOR+x}}" != x ]; then '
        "warning_start=$(printf '\\033[33m'); warning_end=$(printf '\\033[0m'); fi; "
        'printf "%s" "$warning_start" >&2; '
        f'printf "远程仓库 commit 不一致：local=%s remote=%s\\n" {shlex.quote(expected)} "$actual" >&2; '
        f'printf "%s\\n" {shlex.quote(hint)} >&2; printf "%s" "$warning_end" >&2; exit 2; fi; '
        + command
    )


def ssh_command(host: str, command: str, *, interactive: bool = False) -> int:
    host_alias(host)
    return subprocess.run(
        [
            str(SSH_DIR / "ssh2"),
            *(["-t"] if interactive else []),
            f"{REMOTE_USER}@{host}",
            command,
        ],
        check=False,
    ).returncode


def update_remote_repositories(host: str, debug: bool = False) -> int:
    """唯一引导例外：仓库不存在/版本不同时也能克隆或快进更新。"""
    host_alias(host)
    commands = ["set -e"]
    # Bootstrap cannot import remote repository helpers yet; use POSIX builtins.
    commands.append(
        "missing=0; for program in git ncat; do "
        'if ! command -v "$program" >/dev/null 2>&1; then '
        'printf "缺少必需命令：%s；请安装后重试（不会自动安装）\\n" "$program" >&2; missing=1; fi; '
        'done; [ "$missing" -eq 0 ] || exit 127'
    )
    if debug:
        commands.append(
            'export GIT_TRACE=1; export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh} -vvv"'
        )
    # 面向程序员的工具优先展示实际命令，不用抽象流程标签代替。
    for name in ("ai-infra-dev", "vimrc"):
        path = '"$HOME"/' + name
        url = shlex.quote(f"git@github.com:{REMOTE_USER}/{name}.git")
        commands.append(
            f"if [ -e {path} ] || [ -L {path} ]; then "
            f"if [ ! -d {path}/.git ] && [ ! -f {path}/.git ]; then "
            f'printf "%s\\n" "Not a Git repository: {name}" >&2; exit 2; fi; '
            f"git -C {path} rev-parse --show-toplevel >/dev/null; "
            f'printf "git -C %s pull --ff-only\\n" {path}; '
            f"git -C {path} pull --ff-only; "
            f'else printf "git clone %s %s\\n" {url} {path}; git clone {url} {path}; '
            + (f"(cd {path} && ./install/setup.sh); " if name == "vimrc" else "")
            + "fi"
        )
    return ssh_command(host, "; ".join(commands))


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


def is_remote_host(host):
    """Classify an optional CLI target; omitted hosts and localhost run locally."""
    return bool(host) and host != "localhost"


def run_remote_cli(host, args=()):
    """Run explicit CLI arguments remotely without printing follow-up hints."""
    return run_remote(host, "cli.py", ["--no-hints", *args])


def run_remote(host, script, args=()):
    host_alias(host)
    path = PurePosixPath(script)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("远程脚本必须位于仓库的 container-dev/devtools/ 下")
    interpreter = "bash" if path.suffix == ".sh" else "python3"
    command = (
        interpreter
        + ' "$HOME"/ai-infra-dev/ssh/container-dev/devtools/'
        + shlex.quote(script)
    )
    if args:
        command += " " + shlex.join(list(args))
    print(f"run repository script on {REMOTE_USER}@{host}: {script}", flush=True)
    return ssh_command(host, checked_command(host, command))
