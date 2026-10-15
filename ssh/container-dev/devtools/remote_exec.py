#!/usr/bin/env python3
"""WSL 通过 ssh2 执行已同步的 devtools 脚本，不负责传文件。"""

import argparse
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import subprocess
import sys

from config import REMOTE_USER


def is_wsl():
    return "microsoft" in platform.release().lower()


def run_remote(host, script, args=()):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host):
        raise ValueError("--host 仅接受 SSH Host 别名，用户固定为 " + REMOTE_USER)
    path = PurePosixPath(script)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("远程脚本必须位于 ~/devtools/")
    interpreter = "bash" if path.suffix == ".sh" else "python3"
    command = interpreter + ' "$HOME"/devtools/' + shlex.quote(script)
    if args:
        command += " " + shlex.join(list(args))
    ssh = Path(__file__).resolve().parents[2] / "ssh2"
    print(f"run on {REMOTE_USER}@{host} using ~/devtools/{script}", flush=True)
    return subprocess.run(
        [str(ssh), f"{REMOTE_USER}@{host}", command], check=False
    ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help=f"SSH Host 别名；用户固定 {REMOTE_USER}")
    parser.add_argument("script", help="相对于 ~/devtools 的脚本")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        host = args.host or input("WSL: SSH Host（用户 jiangyinzuo）: ").strip()
        return run_remote(host, args.script, args.args)
    except (OSError, ValueError, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
