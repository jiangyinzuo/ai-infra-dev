#!/usr/bin/env python3
"""开发工具：仓库、Neovim 插件、工具同步与容器管理（Python 3.9+）。"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from containers import interface as containers
from workflow import hints
import models


def build_parser() -> argparse.ArgumentParser:
    from workflow import binaries, update_git_repo, sync_all, checkhealth, images

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hints", action="store_true", help="不显示后续操作提示")
    commands = parser.add_subparsers(dest="command", required=True)
    binaries.register(commands)
    update_git_repo.register(commands)
    containers.register(commands)
    sync_all.register(commands)
    models.register(commands)
    checkhealth.register(commands)
    images.register(commands)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = args.handler(args) or 0
        if result == 0:
            hints.show(args)
        return result
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        EOFError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            host = getattr(args, "host", None)
            print(
                f"error: 远程操作超时（主机 {host}，{exc.timeout} 秒）。"
                "模型扫描可用 --root 缩小范围，或用 --timeout 增加等待时间。",
                file=sys.stderr,
            )
            return 1
        if isinstance(exc, subprocess.CalledProcessError):
            if exc.stderr:
                print(exc.stderr.rstrip(), file=sys.stderr)
            else:
                command = exc.cmd
                program = (
                    Path(command[0]).name
                    if isinstance(command, (list, tuple))
                    else "子进程"
                )
                print(
                    f"error: {program} 执行失败（退出码 {exc.returncode}）",
                    file=sys.stderr,
                )
            return exc.returncode if exc.returncode >= 0 else 128 - exc.returncode
        print("error: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
