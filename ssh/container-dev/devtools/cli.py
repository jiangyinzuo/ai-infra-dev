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


class HelpOnEmptyParser(argparse.ArgumentParser):
    """Show help only when an empty command has required arguments."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        return super().parse_args(args, namespace)

    def _requires_arguments(self):
        return any(action.required for action in self._actions) or any(
            group.required for group in self._mutually_exclusive_groups
        )

    def parse_known_args(self, args=None, namespace=None):
        if args is not None and not args and self._requires_arguments():
            self.print_help()
            self.exit()
        return super().parse_known_args(args, namespace)


def build_parser() -> argparse.ArgumentParser:
    from workflow import binaries, update_git_repo, sync_all, checkhealth, images

    parser = HelpOnEmptyParser(description=__doc__)
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
                + "模型扫描可用 --root 缩小范围，或用 --timeout 增加等待时间。",
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
