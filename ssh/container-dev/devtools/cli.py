#!/usr/bin/env python3
"""开发工具：仓库、Neovim 插件、工具同步与容器管理（Python 3.9+）。"""

import argparse
import subprocess
import sys
import tarfile
from containers import interface as containers
from workflow import hints


def build_parser() -> argparse.ArgumentParser:
    from workflow import nvim_plugins, binaries, update_git_repo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hints", action="store_true", help="不显示后续操作提示")
    commands = parser.add_subparsers(dest="command", required=True)
    nvim_plugins.register(commands)
    binaries.register(commands)
    update_git_repo.register(commands)
    containers.register(commands)
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
        print("error: " + str(exc), file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            return exc.returncode if exc.returncode >= 0 else 128 - exc.returncode
        return 1
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
