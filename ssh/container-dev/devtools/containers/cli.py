#!/usr/bin/env python3
"""容器管理：create / delete / download / ls（Python 3.9+）。"""

import argparse
import subprocess
import sys
from commands import create, delete, download, ls
from core.remote import add_target_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    creator = commands.add_parser("create", help="创建 root 容器，默认 bindfs 映射")
    creator.add_argument("--image", help="镜像名或 ID；默认交互选择")
    creator.add_argument("--name", help="容器名；默认递增并交互确认")
    creator.add_argument(
        "--no-bindfs",
        action="store_true",
        help="直接挂载，避免 FUSE 开销；新文件可能归宿主机 root 所有",
    )
    creator.add_argument(
        "--share", action="append", default=[], help="额外共享目录，可重复指定"
    )
    creator.add_argument("--test-home", help="以隔离目录代替真实 HOME")
    creator.set_defaults(handler=create.execute)
    remover = commands.add_parser("delete", help="删除容器或清理未引用映射")
    remover.add_argument("--name", help="容器名称；默认交互选择")
    mode = remover.add_mutually_exclusive_group()
    mode.add_argument(
        "--keep-bindfs", action="store_true", help="删除容器及匿名 volume，保留映射"
    )
    mode.add_argument(
        "--remove-bindfs", action="store_true", help="同时清理未引用的托管映射"
    )
    remover.add_argument(
        "--cleanup-bindfs", action="store_true", help="仅清理未引用的残留映射"
    )
    remover.add_argument(
        "--yes", action="store_true", help="确认残留清理，适用于自动化"
    )
    remover.set_defaults(handler=delete.execute)
    fetcher = commands.add_parser("download", help="下载 NPU 容器镜像")
    add_target_args(fetcher)
    fetcher.add_argument("--date", default="20261014", help="日期 YYYYMMDD")
    fetcher.add_argument("--version", default="26.2.0", help="镜像版本")
    fetcher.set_defaults(handler=download.execute)
    listing = commands.add_parser(
        "ls", help="列出 jiangyinzuo 容器及 bindfs 映射，不作修改"
    )
    listing.add_argument(
        "--json", action="store_true", help="JSON 输出；命令日志写 stderr"
    )
    listing.set_defaults(handler=ls.execute)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args) or 0
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
