"""WSL 上依次拉取、下载、同步、解压；复用各命令的执行函数。"""

from argparse import Namespace
from ssh_common.arguments import add_jobs_argument
from ssh_common.remote import add_host_argument, is_wsl, is_remote_host
from ssh_common.terminal import color
from . import update_git_repo, sync


def register(actions):
    parser = actions.add_parser("sync-all", help="WSL 一键拉取、下载、同步、解压")
    add_host_argument(parser, required=True)
    add_jobs_argument(parser)
    parser.set_defaults(handler=execute, command_parser=parser)


def execute(args):
    if not is_wsl() or not is_remote_host(args.host):
        raise ValueError("sync-all 只能在 WSL 执行，且 --host 必须指定远程服务器")
    from . import binaries

    steps = [
        (
            "拉取远程仓库",
            update_git_repo.execute,
            Namespace(host=args.host, debug=False, command_parser=args.command_parser),
        ),
        (
            "下载至本地 WSL",
            binaries.execute,
            Namespace(
                host=None,
                binaries_action="download",
                jobs=args.jobs,
                lock=None,
                repos=None,
                output=None,
                command_parser=args.command_parser,
            ),
        ),
        ("同步资源", sync.execute, Namespace(host=[args.host], apply=True)),
        (
            "远程解压",
            binaries.execute,
            Namespace(
                host=args.host,
                binaries_action="extract",
                jobs=args.jobs,
                output=None,
                command_parser=args.command_parser,
            ),
        ),
    ]
    for index, (label, handler, options) in enumerate(steps, 1):
        print(color(f"[{index}/4] {label}", "1;35"), flush=True)
        result = handler(options) or 0
        if result:
            return result
    return 0
