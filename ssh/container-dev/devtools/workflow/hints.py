"""Human-facing next steps; never mix hints into JSON output."""

from pathlib import Path
import shlex
import sys
from ssh_common.remote import is_wsl
from ssh_common.terminal import color


def suggestions(args) -> list:
    host = getattr(args, "host", None)
    hosts = (
        host
        if isinstance(host, list)
        else ([host] if host and host != "localhost" else [])
    )
    remote = ["--host", hosts[0]] if hosts else []
    rows = []

    def add(label, command):
        rows.append((label, command))

    local = not is_wsl()
    add("更新开发仓库", ["git-repo", "update", *remote])
    add("下载 Neovim 插件", ["nvim-plugins", "download", *remote])
    add("下载命令行工具", ["binaries", "download", *remote])
    if hosts:
        add(
            "同步工具到服务器",
            [
                "binaries",
                "sync",
                *[word for h in hosts for word in ("--host", h)],
                "--apply",
            ],
        )
    elif not local:
        add("同步工具到服务器", ["binaries", "sync", "--host", "<HOST>", "--apply"])
    target = remote if remote or local else ["--host", "<HOST>"]
    # Remote-only commands need an explicit target on WSL.
    if not local and not remote:
        rows[0] = ("更新开发仓库", ["git-repo", "update", *target])
    add("解压 Neovim 插件", ["nvim-plugins", "extract", *target])
    add("解压工具并安装 bindfs", ["binaries", "extract", *target])
    add("创建开发容器", ["containers", "create", *target])
    add("查看容器和映射", ["containers", "ls", *target])
    add("删除容器或清理映射", ["containers", "delete", *target])
    add("下载容器镜像", ["containers", "download", *target])
    current = [getattr(args, "command", "")]
    action = next(
        (
            getattr(args, key)
            for key in (
                "plugin_action",
                "binaries_action",
                "container_action",
                "repo_action",
            )
            if getattr(args, key, None)
        ),
        None,
    )
    if action:
        current.append(action)
    return [
        (label, command)
        for label, command in rows
        if command[: len(current)] != current
    ]


def show(args) -> None:
    if getattr(args, "no_hints", False) or getattr(args, "json", False):
        return
    stream = sys.stderr

    cli = Path(__file__).resolve().parents[1] / "cli.py"
    print("\n" + color("接下来可以…", "1;35", stream), file=stream)
    host = getattr(args, "host", None)
    if isinstance(host, list) and len(host) > 1:
        print(
            "  单主机操作以下以 {} 为例；同步保留全部主机。".format(host[0]),
            file=stream,
        )
    for label, command in suggestions(args):
        line = shlex.join(["python3", str(cli), *command])
        print("  {}  {}".format(color(label, "36", stream), line), file=stream)
