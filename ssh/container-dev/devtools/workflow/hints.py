"""面向用户的后续命令；JSON 输出不附加提示。"""

from pathlib import Path
import shlex
import sys
from ssh_common.remote import is_wsl
from ssh_common.terminal import color


def groups(args) -> list:
    host = getattr(args, "host", None)
    hosts = (
        host
        if isinstance(host, list)
        else ([host] if host and host != "localhost" else [])
    )
    remote = ["--host", hosts[0]] if hosts else []
    wsl = is_wsl()
    target = remote if remote or not wsl else ["--host", "<HOST>"]
    sync_hosts = hosts or ["<HOST>"]
    workflow = [
        ("拉取仓库", ["git-repo", "update", *target]),
        ("下载至本地 WSL" if wsl else "本地下载", ["binaries", "download"]),
        (
            "同步资源",
            [
                "binaries",
                "sync",
                *[word for h in sync_hosts for word in ("--host", h)],
                "--apply",
            ],
        ),
        ("解压安装", ["binaries", "extract", *target]),
    ]
    if wsl:
        workflow.append(("一键执行以上四步", ["sync-all", *target]))
    container = [
        ("创建容器", ["containers", "create", *target]),
        ("查看容器和映射", ["containers", "ls", *target]),
        ("删除容器或清理映射", ["containers", "delete", *target]),
        ("下载容器镜像", ["containers", "download", *target]),
    ]
    return [("Workflow", workflow), ("Container", container)]


def suggestions(args) -> list:
    return [row for _, rows in groups(args) for row in rows]


def show(args) -> None:
    if getattr(args, "no_hints", False) or getattr(args, "json", False):
        return
    stream = sys.stderr
    cli = Path(__file__).resolve().parents[1] / "cli.py"
    print("\n" + color("接下来可以…", "1;35", stream), file=stream)
    host = getattr(args, "host", None)
    if isinstance(host, list) and len(host) > 1:
        print(f"  单主机操作以 {host[0]} 为例；同步保留全部主机。", file=stream)
    for title, rows in groups(args):
        print("  " + color(title, "1;35", stream), file=stream)
        for label, command in rows:
            line = shlex.join(["python3", str(cli), *command])
            if command[0] == "sync-all":
                print("    " + color(f"★ {label}  {line}", "1;93", stream), file=stream)
            else:
                print(
                    "    {}  {}".format(color(label, "36", stream), line), file=stream
                )
