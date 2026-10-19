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
    workflow.append(("检查开发环境", ["checkhealth", *target]))
    container = [
        ("创建容器", ["containers", "create", *target]),
        ("查看容器和映射", ["containers", "ls", *target]),
        ("删除容器或清理映射", ["containers", "delete", *target]),
        ("下载容器镜像", ["containers", "download", *target]),
    ]
    model = [("发现模型", ["models", "discovery", *target])]
    if wsl:
        model.append(
            (
                "拉取服务器模型",
                ["models", "fetch", *target, "--output", str(Path.home() / "models")],
            )
        )
        model.append(
            (
                "上传模型",
                [
                    "models",
                    "upload",
                    *target,
                    "--model",
                    str(Path.home() / "models/<MODEL>"),
                ],
            )
        )
    return [("Workflow", workflow), ("Container", container), ("Model", model)]


def suggestions(args) -> list:
    return [row for _, rows in groups(args) for row in rows]


def show(args) -> None:
    if getattr(args, "command", None) == "checkhealth":
        return
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
    if getattr(args, "command", None) != "containers":
        return
    show_dev(args)


def show_dev(args):
    stream = sys.stderr
    host = getattr(args, "host", None)
    dev = Path(__file__).resolve().parents[4] / "code/vllm-ascend-workspace/dev"
    selected_host = host[0] if isinstance(host, list) and host else host
    connection = []
    if is_wsl() or (selected_host and selected_host != "localhost"):
        name = (
            getattr(args, "container", None) or getattr(args, "name", None)
            if getattr(args, "container_action", "create") == "create"
            else None
        )
        connection = [
            "--host",
            selected_host or "<HOST>",
            "--container",
            name or "<CONTAINER>",
        ]
    config = "qwen3.5-0.8b.py"
    print("  " + color("vLLM Ascend Dev", "1;35", stream), file=stream)
    for label, arguments in [
        ("安装开发工具", ["devtools", "install"]),
        ("安装 vllm-ascend", ["vllm-ascend", "install"]),
        ("编译算子并安装 vllm-ascend", ["vllm-ascend", "install", "--compile-ops"]),
        ("启动服务", ["vllm-ascend", "serve", "--script", config]),
        ("停止服务", ["vllm-ascend", "serve", "stop"]),
    ]:
        line = shlex.join(["python3", str(dev), *arguments, *connection])
        print("    {}  {}".format(color(label, "36", stream), line), file=stream)
