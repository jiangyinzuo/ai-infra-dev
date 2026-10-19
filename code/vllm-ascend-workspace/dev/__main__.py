#!/usr/bin/env python3
"""在容器中 editable 安装 vllm-ascend；也可从 WSL 驱动远程容器。"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ssh"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_common.remote import checked_command, host_alias, is_wsl, ssh_command
from ssh_common.container import is_container
from ssh_common.config import USER_IN_CONTAINER


def main(argv=None):
    if sys.version_info < (3, 11):
        raise ValueError("dev 要求 Python >= 3.11")
    parser = argparse.ArgumentParser(description=__doc__)
    resources = parser.add_subparsers(dest="resource", required=True)
    for resource in ("vllm-ascend", "devtools"):
        group = resources.add_parser(resource)
        actions = group.add_subparsers(dest="action", required=True)
        install = actions.add_parser("install", help="安装 " + resource)
        install.add_argument(
            "--host", type=host_alias, help="WSL 远程服务器 SSH Host 别名"
        )
        install.add_argument("--container", help="远程容器名称或 ID")
        install.add_argument("--no-hints", action="store_true", help=argparse.SUPPRESS)
        if resource == "vllm-ascend":
            install.add_argument(
                "--compile-ops", action="store_true", help="编译算子（首次安装时使用）"
            )
            serve = actions.add_parser("serve", help="后台启动服务或停止本工具的服务")
            serve.add_argument("serve_action", nargs="?", choices=["stop"])
            serve.add_argument("--host", type=host_alias)
            serve.add_argument("--container")
            serve.add_argument("--model", help="模型目录；省略时交互选择")
            serve.add_argument(
                "--script",
                help="定义 config = ServeConfig(...) 的 Python 配置文件（容器内路径）",
            )
            serve.add_argument(
                "--devices", help="逗号分隔的空闲 NPU device IDs；省略时选择"
            )
            serve.add_argument(
                "--no-hints", action="store_true", help=argparse.SUPPRESS
            )
    args = parser.parse_args(argv)
    result = execute(args, parser)
    if result == 0 and not args.no_hints:
        sys.path.insert(0, str(ROOT / "ssh/container-dev/devtools"))
        from workflow.hints import show_dev

        show_dev(args)
    return result


def execute(args, parser):
    if args.action == "serve" and args.serve_action != "stop" and not args.script:
        parser.error("启动服务必须指定 --script，命令及参数由用户维护")
    if is_wsl():
        if not args.host or not args.container:
            parser.error("WSL 调用必须同时指定 --host 和 --container")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container):
            parser.error("--container 必须为容器名称或 ID")
        # 本项目容器共享宿主机 HOME；在容器内展开 HOME，且用 login shell 加载工具环境。
        command = 'exec python3 "$HOME"/ai-infra-dev/code/vllm-ascend-workspace/dev'
        command += " " + shlex.join([args.resource, args.action])
        command += " --no-hints"
        if getattr(args, "compile_ops", False):
            command += " --compile-ops"
        if args.action == "serve":
            if args.serve_action:
                command += " " + args.serve_action
            for key in ("model", "devices", "script"):
                if getattr(args, key):
                    command += " " + shlex.join(["--" + key, getattr(args, key)])
        if args.resource == "vllm-ascend":
            remote = shlex.join(
                [
                    "docker",
                    "exec",
                    "--user",
                    USER_IN_CONTAINER,
                    "-it",
                    args.container,
                    "bash",
                    "-lc",
                    command,
                ]
            )
            return ssh_command(
                args.host, checked_command(args.host, remote), interactive=True
            )
        remote = shlex.join(
            [
                "docker",
                "exec",
                "--user",
                USER_IN_CONTAINER,
                args.container,
                "bash",
                "-lc",
                command,
            ]
        )
        return ssh_command(args.host, checked_command(args.host, remote))
    if args.host or args.container:
        parser.error("--host 和 --container 仅用于 WSL 远程调用")
    if not is_container():
        raise ValueError(
            "禁止在宿主机编译；请在容器内执行，或从 WSL 指定 --host 和 --container"
        )
    if (
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("CONDA_PREFIX")
        or sys.prefix != sys.base_prefix
    ):
        raise ValueError("禁止在虚拟环境内构建；请使用容器自带的 pip")
    if args.action == "serve":
        import serve

        return serve.execute(args)
    if args.resource == "devtools":
        requirements = Path(__file__).resolve().parent / "requirements-dev.txt"
        command = ["pip", "install", "-r", str(requirements)]
        print(shlex.join(command), flush=True)
        return subprocess.run(command, check=False).returncode
    project = Path(__file__).resolve().parents[1] / "vllm-ascend"
    if not project.exists() and not project.is_symlink():
        url = (
            input(
                "vllm-ascend 仓库 URL [git@github.com:vllm-project/vllm-ascend.git]："
            ).strip()
            or "git@github.com:vllm-project/vllm-ascend.git"
        )
        branch = input("分支名 [main]：").strip() or "main"
        clone = [
            "git",
            "clone",
            "--depth=1000",
            "--branch",
            branch,
            "--",
            url,
            str(project),
        ]
        print(shlex.join(clone), flush=True)
        result = subprocess.run(clone, check=False)
        if result.returncode:
            return result.returncode
    if (
        not (project / "pyproject.toml").is_file()
        and not (project / "setup.py").is_file()
    ):
        raise ValueError("未找到 vllm-ascend 项目：" + str(project))
    environment = dict(
        os.environ,
        VLLM_TARGET_DEVICE="empty",
        COMPILE_CUSTOM_KERNELS="1" if args.compile_ops else "0",
    )
    command = ["pip", "install", "-v", "-e", ".", "--no-deps", "--no-build-isolation"]
    print("工作目录：" + str(project), flush=True)
    print(
        "VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS="
        + environment["COMPILE_CUSTOM_KERNELS"]
        + " "
        + shlex.join(command),
        flush=True,
    )
    return subprocess.run(command, cwd=project, env=environment, check=False).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        RuntimeError,
        EOFError,
        subprocess.SubprocessError,
    ) as error:
        print("error: " + str(error), file=sys.stderr)
        sys.exit(1)
