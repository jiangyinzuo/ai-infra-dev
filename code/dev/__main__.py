#!/usr/bin/env python3
"""在容器中管理 vLLM 服务、安装 Ascend 插件；也可从 WSL 驱动远程容器。"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ssh"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_common.remote import checked_command, host_alias, is_wsl, ssh_command
from ssh_common.container import is_container
from container_command import container_exec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    resources = parser.add_subparsers(dest="resource", required=True)
    for resource in ("vllm", "vllm-ascend", "devtools"):
        group = resources.add_parser(resource)
        actions = group.add_subparsers(dest="action", required=True)
        if resource == "vllm":
            actions.add_parser("install", help="查找已发布 CUDA wheel 并执行 uv pip install -e .")
        if resource != "vllm":
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
                install.add_argument(
                    "--remove-csrc-build",
                    action="store_true",
                    help="安装前删除 csrc/build，清除跨 Python/镜像复用的构建配置",
                )
        if resource != "devtools":
            serve = actions.add_parser("serve", help="后台启动 GPU 服务" if resource == "vllm" else "后台启动 NPU 服务")
            serve.add_argument("serve_action", nargs="?", choices=["stop"])
            serve.add_argument("--host", type=host_alias)
            serve.add_argument("--container")
            serve.add_argument(
                "--model", help="本地模型目录；省略时使用配置默认模型或交互选择"
            )
            serve.add_argument(
                "--script",
                help="serve_config/ 下的 Python 配置文件名，例如 qwen3.5-0.8b.py",
            )
            serve.add_argument(
                "--devices",
                help="GPU index/UUID，逗号分隔；省略时选择空闲卡" if resource == "vllm" else "NPU 全局 IDs，逗号分隔；省略时选择空闲卡",
            )
            serve.add_argument(
                "--no-hints", action="store_true", help=argparse.SUPPRESS
            )
            serve.add_argument(
                "--print-log", action="store_true", help=argparse.SUPPRESS
            )
    return parser.parse_args(argv), parser


def execute(args, parser):
    match args.resource, args.action:
        case "vllm", "install":
            return install_gpu()
    validate_serve(args, parser)
    if is_wsl():
        return execute_remote(args, parser)
    if args.host or args.container:
        parser.error("--host 和 --container 仅用于 WSL 远程调用")
    match args.resource, args.action:
        case "vllm" | "vllm-ascend", "serve":
            import serve

            return serve.execute(args)
        case "devtools", "install":
            validate_container_install()
            return install_devtools()
        case "vllm-ascend", "install":
            validate_container_install()
            return install_ascend(args)
        case _:
            parser.error("不支持的命令组合")


def validate_serve(args, parser):
    if (
        args.action == "serve"
        and args.serve_action != "stop"
        and not args.script
        and not args.print_log
    ):
        parser.error("启动服务必须指定 --script，命令及参数由用户维护")


def remote_command(args):
    # 本项目容器共享宿主机 HOME；在容器内展开 HOME，且用 login shell 加载工具环境。
    command = 'exec python3 "$HOME"/ai-infra-dev/code/dev'
    command += " " + shlex.join([args.resource, args.action])
    command += " --no-hints"
    if getattr(args, "compile_ops", False):
        command += " --compile-ops"
    if getattr(args, "remove_csrc_build", False):
        command += " --remove-csrc-build"
    if args.action == "serve":
        if args.serve_action:
            command += " " + args.serve_action
        for key in ("model", "devices", "script"):
            if getattr(args, key):
                command += " " + shlex.join(["--" + key, getattr(args, key)])
    return command


def execute_remote(args, parser):
    if not args.host or not args.container:
        parser.error("WSL 调用必须同时指定 --host 和 --container")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container):
        parser.error("--container 必须为容器名称或 ID")
    interactive = args.resource != "devtools"
    remote = shlex.join(container_exec(
        args.container, ["bash", "-lc", remote_command(args)], interactive=interactive
    ))
    result = ssh_command(
        args.host, checked_command(args.host, remote), interactive=interactive
    )
    if result == 0 and args.action == "serve" and args.serve_action != "stop":
        from log_view import attach

        attach(args)
    return result


def install_gpu():
    from precompiled import find_vllm_precompiled_commit

    project = Path(__file__).resolve().parents[1] / "vllm-workspace/vllm"
    environment = dict(os.environ, **find_vllm_precompiled_commit())
    command = ["uv", "pip", "install", "-e", "."]
    print("工作目录：" + str(project), flush=True)
    print(
        shlex.join([f"{key}={environment[key]}" for key in (
            "VLLM_USE_PRECOMPILED", "VLLM_PRECOMPILED_WHEEL_COMMIT",
            "VLLM_PRECOMPILED_WHEEL_VARIANT",
        ) if key in environment]) + " " + shlex.join(command),
        flush=True,
    )
    return subprocess.run(command, cwd=project, env=environment, check=False).returncode


def validate_container_install():
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


def install_devtools():
    requirements = Path(__file__).resolve().parent / "requirements-dev.txt"
    command = ["pip", "install", "-r", str(requirements)]
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def install_ascend(args):
    project = Path(__file__).resolve().parents[1] / "vllm-ascend-workspace/vllm-ascend"
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
    if args.remove_csrc_build:
        # 不同镜像的 Python/ninja 路径可能不同；仅显式要求时清理共享源码的构建目录。
        cleanup = ["rm", "-rf", "--", str(project / "csrc/build")]
        print(shlex.join(cleanup), flush=True)
        subprocess.run(cleanup, check=True)
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


def main(argv=None):
    if sys.version_info < (3, 11):
        raise ValueError("dev 要求 Python >= 3.11")
    args, parser = parse_args(argv)
    result = execute(args, parser)
    if result == 0 and not getattr(args, "no_hints", True):
        sys.path.insert(0, str(ROOT / "ssh/container-dev/devtools"))
        from workflow.hints import show_dev

        show_dev(args)
    return result


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
