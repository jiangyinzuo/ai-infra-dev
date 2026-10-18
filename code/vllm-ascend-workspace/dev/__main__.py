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
from ssh_common.remote import checked_command, host_alias, is_wsl, ssh_command
from ssh_common.container import is_container


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
        if resource == "vllm-ascend":
            install.add_argument(
                "--compile-ops", action="store_true", help="编译算子（首次安装时使用）"
            )
    args = parser.parse_args(argv)
    if is_wsl():
        if not args.host or not args.container:
            parser.error("WSL 调用必须同时指定 --host 和 --container")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container):
            parser.error("--container 必须为容器名称或 ID")
        # 本项目容器共享宿主机 HOME；在容器内展开 HOME，且用 login shell 加载工具环境。
        command = 'exec python3 "$HOME"/ai-infra-dev/code/vllm-ascend-workspace/dev'
        command += " " + shlex.join([args.resource, args.action])
        if getattr(args, "compile_ops", False):
            command += " --compile-ops"
        remote = shlex.join(["docker", "exec", args.container, "bash", "-lc", command])
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
    if args.resource == "devtools":
        requirements = Path(__file__).resolve().parent / "requirements-dev.txt"
        command = ["pip", "install", "-r", str(requirements)]
        print(shlex.join(command), flush=True)
        return subprocess.run(command, check=False).returncode
    project = Path(__file__).resolve().parents[1] / "vllm-ascend"
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
    except (OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        sys.exit(1)
