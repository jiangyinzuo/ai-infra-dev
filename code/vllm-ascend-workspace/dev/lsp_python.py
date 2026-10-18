#!/usr/bin/env python3
"""为 Pyrefly 输出当前开发环境的解释器路径；stdout 只输出一行。"""

from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE.parents[1] / "ssh"))
from ssh_common.container import is_container
from ssh_common.remote import is_wsl


def interpreter():
    if is_wsl():
        python = WORKSPACE / ".venv/bin/python"
        if not python.is_file():
            raise ValueError("请先初始化 workspace/.venv")
        # 不 resolve：venv 的符号链接路径决定 Python 如何识别虚拟环境。
        return str(python)
    if not is_container():
        raise ValueError("LSP 仅支持 WSL 或开发容器")
    if sys.prefix != sys.base_prefix:
        raise ValueError("容器 LSP 必须使用全局 Python，不得使用虚拟环境")
    return sys.executable


if __name__ == "__main__":
    try:
        print(interpreter())
    except (OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        sys.exit(1)
