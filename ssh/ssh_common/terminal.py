"""统一终端颜色和命令展示，保留各业务的输出流选择。"""

import os
import shlex
import subprocess
import sys
from typing import Any, Optional, TextIO


def color(text: str, code: str, stream: Optional[TextIO] = None) -> str:
    stream = stream or sys.stdout
    if stream.isatty() and "NO_COLOR" not in os.environ:
        return f"\033[{code}m{text}\033[0m"
    return text


def green(text: str) -> str:
    return color(text, "32")


def yellow(text: str) -> str:
    return color(text, "33")


def title(text: str) -> str:
    return color(text, "1;35")


def run(*args: str, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    command = shlex.join(args)
    if kwargs.get("cwd") is not None:
        command = f"(cd {shlex.quote(str(kwargs['cwd']))} && {command})"
    print(command, flush=True)
    return subprocess.run(list(args), check=check, **kwargs)
