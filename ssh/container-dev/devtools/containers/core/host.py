"""创建容器前的宿主机检查和资源读取。"""

import os
from pathlib import Path
import re
import socket

from ssh_common.remote import is_wsl


def require_host() -> None:
    if is_wsl():
        raise ValueError("不能在 WSL 本机创建容器，请通过 --host 指定服务器")
    # 容器检测没有通用且不可绕过的接口；组合常见运行时标记与本项目 hostname 约定。
    markers = (
        Path("/.dockerenv"),
        Path("/run/.containerenv"),
        Path("/run/systemd/container"),
    )
    if socket.gethostname().endswith("-container") or any(p.exists() for p in markers):
        raise ValueError("不能在容器内创建容器，请在服务器宿主机执行")
    for path in (Path("/proc/self/cgroup"), Path("/proc/1/cgroup")):
        text = path.read_text()
        if re.search(r"(?:docker|kubepods|libpod|lxc)(?:[/:.-]|$)", text):
            raise ValueError("不能在容器内创建容器，请在服务器宿主机执行")


def shm_capacity() -> int:
    require_host()
    info = os.statvfs("/dev/shm")
    size = info.f_frsize * info.f_blocks
    if size <= 0:
        raise ValueError("宿主机 /dev/shm 容量必须大于零")
    return size
