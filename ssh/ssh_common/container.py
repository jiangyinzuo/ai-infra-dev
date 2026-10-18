"""识别常见容器环境；启发式判断，不是安全隔离边界。"""

from pathlib import Path
import re
import socket


def is_container() -> bool:
    if socket.gethostname().endswith("-container") or any(
        Path(path).exists()
        for path in ("/.dockerenv", "/run/.containerenv", "/run/systemd/container")
    ):
        return True
    for path in ("/proc/self/cgroup", "/proc/1/cgroup"):
        if re.search(
            r"(?:docker|kubepods|libpod|lxc)(?:[/:.-]|$)", Path(path).read_text()
        ):
            return True
    return False
