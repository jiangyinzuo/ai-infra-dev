"""创建容器前的宿主机检查。"""

from ssh_common.remote import is_wsl
from ssh_common.container import is_container


def require_host() -> None:
    if is_wsl():
        raise ValueError("不能在 WSL 本机创建容器，请通过 --host 指定服务器")
    if is_container():
        raise ValueError("不能在容器内创建容器，请在服务器宿主机执行")
