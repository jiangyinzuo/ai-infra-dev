"""先验证指定密钥的 root 登录，仅在需要时用密码安装公钥。"""

import getpass
from pathlib import Path
import subprocess

from ssh_auth import copy_key, ssh_environment
from ssh_config import host_passwords


def ensure_root_access(target, check=False):
    key = Path(target["public_key"])
    private = key.with_suffix("")
    if key.name != "id_ed25519.pub" or not key.is_file() or not private.is_file():
        raise ValueError(f"{target['host']}: 缺少 id_ed25519 密钥对：{key.parent}")
    windows = target["password_method"] == "windows-askpass"
    with ssh_environment(target["ssh"], windows) as environment:
        identity = str(private)
        options = [
            "-F",
            target["ssh_config"],
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ControlMaster=no",
        ]
        probe = [
            "ssh",
            *options,
            "-i",
            identity,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "IdentityAgent=none",
            "-o",
            "BatchMode=yes",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "root@" + target["host"],
            "true",
        ]

        def authenticated():
            return subprocess.run(
                probe,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        result = authenticated()
        if result.returncode == 0:
            print(f"{target['host']}: root 公钥登录可用，跳过安装", flush=True)
            return
        # 连接、主机密钥等故障不应触发密码安装。
        if "permission denied" not in result.stderr.lower():
            raise ValueError(
                f"{target['host']}: root SSH 检查失败：{result.stderr.strip()}"
            )
        if check:
            raise ValueError(
                f"{target['host']}: root 尚未支持指定密钥登录；请先 --apply"
            )
        password = host_passwords(Path(target["config"]))[target["host"]]
        if password is None:
            password = getpass.getpass(f"{target['host']} root SSH 密码：")
        command = ["ssh-copy-id", "-i", str(key), *options, "root@" + target["host"]]
        code = copy_key(command, password, environment, 120, target["password_method"])
        if code:
            raise ValueError(f"{target['host']}: root 公钥安装失败，退出码 {code}")
        result = authenticated()
        if result.returncode:
            raise ValueError(
                f"{target['host']}: 安装后 root 公钥登录验证失败：{result.stderr.strip()}"
            )
