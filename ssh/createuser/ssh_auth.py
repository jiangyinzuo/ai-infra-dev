"""SSH 公钥安装共用的密码认证和客户端适配。"""

from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from ssh_hosts import windows_paths


@contextmanager
def ssh_environment(executable: str, windows_client: bool = False):
    """ssh-copy-id 固定调用 ssh，通过临时 PATH 转发到指定程序。"""
    resolved = shutil.which(str(Path(executable).expanduser()))
    if resolved is None:
        raise ValueError(f"找不到 SSH 程序：{executable}")
    with tempfile.TemporaryDirectory(prefix="createuser-ssh-") as directory:
        wrapper = Path(directory) / "ssh"
        if windows_client:
            wrapper.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nargs=()\n"
                "while (($#)); do\n"
                '  if [[ "$1" == -i ]]; then\n'
                '    args+=("-i" "$(wslpath -w "$2")"); shift 2\n'
                "  else\n"
                '    args+=("$1"); shift\n'
                "  fi\ndone\n"
                f'exec {shlex.quote(resolved)} "${{args[@]}}"\n'
            )
        else:
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(resolved)} "$@"\n')
        wrapper.chmod(0o700)
        yield {
            **os.environ,
            "PATH": directory + os.pathsep + os.environ.get("PATH", ""),
        }


@contextmanager
def authentication_command(command: list[str], password: str | None):
    """预览和执行共用命令构造；密码只通过继承的文件描述符传递。"""
    if password is None:
        yield command, ()
        return
    descriptor = os.memfd_create("ssh-password", os.MFD_CLOEXEC)
    try:
        os.write(descriptor, (password + "\n").encode())
        os.lseek(descriptor, 0, os.SEEK_SET)
        yield ["sshpass", "-d", str(descriptor), *command], (descriptor,)
    finally:
        os.close(descriptor)


@contextmanager
def windows_askpass(password: str, environment: dict[str, str]):
    """辅助脚本不含密码；仅通过 WSLENV 传入 Windows 子进程环境。"""
    home = windows_paths()["WINDOWS_SSH_HOME"]
    with tempfile.TemporaryDirectory(prefix="ssh-askpass-", dir=home) as directory:
        helper = Path(directory) / "askpass.cmd"
        helper.write_text(
            "@echo off\r\npowershell.exe -NoProfile -NonInteractive -Command "
            '"[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
            "[Console]::Out.WriteLine([Environment]::GetEnvironmentVariable('MY_SSH_PASSWORD'))"
            '"\r\n',
            encoding="utf-8",
        )
        variables = {
            "SSH_ASKPASS": subprocess.check_output(
                ["wslpath", "-w", str(helper)], text=True
            ).strip(),
            "SSH_ASKPASS_REQUIRE": "force",
            "MY_SSH_PASSWORD": password,
            "DISPLAY": "unused",
        }
        entries = [
            entry
            for entry in environment.get("WSLENV", "").split(":")
            if entry and entry.split("/")[0] not in variables
        ]
        yield {**environment, **variables, "WSLENV": ":".join([*entries, *variables])}


def copy_key(
    command: list[str],
    password: str | None,
    environment: dict[str, str],
    timeout: float,
    password_method: str,
) -> int:
    if password is not None and password_method == "windows-askpass":
        with windows_askpass(password, environment) as windows_environment:
            return subprocess.run(
                command,
                env=windows_environment,
                cwd="/mnt/c",
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            ).returncode
    if password is not None and shutil.which("sshpass") is None:
        raise ValueError("配置了 Password，但找不到 sshpass；请先安装 sshpass")
    with authentication_command(command, password) as (arguments, descriptors):
        return subprocess.run(
            arguments,
            pass_fds=descriptors,
            env=environment,
            timeout=timeout,
            check=False,
        ).returncode
