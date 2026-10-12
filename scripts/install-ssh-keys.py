#!/usr/bin/env python3
"""从 SSH config 筛选具体主机别名，使用 ssh-copy-id 批量安装已有公钥。

DEPENDENCIES:
    OS: WSL
    pip install sshconf==0.2.7
    ssh-copy-id
    sudo apt install sshpass。

SSHCONFIG Example:
    # IgnoreUnknown 必须在 Password 之前
    IgnoreUnknown Password
    Host xiaoqiaoling-one
        HostName 192.0.2.21
        User root
        PassWord "密码含空格或#时需加引号"

LIMIT:
    只枚举当前文件中的具体 Host 别名，不展开 Include 或通配符 Host。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fnmatch
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from ssh_config import host_passwords


@contextmanager
def ssh_environment(executable: str, windows_client: bool = False):
    """ssh-copy-id 固定调用 ssh，通过临时 PATH 转发到指定程序。"""
    resolved = shutil.which(str(Path(executable).expanduser()))
    if resolved is None:
        raise ValueError(f"找不到 SSH 程序：{executable}")
    with tempfile.TemporaryDirectory(prefix="install-ssh-keys-") as directory:
        wrapper = Path(directory) / "ssh"
        if windows_client:
            wrapper.write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\nargs=()\n'
                'while (($#)); do\n'
                '  if [[ "$1" == -i ]]; then\n'
                '    args+=("-i" "$(wslpath -w "$2")"); shift 2\n'
                '  else\n'
                '    args+=("$1"); shift\n'
                '  fi\ndone\n'
                f'exec {shlex.quote(resolved)} "${{args[@]}}"\n'
            )
        else:
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(resolved)} "$@"\n')
        wrapper.chmod(0o700)
        yield {**os.environ, "PATH": directory + os.pathsep + os.environ.get("PATH", "")}


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
    profile = subprocess.check_output(
        ["cmd.exe", "/d", "/c", "chcp 65001 >nul & echo %USERPROFILE%"],
        cwd="/mnt/c", text=True, encoding="utf-8",
    ).strip()
    home = subprocess.check_output(["wslpath", "-u", profile], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="ssh-askpass-", dir=Path(home) / ".ssh") as directory:
        helper = Path(directory) / "askpass.cmd"
        helper.write_text(
            '@echo off\r\npowershell.exe -NoProfile -NonInteractive -Command '
            '"[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
            "[Console]::Out.WriteLine([Environment]::GetEnvironmentVariable('MY_SSH_PASSWORD'))"
            '"\r\n', encoding="utf-8",
        )
        variables = {
            "SSH_ASKPASS": subprocess.check_output(["wslpath", "-w", str(helper)], text=True).strip(),
            "SSH_ASKPASS_REQUIRE": "force", "MY_SSH_PASSWORD": password, "DISPLAY": "unused",
        }
        entries = [entry for entry in environment.get("WSLENV", "").split(":")
                   if entry and entry.split("/")[0] not in variables]
        yield {**environment, **variables, "WSLENV": ":".join([*entries, *variables])}


def copy_key(command: list[str], password: str | None, environment: dict[str, str],
             timeout: float, password_method: str) -> int:
    if password is not None and password_method == "windows-askpass":
        with windows_askpass(password, environment) as windows_environment:
            return subprocess.run(
                command, env=windows_environment, cwd="/mnt/c", stdin=subprocess.DEVNULL,
                timeout=timeout, check=False,
            ).returncode
    if password is not None and shutil.which("sshpass") is None:
        raise ValueError("配置了 Password，但找不到 sshpass；请先安装 sshpass")
    with authentication_command(command, password) as (arguments, descriptors):
        return subprocess.run(
            arguments, pass_fds=descriptors, env=environment, timeout=timeout, check=False,
        ).returncode


def print_sshpass_preview(command: list[str], executable: str, has_password: bool) -> None:
    """输出独立 Bash 子 shell，重建转发环境，不暴露配置密码。"""
    program = shutil.which(str(Path(executable).expanduser()))
    print("(")
    print("  set -e")
    print(f"  export SSH_PROGRAM={shlex.quote(program or executable)}")
    print('  ssh_dir=$(mktemp -d)')
    print("  trap 'rm -rf -- \"$ssh_dir\"' EXIT")
    print("  printf '%s\\n' '#!/bin/sh' 'exec \"$SSH_PROGRAM\" \"$@\"' > \"$ssh_dir/ssh\"")
    print('  chmod 700 "$ssh_dir/ssh"')
    print('  export PATH="$ssh_dir:$PATH"')
    if has_password:
        print("  # 安全起见不输出配置密码；粘贴执行时隐藏输入。")
        print("  IFS= read -r -s -p 'SSH 密码：' password </dev/tty")
        print("  printf '\\n' >/dev/tty")
        print("  exec 3<<<\"$password\"")
        print("  unset password")
        command = ["sshpass", "-d", "3", *command]
    print("  " + shlex.join(command))
    print(")", flush=True)


def print_windows_preview(args: argparse.Namespace, config: Path, ssh_config: str,
                          key: Path, host: str) -> None:
    """输出重新读取配置密码的单机命令，不暴露秘密或依赖 sshpass。"""
    print(shlex.join([
        sys.executable, str(Path(__file__).resolve()), "--config", str(config),
        "--ssh-config", ssh_config, "--ssh", args.ssh,
        "--public-key", str(key), "--host-glob", host,
        "--password-method", args.password_method, "--timeout", str(args.timeout),
    ]), flush=True)


def parse_arguments() -> argparse.Namespace:
    """集中定义命令行接口。"""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--config", required=True, help="SSH 配置文件路径（在 WSL 中读取）")
    parser.add_argument("--ssh-config", help="传给 SSH 客户端的配置路径；默认同 --config，Windows 客户端需 Windows 路径")
    parser.add_argument("--ssh", default="ssh", help="SSH 程序或包装脚本路径（默认：ssh）")
    parser.add_argument("--password-method", choices=("sshpass", "windows-askpass"), default="sshpass",
                        help="配置密码输入机制：WSL SSH 用 sshpass，Windows SSH 用 windows-askpass")
    # 公钥由 WSL 中的 ssh-copy-id 读取，始终使用 WSL/Linux 路径。
    parser.add_argument("--public-key", required=True, help="要安装的公钥文件路径（.pub）")
    parser.add_argument("--host-glob", required=True, action="append", help="主机别名 glob，可重复指定；匹配任一模式即选中，例如 'suzhou-*'")
    parser.add_argument("--dry-run", action="store_true", help="仅打印目标和命令，不连接远端、不修改文件")
    parser.add_argument("--timeout", type=float, default=20, help="每台安装超时秒数，包括交互等待（默认：20）")
    return parser.parse_args()


def install_host(command: list[str], host: str, password: str | None,
                 environment: dict[str, str], timeout: float, password_method: str) -> bool:
    """处理单机安装结果；失败不阻断批量处理。"""
    print(f"[安装] {host}", flush=True)
    try:
        result = copy_key(command, password, environment, timeout, password_method)
        if result:
            print(f"[失败] {host}：安装返回码 {result}", file=sys.stderr)
            return False
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"[失败] {host}：{exc}", file=sys.stderr)
        return False
    print(f"[成功] {host}", flush=True)
    return True


def run_batch(args: argparse.Namespace) -> int:
    """筛选目标，再按相同命令计划预览或安装。"""
    if args.timeout <= 0:
        raise ValueError("超时必须大于 0")
    config = Path(args.config).expanduser().resolve()
    ssh_config = args.ssh_config if args.ssh_config is not None else str(config)
    key = Path(args.public_key).expanduser().resolve()
    hosts = {alias: password for alias, password in host_passwords(config).items()
             if any(fnmatch.fnmatchcase(alias, pattern) for pattern in args.host_glob)}
    failures = 0
    with ssh_environment(args.ssh, args.password_method == "windows-askpass") as environment:
        for host, password in hosts.items():
            command = ["ssh-copy-id", "-i", str(key), "-F", ssh_config,
                       "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", host]
            if args.dry_run:
                print(f"# {host}", flush=True)
                if args.password_method == "windows-askpass":
                    print_windows_preview(args, config, ssh_config, key, host)
                else:
                    print_sshpass_preview(command, args.ssh, password is not None)
            else:
                failures += not install_host(command, host, password, environment,
                                             args.timeout, args.password_method)
    action = "计划" if args.dry_run else "处理"
    print(f"# {action} {len(hosts)} 台，失败 {failures} 台")
    return 1 if failures else 0


def main() -> int:
    args = parse_arguments()
    try:
        return run_batch(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
