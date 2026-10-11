#!/usr/bin/env python3
"""从 SSH config 筛选具体主机别名，使用 ssh-copy-id 批量安装已有公钥。

DEPENDENCIES:
    OS: WSL
    pip install sshconf==0.2.7
    ssh-copy-id
    sudo apt install sshpass。

USAGE:
    python3 ~/ai-infra-dev/scripts/install-ssh-keys.py --config ~/.ssh/config \
        --ssh /usr/bin/ssh --public-key ~/.ssh/id_ed25519.pub \
        --host-glob 'xiaoqiaoling*' --timeout 120 --dry-run


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
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from sshconf import SshConfigFile


def host_passwords(config: Path) -> dict[str, str | None]:
    """枚举具体别名及所在 Host 段的 Password，字段名不区分大小写。"""
    # sshconf 不识别关键字的等号分隔写法，先统一为标准空格写法。
    lines = [re.sub(r"^(\s*\w+)\s*=\s*", r"\1 ", line)
             for line in config.read_text(encoding="utf-8-sig").splitlines()]
    # sshconf 直接截断 #，先隐藏密码值，保护引号内的特殊字符。
    passwords: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^\s*password\s+(.+)$", line, re.IGNORECASE)
        if match:
            try:
                values = shlex.split(match[1], comments=True)
            except ValueError:
                raise ValueError("Password 的引号格式无效") from None
            if len(values) != 1:
                raise ValueError("Password 必须是单个值；含空格或 # 时请使用引号")
            token = f"password-{index}"
            passwords[token] = values[0]
            lines[index] = f"Password {token}"
    parsed = SshConfigFile(lines)
    hosts: dict[str, str | None] = {}
    for group in parsed.hosts():
        password = parsed.host(group).get("password")
        if password is not None:
            if not isinstance(password, str):
                raise ValueError("同一 Host 段只能配置一个 Password")
            password = passwords[password]
            if "\n" in password or "\r" in password:
                raise ValueError("Password 不允许包含换行符")
        for alias in shlex.split(group):
            if any(char in alias for char in "*?!["):
                continue
            if alias.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.:\-]+", alias):
                raise ValueError(f"{config}：不支持的主机别名：{alias}")
            hosts.setdefault(alias, password)
    return hosts


@contextmanager
def ssh_environment(executable: str):
    """ssh-copy-id 固定调用 ssh，通过临时 PATH 转发到指定程序。"""
    resolved = shutil.which(str(Path(executable).expanduser()))
    if resolved is None:
        raise ValueError(f"找不到 SSH 程序：{executable}")
    with tempfile.TemporaryDirectory(prefix="install-ssh-keys-") as directory:
        wrapper = Path(directory) / "ssh"
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


def copy_key(command: list[str], password: str | None, environment: dict[str, str], timeout: float) -> int:
    if password is not None and shutil.which("sshpass") is None:
        raise ValueError("配置了 Password，但找不到 sshpass；请先安装 sshpass")
    with authentication_command(command, password) as (arguments, descriptors):
        return subprocess.run(
            arguments, pass_fds=descriptors, env=environment, timeout=timeout, check=False,
        ).returncode


def print_preview(command: list[str], executable: str, has_password: bool) -> None:
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


def parse_arguments() -> argparse.Namespace:
    """集中定义命令行接口。"""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--config", required=True, help="SSH 配置文件路径（在 WSL 中读取）")
    parser.add_argument("--ssh", default="ssh", help="SSH 程序或包装脚本路径（默认：ssh）")
    # 所有本地路径统一使用 WSL/Linux 格式，直接交给指定客户端。
    parser.add_argument("--public-key", required=True, help="要安装的公钥文件路径（.pub）")
    parser.add_argument("--host-glob", required=True, action="append", help="主机别名 glob，可重复指定；匹配任一模式即选中，例如 'suzhou-*'")
    parser.add_argument("--dry-run", action="store_true", help="仅打印目标和命令，不连接远端、不修改文件")
    parser.add_argument("--timeout", type=float, default=20, help="每台安装超时秒数，包括交互等待（默认：20）")
    return parser.parse_args()


def install_host(command: list[str], host: str, password: str | None,
                 environment: dict[str, str], timeout: float) -> bool:
    """处理单机安装结果；失败不阻断批量处理。"""
    print(f"[安装] {host}", flush=True)
    try:
        result = copy_key(command, password, environment, timeout)
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
    key = Path(args.public_key).expanduser().resolve()
    hosts = {alias: password for alias, password in host_passwords(config).items()
             if any(fnmatch.fnmatchcase(alias, pattern) for pattern in args.host_glob)}
    failures = 0
    with ssh_environment(args.ssh) as environment:
        for host, password in hosts.items():
            command = ["ssh-copy-id", "-i", str(key), "-F", str(config),
                       "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", host]
            if args.dry_run:
                print(f"# {host}", flush=True)
                print_preview(command, args.ssh, password is not None)
            else:
                failures += not install_host(command, host, password, environment, args.timeout)
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
