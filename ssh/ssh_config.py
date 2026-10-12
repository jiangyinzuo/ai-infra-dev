"""解析 SSH config 的具体 Host 和自定义 Password；不展开 Include 或通配符。"""

from pathlib import Path
import re
import shlex

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
