"""只读计算用户 home 选址；由 Ansible 在远端执行。"""

import json
import os
from pathlib import Path
import pwd
import subprocess
import sys

HOME_MIN_FREE = 40 * 1024**3
CANDIDATES = ("/home", "/data", "/data1", "/data2", "/data3")


def available(path):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def plan_home(name):
    candidates = [Path(path) for path in CANDIDATES if Path(path).is_dir()]
    if not candidates:
        raise ValueError("没有可用的 home 候选目录")
    destination = max(candidates, key=available)
    free = available(destination)
    if free < HOME_MIN_FREE:
        raise ValueError("所有 home 候选目录的可用空间均不足 40 GiB")
    try:
        account = pwd.getpwnam(name)
    except KeyError:
        account = None
    if account:
        current = Path(account.pw_dir)
        if not current.is_dir() or current.is_symlink():
            raise ValueError(f"当前 home 不存在或是符号链接：{current}")
        if available(current) >= HOME_MIN_FREE:
            return {"home": str(current), "move": False}
        if current.stat().st_uid != account.pw_uid:
            raise ValueError(f"home 所有者不匹配：{current}")
        if current.stat().st_dev == destination.stat().st_dev:
            raise ValueError("没有其他空间充足的文件系统可供迁移")
        # 迁移不处理嵌套挂载，避免遗漏数据或修改其他挂载的数据。
        mounts = Path("/proc/self/mountinfo").read_text().splitlines()
        prefix = str(current.resolve()) + "/"
        for line in mounts:
            mount = line.split()[4]
            for escaped, literal in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
                mount = mount.replace(escaped, literal)
            if mount == str(current.resolve()) or mount.startswith(prefix):
                raise ValueError(f"home 包含挂载点，不能自动迁移：{mount}")
        for process in Path("/proc").iterdir():
            if process.name.isdigit():
                try:
                    if process.stat().st_uid == account.pw_uid:
                        raise ValueError(
                            f"用户 {name} 有运行进程，请退出会话并停止任务后重试"
                        )
                except FileNotFoundError:
                    pass
        size = int(
            subprocess.check_output(
                ["du", "-s", "-B1", "--", str(current)], text=True
            ).split()[0]
        )
        if free - size < HOME_MIN_FREE:
            raise ValueError("目标空间不足以迁移 home 并保留 40 GiB 可用空间")
    target = destination / name
    if os.path.lexists(target):
        raise ValueError(f"目标 home 已存在，不自动覆盖：{target}")
    if account and (
        target == current or current in target.parents or target in current.parents
    ):
        raise ValueError("新旧 home 路径不能互相包含")
    return {"home": str(target), "move": account is not None}


if __name__ == "__main__":
    try:
        print(json.dumps(plan_home(sys.argv[1])))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
