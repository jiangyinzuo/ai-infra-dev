#!/usr/bin/env python3
"""Select first-use rootless Podman storage using createuser's home disk policy."""

import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

from home_plan import CANDIDATES, available

PODMAN_MIN_FREE = 200 * 1024**3


def local_directory(path):
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-T", str(path), "-o", "FSTYPE"], text=True
    ).strip()
    # Use filesystems supported by the deployment, not remote home mounts.
    return filesystem in ("xfs", "ext4", "btrfs")


def select_storage(home, username):
    candidates = [home]
    for root in map(Path, CANDIDATES):
        if not root.is_dir():
            continue
        user_dir = root / username
        if (
            user_dir.is_dir()
            and not user_dir.is_symlink()
            or not user_dir.exists()
            and os.access(root, os.W_OK | os.X_OK)
        ):
            candidates.append(user_dir)
    usable = []
    for path in dict.fromkeys(candidates):
        parent = path if path.is_dir() else path.parent
        if (
            os.access(parent, os.W_OK | os.X_OK)
            and local_directory(parent)
            and available(parent) >= PODMAN_MIN_FREE
        ):
            usable.append((available(parent), path / "podman-storage"))
    if not usable:
        raise ValueError(
            "没有当前用户可写、至少剩余 200 GiB 的本地存储目录；请管理员分配目录后手动配置 storage.conf"
        )
    target = max(usable, key=lambda item: item[0])[1]
    if target.is_symlink() or (
        target.exists() and (not target.is_dir() or any(target.iterdir()))
    ):
        raise ValueError(f"目标不是空目录，不自动接管：{target}")
    return target


def configure():
    home = Path.home()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    config = config_home / "containers/storage.conf"
    if os.environ.get("CONTAINERS_STORAGE_CONF"):
        print("skip Podman storage (CONTAINERS_STORAGE_CONF is set)")
        return
    if os.path.lexists(config):
        print(f"skip {config} (existing storage configuration preserved)")
        return
    data_home = Path(os.environ.get("XDG_DATA_HOME") or home / ".local/share")
    default = data_home / "containers/storage"
    if default.exists() and any(default.iterdir()):
        print(f"skip Podman storage (existing default store preserved: {default})")
        return
    target = select_storage(home, pwd.getpwuid(os.getuid()).pw_name)
    target.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open("x") as stream:
        stream.write(
            '[storage]\ndriver = "overlay"\ngraphroot = '
            + json.dumps(str(target))
            + "\n"
        )
    print(f"configured {config}: graphroot={target}")
    print(
        "SELinux 启用时，新路径可能需要管理员配置容器存储标签；请运行 podman info --debug | grep graphRoot 验证。"
    )


if __name__ == "__main__":
    try:
        configure()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
