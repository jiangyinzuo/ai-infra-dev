#!/usr/bin/env python3
"""发现本地或远端 Hugging Face 模型候选目录（Python 3.9+）。"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Optional, Union

from ssh_common.config import REMOTE_USER
from ssh_common.remote import (
    SSH_DIR,
    add_host_argument,
    checked_command,
    host_alias,
    is_remote_host,
)

DEFAULT_ROOTS = tuple(
    f"/data{suffix}/{name}"
    for suffix in ("", "1", "2", "3")
    for name in ("weight", "weights", "model", "models")
)


def _raise_walk_error(error: OSError) -> None:
    raise error


def discover_models(
    host: Optional[str] = None,
    roots: Optional[Iterable[Union[str, Path]]] = None,
    *,
    timeout: float = 60,
) -> list[str]:
    """返回排序、去重的绝对路径；host 省略或 localhost 时扫描本机。

    roots 默认为 DEFAULT_ROOTS，空列表表示不扫描。递归扫描但不跟随子目录
    符号链接；配置和权重文件允许是有效符号链接。不验证配置内容、分片
    完整性或推理框架兼容性。缺失根目录跳过，其他文件系统错误抛出。
    timeout 仅限制远端 SSH 调用；SSH 失败抛出 CalledProcessError。
    """
    search_roots = [
        os.fspath(root) for root in (DEFAULT_ROOTS if roots is None else roots)
    ]
    if is_remote_host(host):
        host_alias(host)
        command = 'python3 "$HOME"/ai-infra-dev/ssh/discovery_models.py --json'
        # Repeatable --root cannot distinguish omitted roots from an empty list.
        if not search_roots:
            return []
        command += " " + shlex.join([f"--root={root}" for root in search_roots])
        result = subprocess.run(
            [
                str(SSH_DIR / "ssh2"),
                "-T",
                f"{REMOTE_USER}@{host}",
                checked_command(host, command),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return json.loads(result.stdout)

    models = set()
    for value in search_roots:
        root = Path(value).expanduser().absolute()
        try:
            root.stat()
        except FileNotFoundError:
            continue
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        for directory, _, filenames in os.walk(
            root, followlinks=False, onerror=_raise_walk_error
        ):
            if "config.json" not in filenames:
                continue
            path = Path(directory)
            if not (path / "config.json").is_file():
                continue
            if any(
                (
                    name.endswith(".safetensors")
                    or (name.startswith("pytorch_model") and name.endswith(".bin"))
                )
                and (path / name).is_file()
                for name in filenames
            ):
                models.add(str(path))
    return sorted(models)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_host_argument(parser)
    parser.add_argument(
        "--root", action="append", help="搜索根目录；可重复，覆盖默认的 16 个目录"
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 路径数组")
    parser.add_argument("--timeout", type=float, default=60, help="远端调用超时秒数")
    args = parser.parse_args(argv)
    try:
        models = discover_models(args.host, args.root, timeout=args.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.rstrip(), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(models, ensure_ascii=False))
    else:
        for model in models:
            print(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
