#!/usr/bin/env python3
"""发现本地或远端 Hugging Face 模型候选目录（Python 3.9+）。"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import shutil
from typing import Iterable, Optional, Union

from ssh_common.config import REMOTE_USER
from ssh_common.remote import (
    SSH_DIR,
    add_host_argument,
    checked_command,
    host_alias,
    is_remote_host,
    is_wsl,
)

DEFAULT_ROOTS = tuple(
    f"/data{suffix}/{name}".rstrip("/")
    for suffix in ("", "1", "2", "3")
    for name in ("", "weight", "weights", "model", "models")
)


def _raise_walk_error(error: OSError) -> None:
    if not isinstance(error, PermissionError):
        raise error


def discover_models(
    host: Optional[str] = None,
    roots: Optional[Iterable[Union[str, Path]]] = None,
    *,
    timeout: float = 60,
) -> list[str]:
    """返回排序、去重的绝对路径；服务器默认扫描本机，WSL 必须指定远端 host。

    roots 默认为 DEFAULT_ROOTS，空列表表示不扫描。递归扫描但不跟随子目录
    符号链接；配置和权重文件允许是有效符号链接。不验证配置内容、分片
    完整性或推理框架兼容性。缺失或无权限的目录跳过，其他文件系统错误抛出。
    timeout 仅限制远端 SSH 调用；SSH 失败抛出 CalledProcessError。
    """
    if is_wsl() and not is_remote_host(host):
        raise ValueError("WSL 中必须指定 --host 为远端服务器别名，不能扫描 WSL 本机")
    search_roots = [
        os.fspath(root) for root in (DEFAULT_ROOTS if roots is None else roots)
    ]
    if is_remote_host(host):
        host_alias(host)
        command = 'python3 "$HOME"/ai-infra-dev/ssh/container-dev/devtools/cli.py --no-hints models discovery --json'
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
        except (FileNotFoundError, PermissionError):
            continue
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        if not os.access(root, os.R_OK | os.X_OK):
            continue
        for directory, subdirs, filenames in os.walk(
            root, followlinks=False, onerror=_raise_walk_error
        ):
            subdirs[:] = [
                name
                for name in subdirs
                if os.access(Path(directory) / name, os.R_OK | os.X_OK)
            ]
            if "config.json" not in filenames:
                continue
            path = Path(directory)
            try:
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
            except PermissionError:
                continue
    return sorted(models)


def register(subparsers) -> None:
    models = subparsers.add_parser("models", help="模型管理")
    actions = models.add_subparsers(dest="models_action", required=True)
    parser = actions.add_parser("discovery", help="发现本地或远端模型候选目录")
    add_host_argument(parser)
    parser.add_argument(
        "--root", action="append", help="搜索根目录；可重复，覆盖默认的 20 个目录"
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 路径数组")
    parser.add_argument("--timeout", type=float, default=60, help="远端调用超时秒数")
    parser.set_defaults(handler=execute)
    fetch = actions.add_parser("fetch", help="选择服务器上的模型并下载到 WSL")
    add_host_argument(fetch, required=True)
    fetch.add_argument(
        "--output", required=True, help="本地父目录，模型保存到其下的同名目录"
    )
    source = fetch.add_mutually_exclusive_group()
    source.add_argument("--root", action="append", help="远端搜索目录，可重复")
    source.add_argument("--model", help="远端模型目录绝对路径；跳过发现和交互选择")
    fetch.add_argument("--timeout", type=float, default=60, help="远端模型发现超时秒数")
    fetch.set_defaults(handler=fetch_model)


def execute(args) -> int:
    models = discover_models(args.host, args.root, timeout=args.timeout)
    if args.json:
        print(json.dumps(models, ensure_ascii=False))
    else:
        for model in models:
            print(model)
    return 0


def fetch_model(args) -> int:
    if not is_wsl():
        raise ValueError("models fetch 暂时只支持下载到 WSL")
    if not is_remote_host(args.host):
        raise ValueError("请指定远端 --host，不能使用 localhost")
    host_alias(args.host)
    if not shutil.which("rsync"):
        raise ValueError("请先安装 rsync")
    if args.model:
        source = Path(args.model)
    else:
        candidates = discover_models(args.host, args.root, timeout=args.timeout)
        if not candidates:
            print("未发现模型候选目录")
            return 0
        for index, path in enumerate(candidates, 1):
            print(f"{index}) {path}")
        while True:
            choice = input("选择要下载的模型编号（q 取消）：").strip()
            if choice.lower() == "q":
                print("已取消")
                return 0
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                source = Path(candidates[int(choice) - 1])
                break
            print("请输入有效编号")
    if not source.is_absolute() or not source.name:
        raise ValueError("远端返回的模型目录不是有效绝对路径")
    destination = Path(args.output).expanduser().absolute() / source.name
    destination.mkdir(parents=True, exist_ok=True)
    print(f"下载 {args.host}:{source} -> {destination}", flush=True)
    # rsync -s protects remote spaces/metacharacters; -L materializes HF cache links.
    # Do not use --delete or --inplace: retain unrelated files and partial downloads.
    return subprocess.run(
        [
            "rsync",
            "-aL",
            "--protect-args",
            "--partial",
            "--info=progress2",
            "-e",
            shlex.quote(str(SSH_DIR / "ssh2")),
            "--rsync-path",
            checked_command(args.host, "exec rsync"),
            "--",
            f"{REMOTE_USER}@{args.host}:{source}/",
            str(destination) + "/",
        ],
        check=False,
    ).returncode
