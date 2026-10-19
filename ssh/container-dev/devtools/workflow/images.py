#!/usr/bin/env python3
"""下载 NPU 容器镜像；WSL 必须指定 --host。"""

import os
from pathlib import Path
import subprocess
import tempfile

import re
import shlex
import shutil
from ssh_common.config import REMOTE_USER, REMOTE_DIR
from ssh_common.remote import (
    SSH_DIR,
    add_host_argument,
    checked_command,
    is_wsl,
    is_remote_host,
    run_remote_cli,
)
from workflow.sync import remote_path
from workflow.paths import DATA_ROOT


def register(subparsers):
    parser = subparsers.add_parser("images", help="镜像归档下载与上传")
    actions = parser.add_subparsers(dest="images_action", required=True)
    fetcher = actions.add_parser("download", help="下载镜像归档")
    add_host_argument(fetcher)
    fetcher.add_argument("--archive", required=True, help="完整镜像归档文件名")
    fetcher.set_defaults(handler=execute)
    uploader = actions.add_parser(
        "upload", help="从 WSL 上传镜像归档，不自动导入 Docker"
    )
    add_host_argument(uploader, required=True)
    uploader.add_argument(
        "--archive", required=True, type=Path, help="WSL 本地归档路径"
    )
    uploader.set_defaults(handler=upload)


def download(args):
    filename = args.archive
    match = re.fullmatch(
        r"vllm-ascend:dev-(\d+\.\d+\.\d+)(?:\.B\d+)?[.-]day(\d{8})-[A-Za-z0-9_.-]+\.tar\.gz",
        filename,
    )
    if not match:
        raise ValueError(
            "--archive 应为 vllm-ascend:dev-版本[.B构建号]-dayYYYYMMDD-平台.tar.gz"
        )
    version, date = match.groups()
    url = f"https://cmc-nkg-artifactory.cmc.tools.huawei.com/artifactory/cmc-nkg-inner/MindIE/vLLM%20Ascend/{version}.daily/{date[:6]}/{date}/vLLM-Ascend-images/{filename}"
    directory = DATA_ROOT / "containers"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / filename
    if archive.is_file():
        print(f"skip {archive} (archive exists)")
        return 0
    fd, name = tempfile.mkstemp(prefix=archive.name + ".tmp.", dir=directory)
    os.close(fd)
    try:
        print(f"download {url} -> {archive}", flush=True)
        result = subprocess.run(
            ["wget", "--no-check-certificate", "-O", name, url], check=False
        )
        if result.returncode:
            return result.returncode
        os.replace(name, archive)
    finally:
        Path(name).unlink(missing_ok=True)
    return 0


def execute(args):
    if is_wsl() and not args.host:
        raise ValueError("WSL 必须指定 --host（本地使用 localhost）")
    if is_remote_host(args.host):
        return run_remote_cli(
            args.host, ["images", "download", "--archive", args.archive]
        )
    return download(args)


def upload(args):
    if not is_wsl() or not is_remote_host(args.host):
        raise ValueError("images upload 必须从 WSL 指定远端 --host")
    archive = args.archive.expanduser().absolute()
    if not archive.is_file():
        raise ValueError("镜像归档不存在：" + str(archive))
    if not shutil.which("rsync"):
        raise ValueError("请先安装 rsync")
    destination = REMOTE_DIR + "/containers/"
    remote = "mkdir -p -- " + remote_path(destination) + " && exec rsync"
    return subprocess.run(
        [
            "rsync",
            "-rt",
            "--partial",
            "--protect-args",
            "--blocking-io",
            "--info=progress2",
            "-e",
            shlex.quote(str(SSH_DIR / "ssh2")),
            "--rsync-path",
            checked_command(args.host, remote),
            "--",
            str(archive),
            f"{REMOTE_USER}@{args.host}:{destination}",
        ],
        check=False,
    ).returncode
