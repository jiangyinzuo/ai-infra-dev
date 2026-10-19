#!/usr/bin/env python3
"""下载 NPU 容器镜像；WSL 必须指定 --host。"""

import os
from pathlib import Path
import subprocess
import tempfile

import re
import json
import shlex
import shutil
import sys
import argparse
from datetime import date
from ssh_common.config import REMOTE_USER, REMOTE_DIR
from ssh_common.remote import (
    SSH_DIR,
    add_host_argument,
    checked_command,
    is_wsl,
    is_remote_host,
    run_remote_cli,
    host_alias,
)
from workflow.image_store import ensure_image
from workflow.sync import remote_path
from workflow.paths import DATA_ROOT


def register(subparsers):
    parser = subparsers.add_parser("images", help="镜像归档下载与上传")
    actions = parser.add_subparsers(dest="images_action", required=True)
    fetcher = actions.add_parser(
        "download",
        help="下载镜像归档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  images download --host localhost --from quay --archive nightly-main-a3-openeuler.tar.gz
  URL: https://cmc-lfg-artifactory.cmc.tools.huawei.com/artifactory/generic-8166/hw-cr/quay.io/ascend/vllm-ascend/nightly-main-a3-openeuler.tar.gz
  images download --host localhost --from cmc.rnd --archive vllm-ascend:dev-26.2.0.B109-day20261019-800I-A3-py311-openEuler24.03-lts-aarch64.tar.gz
  URL: https://cmc-nkg-artifactory.cmc.tools.huawei.com/artifactory/cmc-nkg-inner/MindIE/vLLM%20Ascend/26.2.0.daily/202610/20261019/vLLM-Ascend-images/vllm-ascend:dev-26.2.0.B109-day20261019-800I-A3-py311-openEuler24.03-lts-aarch64.tar.gz
""",
    )
    add_host_argument(fetcher)
    fetcher.add_argument("--archive", required=True, help="完整镜像归档文件名")
    fetcher.add_argument(
        "--from",
        dest="source",
        choices=["quay", "cmc.rnd"],
        default="cmc.rnd",
        help="quay：CMC-LFG Quay 归档；cmc.rnd：原 daily 地址（默认）",
    )
    fetcher.set_defaults(handler=execute)
    uploader = actions.add_parser("upload", help="从 WSL 上传镜像归档并导入 Docker")
    add_host_argument(uploader, required=True)
    uploader.add_argument(
        "--archive", required=True, type=Path, help="WSL 本地归档路径"
    )
    uploader.set_defaults(handler=upload)
    listing = actions.add_parser(
        "ls", help="列出 vllm-ascend 与 quay.io/ascend/vllm-ascend 镜像"
    )
    add_host_argument(listing)
    listing.set_defaults(handler=list_images)


def download_location(filename, source="cmc.rnd"):
    if source == "quay":
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\.tar\.gz", filename):
            raise ValueError("Quay 归档名应为 TAG.tar.gz，不包含路径或 repository")
        url = (
            "https://cmc-lfg-artifactory.cmc.tools.huawei.com/artifactory/generic-8166/hw-cr/quay.io/ascend/vllm-ascend/"
            + filename
        )
        saved = (
            filename[:-7] + "-" + date.today().strftime("%Y%m%d") + ".tar.gz"
            if filename.startswith("nightly-")
            else filename
        )
        return url, saved
    # Internal daily archives encode the version/date in their complete filename.
    match = re.fullmatch(
        r"vllm-ascend:dev-(\d+\.\d+\.\d+)(?:\.B\d+)?[.-]day(\d{8})-[A-Za-z0-9_.-]+\.tar\.gz",
        filename,
    )
    if not match:
        raise ValueError(
            "--archive 应为 vllm-ascend:dev-版本[.B构建号]-dayYYYYMMDD-平台.tar.gz"
        )
    version, image_date = match.groups()
    url = f"https://cmc-nkg-artifactory.cmc.tools.huawei.com/artifactory/cmc-nkg-inner/MindIE/vLLM%20Ascend/{version}.daily/{image_date[:6]}/{image_date}/vLLM-Ascend-images/{filename}"
    return url, filename


def download(args):
    url, filename = download_location(args.archive, getattr(args, "source", "cmc.rnd"))
    directory = DATA_ROOT / "containers"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / filename
    server = not is_wsl()
    image = image_from_archive(filename)
    emit = lambda command: print(shlex.join(command), flush=True)
    if server and ensure_image(directory, image, emit) != "missing":
        return 0
    if archive.is_file():
        print(f"skip {archive} (archive exists)")
        return 0
    fd, name = tempfile.mkstemp(prefix=archive.name + ".tmp.", dir=directory)
    os.close(fd)
    try:
        print(f"download {url} -> {archive}", flush=True)
        result = subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "-O",
                name,
                "--progress=bar:force:noscroll",
                url,
            ],
            check=False,
        )
        if result.returncode:
            return result.returncode
        os.replace(name, archive)
    finally:
        Path(name).unlink(missing_ok=True)
    if server and ensure_image(directory, image, emit) == "missing":
        raise RuntimeError("下载后未找到镜像归档")
    return 0


def execute(args):
    if is_wsl() and not args.host:
        raise ValueError("WSL 必须指定 --host（本地使用 localhost）")
    if is_remote_host(args.host):
        return run_remote_cli(
            args.host,
            ["images", "download", "--archive", args.archive, "--from", args.source],
        )
    return download(args)


def image_from_archive(filename):
    quay = re.fullmatch(
        r"((?:nightly-|v\d|qwen)[A-Za-z0-9_.-]*?)(?:-(\d{8}))?\.tar\.gz", filename
    )
    if quay:
        return (
            "quay.io/ascend/vllm-ascend:"
            + quay[1]
            + ("-" + quay[2] if quay[2] and quay[1].startswith("nightly-") else "")
        )
    match = re.fullmatch(
        r"vllm-ascend[:_]([A-Za-z0-9_][A-Za-z0-9_.-]{0,127})\.tar\.gz", filename
    )
    if not match:
        raise ValueError(
            "归档名应为 vllm-ascend:TAG.tar.gz、vllm-ascend_TAG.tar.gz 或 Quay 的 nightly-/v版本/qwen 开头的 TAG[-YYYYMMDD].tar.gz"
        )
    return "vllm-ascend:" + match[1]


REMOTE_LOAD = "from workflow.image_store import ensure_image; import json,sys; print(json.dumps({'status':ensure_image(sys.argv[1],sys.argv[2])}),flush=True)"


def remote_load(host, image):
    host_alias(host)
    command = shlex.join(
        [
            "python3",
            "-c",
            "import sys; from pathlib import Path; sys.path.insert(0,str(Path.home()/'ai-infra-dev/ssh/container-dev/devtools')); "
            + REMOTE_LOAD,
            REMOTE_DIR + "/containers",
            image,
        ]
    )
    with subprocess.Popen(
        [
            str(SSH_DIR / "ssh2"),
            "-T",
            f"{REMOTE_USER}@{host}",
            checked_command(host, command),
        ],
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        status = None
        for line in process.stdout:
            event = json.loads(line)
            if "command" in event:
                print(event["command"], flush=True)
            else:
                status = event["status"]
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)
        if status is None:
            raise RuntimeError("远端未返回镜像状态")
        return status


def upload(args):
    if not is_wsl() or not is_remote_host(args.host):
        raise ValueError("images upload 必须从 WSL 指定远端 --host")
    archive = args.archive.expanduser().absolute()
    image = image_from_archive(archive.name)
    status = remote_load(args.host, image)
    if status != "missing":
        if status == "image":
            print(f"{image}: 镜像已存在，跳过上传和导入", flush=True)
        return 0
    if not archive.is_file():
        raise ValueError("镜像归档不存在：" + str(archive))
    if not shutil.which("rsync"):
        raise ValueError("请先安装 rsync")
    destination = REMOTE_DIR + "/containers/"
    # --protect-args sends ~ literally; anchor relative destinations at remote HOME.
    transfer_destination = (
        destination[2:] if destination.startswith("~/") else destination
    )
    remote = 'cd "$HOME" && mkdir -p -- ' + remote_path(destination) + " && exec rsync"
    result = subprocess.run(
        [
            "rsync",
            "-rt",
            "--partial",
            "--partial-dir=.rsync-partial",
            "--protect-args",
            "--blocking-io",
            "--info=progress2",
            "-e",
            shlex.quote(str(SSH_DIR / "ssh2")),
            "--rsync-path",
            checked_command(args.host, remote),
            "--",
            str(archive),
            f"{REMOTE_USER}@{args.host}:{transfer_destination}",
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode
    if remote_load(args.host, image) == "missing":
        raise RuntimeError("上传后未找到远端归档")
    print("镜像已就绪：" + image, flush=True)
    return 0


def list_images(args):
    if is_remote_host(args.host):
        return run_remote_cli(args.host, ["images", "ls"])
    if is_wsl():
        raise ValueError("WSL 请通过 --host 指定服务器")
    command = [
        "docker",
        "images",
        "--filter",
        "reference=vllm-ascend:*",
        "--filter",
        "reference=quay.io/ascend/vllm-ascend:*",
    ]
    print(shlex.join(command), flush=True)
    return subprocess.run(command, check=False).returncode
