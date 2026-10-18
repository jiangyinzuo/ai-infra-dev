#!/usr/bin/env python3
"""下载 NPU 容器镜像；WSL 必须指定 --host。"""

import os
from pathlib import Path
import subprocess
import tempfile

from ..core.remote import ROOT, NPU_PLATFORMS, dispatch
from remote_exec import is_wsl


def download(args):
    tag = f"dev-{args.version}.day{args.date}-{NPU_PLATFORMS[args.npu_platform]}"
    image = f"vllm-ascend:{tag}"
    directory = ROOT
    directory.mkdir(exist_ok=True)
    archive = directory / f"vllm-ascend_{tag}.tar.gz"
    if archive.is_file():
        print(f"skip {archive} (archive exists)")
        return 0
    if not is_wsl():
        result = subprocess.run(
            ["docker", "images", "vllm-ascend", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        if image in result.stdout.splitlines():
            print(f"skip {image} (Docker image exists)")
            return 0
    url = f"https://cmc-nkg-artifactory.cmc.tools.huawei.com/artifactory/cmc-nkg-inner/MindIE/vLLM%20Ascend/{args.version}.daily/{args.date[:6]}/{args.date}/vLLM-Ascend-images/{image}.tar.gz"
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
    remote = dispatch(
        args, "download", ["--date", args.date, "--version", args.version]
    )
    return download(args) if remote is None else remote
