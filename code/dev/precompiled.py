"""Find a published CUDA wheel; availability does not guarantee checkout ABI compatibility."""

import json
import os
import platform
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

BASE_URL = "https://wheels.vllm.ai/"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def detect_variant():
    """Match CUDA selection in the current vLLM setup.py (CUDA 12/13)."""
    override = os.environ.get("VLLM_PRECOMPILED_WHEEL_VARIANT")
    if override:
        return override
    version = os.environ.get("VLLM_MAIN_CUDA_VERSION")
    if version:
        return "cu" + version.replace(".", "")[:3]
    try:
        import torch

        if torch.version.hip:
            raise ValueError("ROCm 环境请使用 ROCm 专用安装流程；此脚本只发现 CUDA wheel")
        version = torch.version.cuda
    except ImportError:
        version = None
    if not version:
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                version = match.group(1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not version:
        version = "13.0"
        print("未检测到 CUDA，按当前 vLLM 默认值使用 cu130；可用 --variant 显式指定", file=sys.stderr)
    return {12: "cu129", 13: "cu130"}.get(int(version.split(".")[0]), "cu130")


def fetch_metadata(url, timeout):
    with urlopen(url, timeout=timeout) as response:
        metadata = json.load(response)
    if not isinstance(metadata, list) or not all(isinstance(w, dict) for w in metadata):
        raise ValueError(f"无效 wheel 元数据：{url}")
    return metadata


def wheel_candidates(metadata, arch):
    for wheel in metadata:
        if wheel.get("package_name") != "vllm":
            continue
        tag = wheel.get("platform_tag", "")
        if isinstance(tag, str) and tag.endswith("_" + arch):
            yield wheel


def published_commit(wheel_url):
    url = urlsplit(wheel_url)
    if url.scheme != "https" or url.netloc != "wheels.vllm.ai":
        raise ValueError(f"非官方 wheel URL：{wheel_url}")
    for part in url.path.split("/"):
        if COMMIT_RE.fullmatch(part):
            return part
    raise ValueError(f"wheel 路径中没有完整 commit：{wheel_url}")


def check_download(url, timeout):
    # HEAD verifies the artifact without downloading a multi-GB wheel.
    with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"wheel 不可下载：HTTP {response.status} {url}")


def discover(variant, arch, timeout):
    variant_dir = "" if variant == "default" else variant + "/"
    nightly_repo = f"{BASE_URL}nightly/{variant_dir}vllm/"
    nightly_metadata = fetch_metadata(nightly_repo + "metadata.json", timeout)
    candidates = list(wheel_candidates(nightly_metadata, arch))
    if not candidates:
        raise ValueError(f"nightly 没有 {variant}/{arch} 的 vllm wheel")
    for wheel in candidates:
        path = wheel.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("nightly wheel 元数据缺少 path")
        commit = published_commit(urljoin(nightly_repo, path))
        # Verify the exact index that VLLM_PRECOMPILED_WHEEL_COMMIT will use.
        repo = f"{BASE_URL}{commit}/{variant_dir}vllm/"
        try:
            metadata = fetch_metadata(repo + "metadata.json", timeout)
            for pinned in wheel_candidates(metadata, arch):
                pinned_path = pinned.get("path")
                if not isinstance(pinned_path, str) or not pinned_path:
                    raise ValueError("commit wheel 元数据缺少 path")
                wheel_url = urljoin(repo, pinned_path)
                if published_commit(wheel_url) != commit:
                    raise ValueError("commit 索引指向了其他提交的 wheel")
                check_download(wheel_url, timeout)
                return commit, wheel_url
        except HTTPError as exc:
            if exc.code != 404:
                raise
            print(f"跳过尚未完整发布的 commit {commit}：{exc}", file=sys.stderr)
    raise ValueError(f"nightly 中没有通过 commit 索引及下载检查的 {variant}/{arch} wheel")


def find_vllm_precompiled_commit() -> dict[str, str]:
    """Return the verified wheel environment; fail before installation on discovery errors."""
    variant = detect_variant()
    if not re.fullmatch(r"cu[0-9]+|default", variant):
        raise ValueError("VLLM_PRECOMPILED_WHEEL_VARIANT 必须为 cu 加版本号或 default")
    arch = platform.machine()
    commit, wheel_url = discover(variant, arch, 20)
    print(f"已验证 {variant}/{arch}：{wheel_url}", file=sys.stderr)
    environment = {
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_COMMIT": commit,
    }
    if variant != "default":
        environment["VLLM_PRECOMPILED_WHEEL_VARIANT"] = variant
    return environment
