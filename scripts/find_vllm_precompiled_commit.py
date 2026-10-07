#!/usr/bin/env python3
"""Find a published vLLM CUDA wheel commit without downloading the wheel.

Examples:
    python3 scripts/find_vllm_precompiled_commit.py
    python3 scripts/find_vllm_precompiled_commit.py --variant cu129 --format command

Only stdout contains the requested output; diagnostics go to stderr.
Availability does not guarantee ABI compatibility with your source checkout.
"""

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from urllib.error import HTTPError, URLError
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        help="例如 cu129、cu130；default 表示旧版无 variant 的根索引。默认自动检测",
    )
    parser.add_argument("--arch", default=platform.machine(), help="默认当前机器架构")
    parser.add_argument("--timeout", type=float, default=20, help="每个 HTTP 请求超时秒数")
    parser.add_argument("--format", choices=("commit", "env", "command"), default="commit")
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("--timeout 必须大于 0")
        variant = args.variant or detect_variant()
        if not re.fullmatch(r"cu[0-9]+|default", variant):
            raise ValueError("--variant 必须是 cu 加版本号（如 cu129），或 default")
        commit, wheel_url = discover(variant, args.arch, args.timeout)
        print(f"已验证 {variant}/{args.arch}：{wheel_url}", file=sys.stderr)
        assignments = {
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_COMMIT": commit,
        }
        if variant != "default":
            assignments["VLLM_PRECOMPILED_WHEEL_VARIANT"] = variant
        if args.format == "commit":
            print(commit)
        elif args.format == "env":
            for key, value in assignments.items():
                print(f"export {key}={shlex.quote(value)}")
        else:
            env = " ".join(f"{key}={shlex.quote(value)}" for key, value in assignments.items())
            print(f"{env} uv pip install -e .")
    except (OSError, URLError, ValueError) as exc:
        print(f"发现失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
