#!/usr/bin/env python3
"""下载 Neovim 插件源码归档，无需 Git，默认并发 8，保存到 devtools/nvim-plugins/。

用法：cli.py nvim-plugins download [--lock PATH] [--output DIR] [--jobs N]
仓库从 Lua 配置解析；有歧义时用 --repos JSON 指定，不自动猜测。
按 URL 和 SHA-256 复用缓存，失败时保留已完成下载；哈希不用于来源认证。
仅下载源码，不执行插件构建或拉取 Git 子模块。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import urllib.request

from arguments import add_jobs_argument
from config import NVIM_PLUGIN_EXCLUDES
from remote_exec import add_host_argument, is_remote_host, is_wsl, run_remote_cli


REPO = re.compile(r"[\w.-]+/[\w.-]+", re.ASCII)
DECLARATION = re.compile(r"[\"']([\w.-]+/[\w.-]+)[\"']", re.ASCII)
COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def resolve_repositories(config, names, overrides):
    candidates = {}
    for path in config.rglob("*.lua"):
        for repo in DECLARATION.findall(path.read_text()):
            candidates.setdefault(repo.rsplit("/", 1)[1], set()).add(repo)
    result = {}
    for name in names:
        choices = (
            {overrides[name]} if name in overrides else candidates.get(name, set())
        )
        if len(choices) != 1:
            raise ValueError(
                f"{name}: repository missing or ambiguous ({sorted(choices)}); use --repos"
            )
        repo = next(iter(choices))
        if not isinstance(repo, str) or not REPO.fullmatch(repo):
            raise ValueError(f"{name}: invalid repository {repo!r}")
        result[name] = repo
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, content):
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".tmp.", delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(content)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def check_archive(path):
    # Read every regular file to catch truncated payloads, not just bad headers.
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as stream:
                    while stream.read(1024 * 1024):
                        pass


def download(url, destination, cached):
    if (
        cached.get("url") == url
        and destination.is_file()
        and sha256(destination) == cached.get("sha256")
    ):
        print(f"skip {destination.name} (URL and SHA-256 match)", flush=True)
        return cached
    print(f"download {destination.name}", flush=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".tmp.", delete=False
        ) as stream:
            temp = Path(stream.name)
            request = urllib.request.Request(
                url, headers={"User-Agent": "nvim-plugin-source-downloader"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        check_archive(temp)
        digest = sha256(temp)
        temp.replace(destination)
        return {"url": url, "sha256": digest}
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def register(subparsers):
    parser = subparsers.add_parser(
        "nvim-plugins", help="download or extract Neovim plugins"
    )
    actions = parser.add_subparsers(dest="plugin_action", required=True)
    download_parser = actions.add_parser(
        "download",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    download_parser.add_argument("--lock", type=Path)
    download_parser.add_argument("--output", type=Path)
    download_parser.add_argument(
        "--repos", type=Path, help="JSON repository overrides keyed by lock plugin name"
    )
    extract_parser = actions.add_parser(
        "extract", help="extract synced plugin archives"
    )
    for action in (download_parser, extract_parser):
        add_host_argument(action)
        add_jobs_argument(action)
        action.set_defaults(handler=execute)


def execute(args):
    if args.plugin_action == "extract" and is_wsl() and not is_remote_host(args.host):
        raise ValueError("nvim-plugins extract on WSL requires a remote --host")
    if is_remote_host(args.host):
        remote_args = [
            "nvim-plugins",
            args.plugin_action,
            "--jobs",
            str(args.jobs),
        ]
        if args.plugin_action == "download":
            for option in ("lock", "output", "repos"):
                value = getattr(args, option)
                if value is not None:
                    remote_args.extend(["--" + option, str(value)])
        return run_remote_cli(args.host, remote_args)
    if args.plugin_action == "extract":
        return subprocess.run(
            [
                "bash",
                str(Path(__file__).with_name("extract_nvim_plugins.sh")),
                "--jobs",
                str(args.jobs),
            ],
            check=False,
        ).returncode
    args.lock = args.lock or Path.home() / ".config/nvim/lazy-lock.json"
    args.output = args.output or Path(__file__).resolve().parents[1] / "nvim-plugins"
    download_plugins(args)
    return 0


def download_plugins(args):
    lock = {
        name: entry
        for name, entry in json.loads(args.lock.read_bytes()).items()
        if name not in NVIM_PLUGIN_EXCLUDES
    }
    for name, entry in lock.items():
        if not isinstance(entry, dict) or not COMMIT.fullmatch(entry.get("commit", "")):
            raise ValueError(f"{name}: expected a full 40-character commit")
    overrides = json.loads(args.repos.read_text()) if args.repos else {}
    repositories = resolve_repositories(args.lock.parent, lock, overrides)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "archive-hashes.json"
    hashes = json.loads(manifest.read_bytes()) if manifest.is_file() else {}
    downloads = {}

    # Bootstrap lazy.nvim separately because it is not present in this lock file.
    if "lazy.nvim" not in lock:
        downloads["lazy.nvim.tar.gz"] = (
            "https://github.com/folke/lazy.nvim/archive/refs/tags/v11.17.5.tar.gz"
        )
    for name, entry in lock.items():
        repo = repositories[name]
        commit = entry["commit"]
        downloads[f"{name}.tar.gz"] = (
            f"https://github.com/{repo}/archive/{commit}.tar.gz"
        )

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                download, url, args.output / name, hashes.get(name, {})
            ): name
            for name, url in downloads.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                hashes[name] = future.result()
            except Exception as error:
                failures.append(f"{name}: {error}")
    atomic_write(
        manifest, (json.dumps(hashes, indent=2, sort_keys=True) + "\n").encode()
    )
    if failures:
        raise OSError("download failed:\n" + "\n".join(failures))
