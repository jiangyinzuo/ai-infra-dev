#!/usr/bin/env python3
"""下载 Neovim 插件源码归档，无需 Git，默认并发 8，保存到 devtools/nvim-plugins/。

用法：cli.py binaries download [--lock PATH] [--output DIR] [--jobs N]
仓库从 Lua 配置解析；有歧义时用 --repos JSON 指定，不自动猜测。
按 URL 和 SHA-256 复用缓存，失败时保留已完成下载；哈希不用于来源认证。
下载仅获取源码；提取使用系统 tar，并按归档哈希和已安装文件状态增量跳过。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import urllib.request

from ssh_common.config import NVIM_PLUGIN_EXCLUDES
from .paths import DATA_ROOT


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


def installed_state(destination, names):
    """Stat only archive-owned entries; extra build outputs never invalidate a hit."""
    result = {}
    for name in names:
        path = destination / name
        info = path.lstat()
        metadata = [info.st_mode]
        if stat.S_ISLNK(info.st_mode):
            metadata.append(str(path.readlink()))
        elif not stat.S_ISDIR(info.st_mode):
            metadata.extend(
                [
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_ino,
                    info.st_dev,
                ]
            )
        result[name] = metadata
    return result


def archive_members(archive):
    """Record stripped member paths, including parent directories, only on a miss."""
    names = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            parts = member.name.split("/")
            if member.name.startswith("/") or ".." in parts:
                raise ValueError(f"{archive}: unsafe archive member {member.name!r}")
            parts = [part for part in parts[1:] if part and part != "."]
            for count in range(1, len(parts) + 1):
                names.add("/".join(parts[:count]))
    return sorted(names)


def extract_plugin(archive, destination):
    # Keep stamps beside the host's installation, never among synced archives.
    stamp = destination.with_name("." + destination.name + ".extract-stamp.json")
    digest = sha256(archive)
    try:
        cached = json.loads(stamp.read_bytes())
        if (
            cached.get("sha256") == digest
            and isinstance(cached.get("files"), dict)
            and cached["files"] == installed_state(destination, cached["files"])
            and destination.is_dir()
        ):
            print(
                f"skip {archive.name} (archive and installed state match)", flush=True
            )
            return
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    # A failed or interrupted extraction must not leave a usable success stamp.
    stamp.unlink(missing_ok=True)
    names = archive_members(archive)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"extract {archive} -> {destination}", flush=True)
    subprocess.run(
        ["tar", "-xf", str(archive), "--strip-components=1", "-C", str(destination)],
        check=True,
    )
    state = installed_state(destination, names)
    atomic_write(
        stamp,
        (
            json.dumps({"sha256": digest, "files": state}, sort_keys=True) + "\n"
        ).encode(),
    )


def extract_plugins(args):
    archives = sorted((args.output or DATA_ROOT / "nvim-plugins").glob("*.tar.gz"))
    if not archives:
        print("No plugin archives found.")
        return
    home = Path.home()
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for archive in archives:
            name = archive.name[: -len(".tar.gz")]
            destination = (
                home / ".local/share/nvim/lazy/lazy.nvim"
                if name == "lazy.nvim"
                else home / "plugged" / name
            )
            futures[executor.submit(extract_plugin, archive, destination)] = (
                archive.name
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        raise OSError("plugin extraction failed:\n" + "\n".join(failures))
