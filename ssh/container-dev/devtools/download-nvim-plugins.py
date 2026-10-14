#!/usr/bin/env python3
"""Download Neovim plugin sources without Git (Python 3 standard library only).

Usage: python3 download-nvim-plugins.py [--lock PATH] [--output DIR] [--jobs N]
Default output: nvim-plugins/ beside this script (shared across target platforms).
Repositories are resolved from quoted owner/repo declarations in the config's
Lua files. Use --repos JSON for explicit {"plugin-name": "owner/repo"} overrides,
e.g. aliases or ambiguous declarations. Missing/ambiguous repositories fail
before downloading; we never guess a GitHub owner from a lock-file key.

Archives stay compressed; no plugin build hooks or Git submodules are run.
Archives use stable plugin-name.tar.gz filenames, including across repository forks.
archive-hashes.json records each URL and SHA-256; cache hits require both to match.
GitHub provides no official source-archive digest: hashes detect local changes,
not authenticity. Existing archives without metadata are downloaded again.
Downloads run concurrently (default: 8 workers). Completed archive metadata is
saved even on partial failure so successful downloads can be reused.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import urllib.request


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
        choices = {overrides[name]} if name in overrides else candidates.get(name, set())
        if len(choices) != 1:
            raise ValueError(f"{name}: repository missing or ambiguous ({sorted(choices)}); use --repos")
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
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.", delete=False) as stream:
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
    if (cached.get("url") == url and destination.is_file()
            and sha256(destination) == cached.get("sha256")):
        print(f"skip {destination.name} (URL and SHA-256 match)", flush=True)
        return cached
    print(f"download {destination.name}", flush=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".tmp.", delete=False) as stream:
            temp = Path(stream.name)
            request = urllib.request.Request(url, headers={"User-Agent": "nvim-plugin-source-downloader"})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=Path.home() / ".config/nvim/lazy-lock.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "nvim-plugins")
    parser.add_argument("--repos", type=Path, help="JSON repository overrides keyed by lock plugin name")
    parser.add_argument("--jobs", type=int, default=8, help="concurrent downloads (default: 8)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    lock = json.loads(args.lock.read_bytes())
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
        downloads["lazy.nvim.tar.gz"] = "https://github.com/folke/lazy.nvim/archive/refs/tags/v11.17.5.tar.gz"
    for name, entry in lock.items():
        repo = repositories[name]
        commit = entry["commit"]
        downloads[f"{name}.tar.gz"] = f"https://github.com/{repo}/archive/{commit}.tar.gz"

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(download, url, args.output / name, hashes.get(name, {})): name
            for name, url in downloads.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                hashes[name] = future.result()
            except Exception as error:
                failures.append(f"{name}: {error}")
    atomic_write(manifest, (json.dumps(hashes, indent=2, sort_keys=True) + "\n").encode())
    if failures:
        raise OSError("download failed:\n" + "\n".join(failures))



if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError) as error:
        sys.exit(f"error: {error}")
