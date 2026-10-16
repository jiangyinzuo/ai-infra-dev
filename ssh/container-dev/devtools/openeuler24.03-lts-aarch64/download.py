#!/usr/bin/env python3
"""ARM64 tool downloads: Python 3.9+, curl transport, optional Rich progress."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time


class Cancelled(Exception):
    pass


class Display:
    def __init__(self, names):
        self.progress = None
        self.lock = threading.Lock()
        self.states = {}
        if sys.stderr.isatty():
            try:
                from rich.console import Console
                from rich.table import Column
                from rich.progress import (
                    Progress,
                    TextColumn,
                    BarColumn,
                    DownloadColumn,
                    TransferSpeedColumn,
                    TimeRemainingColumn,
                )
            except ImportError:
                print("Rich not installed; using plain logs.", file=sys.stderr)
            else:
                self.progress = Progress(
                    TextColumn("{task.description}"),
                    TextColumn(
                        "{task.fields[state]}", table_column=Column(max_width=38)
                    ),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=Console(stderr=True),
                )
        self.tasks = {}
        if self.progress:
            for name in names:
                self.tasks[name] = self.progress.add_task(
                    name, total=None, state="waiting"
                )

    def update(self, name, state, completed=None, total=None, reset=False):
        with self.lock:
            if self.progress:
                task = self.tasks[name]
                if reset:
                    self.progress.reset(task, total=total, start=True)
                fields = {"state": state}
                if completed is not None:
                    fields["completed"] = completed
                self.progress.update(task, **fields)
            elif self.states.get(name) != state:
                print("[{}] {}".format(name, state), file=sys.stderr, flush=True)
            self.states[name] = state


class Downloader:
    def __init__(self, directory, display):
        self.directory = Path(directory)
        self.display = display
        self.cancel = threading.Event()

    def check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled()

    def transfer(self, name, url, output, size=None, metadata=False):
        """Keep curl's proxy/TLS behavior; own retries so the UI shows each attempt."""
        for attempt in range(3):
            self.check_cancel()
            state = ("query " if metadata else "download ") + url
            if attempt:
                state = "retry {}/2: {}".format(attempt, state)
            self.display.update(name, state, completed=0, total=size, reset=True)
            command = [
                "curl",
                "-fsSL",
                "--connect-timeout",
                "10",
                "--max-time",
                "60" if metadata else "1800",
            ]
            if not metadata:
                command += ["--speed-limit", "1024", "--speed-time", "60"]
            command += ["--output", str(output), url]
            # File-backed stderr avoids pipe deadlocks and preserves HTTP errors.
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stderr=errors)
                try:
                    while process.poll() is None:
                        self.check_cancel()
                        if not metadata:
                            self.display.update(
                                name, state, completed=output.stat().st_size
                            )
                        self.cancel.wait(0.1)
                    self.check_cancel()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if process.returncode == 0:
                    return
                errors.seek(0)
                message = errors.read().decode("utf-8", errors="replace").strip()
            # Retry transient transport errors, not HTTP 4xx or invalid content.
            if attempt == 2 or process.returncode not in (5, 6, 7, 18, 28, 52, 55, 56):
                raise RuntimeError("{}: {}".format(url, message or process.returncode))
            self.display.update(name, "retry pending: " + message)
            if self.cancel.wait(2):
                raise Cancelled()

    def metadata(self, name, url, as_json=True):
        with tempfile.TemporaryDirectory(prefix="tool-metadata-") as directory:
            path = Path(directory) / "response"
            self.transfer(name, url, path, metadata=True)
            text = path.read_text()
        return json.loads(text) if as_json else text

    def digest(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                self.check_cancel()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    return digest.hexdigest()
                digest.update(chunk)

    def download(self, name, filename, url, expected, size=None):
        if not re.fullmatch("[0-9a-fA-F]{64}", expected):
            raise ValueError("missing or invalid SHA-256: " + filename)
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self.display.update(name, "checking " + filename)
        if path.is_file() and self.digest(path) == expected.lower():
            self.display.update(
                name,
                "skip (sha256 matches)",
                completed=path.stat().st_size,
                total=path.stat().st_size,
                reset=True,
            )
            return
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            self.transfer(name, url, temporary, size)
            self.display.update(
                name, "verifying " + filename, completed=temporary.stat().st_size
            )
            if self.digest(temporary) != expected.lower():
                raise ValueError("SHA-256 mismatch: " + filename)
            self.check_cancel()
            temporary.replace(path)
            self.display.update(
                name, "saved (sha256 verified)", completed=path.stat().st_size
            )
        finally:
            temporary.unlink(missing_ok=True)

    def release(self, name, repo, tag="latest"):
        suffix = "latest" if tag == "latest" else "tags/" + tag
        return self.metadata(
            name, "https://api.github.com/repos/{}/releases/{}".format(repo, suffix)
        )

    def asset(self, name, release, pattern, filename=None):
        matches = [
            asset for asset in release["assets"] if re.fullmatch(pattern, asset["name"])
        ]
        if len(matches) != 1:
            raise ValueError("expected exactly one matching asset: " + pattern)
        asset = matches[0]
        digest = asset.get("digest") or ""
        if not digest.startswith("sha256:"):
            raise ValueError("asset lacks SHA-256: " + asset["name"])
        self.download(
            name,
            filename or asset["name"],
            asset["browser_download_url"],
            digest[7:],
            asset.get("size"),
        )

    def github(self, name, repo, pattern, tag="latest", filename=None):
        self.asset(name, self.release(name, repo, tag), pattern, filename)
        if name == "omp":
            (self.directory / "omp").chmod(0o755)

    def node(self):
        filename = "node-v24.21.0-linux-arm64.tar.xz"
        url = "https://nodejs.org/dist/v24.21.0/"
        text = self.metadata("node", url + "SHASUMS256.txt", as_json=False)
        matches = [
            line.split()[0]
            for line in text.splitlines()
            if len(line.split()) == 2 and line.split()[1] == filename
        ]
        if len(matches) != 1:
            raise ValueError("cannot obtain Node.js checksum")
        self.download("node", filename, url + filename, matches[0])

    def fzf(self):
        release = self.release("fzf", "junegunn/fzf")
        tag = release["tag_name"]
        self.asset(
            "fzf",
            release,
            re.escape("fzf-{}-linux_arm64.tar.gz".format(tag.lstrip("v"))),
        )
        for script in ("fzf-preview.sh", "fzf-tmux"):
            info = self.metadata(
                "fzf",
                "https://api.github.com/repos/junegunn/fzf/contents/bin/{}?ref={}".format(
                    script, tag
                ),
            )
            expected = hashlib.sha256(base64.b64decode(info["content"])).hexdigest()
            self.download(
                "fzf",
                "fzf-{}-support/{}".format(tag.lstrip("v"), script),
                "https://raw.githubusercontent.com/junegunn/fzf/{}/bin/{}".format(
                    tag, script
                ),
                expected,
            )

    def tasks(self):
        specs = [
            ("neovim", "neovim/neovim", r"nvim-linux-arm64\.tar\.gz", "nightly", None),
            ("omp", "can1357/oh-my-pi", "omp-linux-arm64", "v18.10.3", "omp"),
            (
                "codex",
                "openai/codex",
                r"codex-package-aarch64-unknown-linux-musl\.tar\.gz",
                "latest",
                None,
            ),
            (
                "ripgrep",
                "BurntSushi/ripgrep",
                r"ripgrep-[0-9][^-]*-aarch64-unknown-linux-musl\.tar\.gz",
                "latest",
                None,
            ),
            (
                "fd",
                "sharkdp/fd",
                r"fd-v?[0-9][^-]*-aarch64-unknown-linux-musl\.tar\.gz",
                "latest",
                None,
            ),
            (
                "delta",
                "dandavison/delta",
                r"delta-[0-9][^-]*-aarch64-unknown-linux-gnu\.tar\.gz",
                "latest",
                None,
            ),
            (
                "tree-sitter",
                "tree-sitter/tree-sitter",
                r"tree-sitter-linux-arm64\.gz",
                "latest",
                None,
            ),
            (
                "tmux",
                "tmux/tmux-builds",
                r"tmux-[0-9][^-]*-linux-arm64\.tar\.gz",
                "latest",
                None,
            ),
        ]
        tasks = {spec[0]: (lambda spec=spec: self.github(*spec)) for spec in specs}
        tasks["node"] = self.node
        tasks["fzf"] = self.fzf
        return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs requires a positive integer")
    if not shutil.which("curl"):
        parser.error("curl is required")
    downloader = Downloader(Path(__file__).resolve().parent, None)
    tasks = downloader.tasks()
    display = Display(tasks)
    downloader.display = display
    old_handlers = {}

    def interrupt(signum, frame):
        downloader.cancel.set()
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, interrupt)
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    failed = False
    errors = []
    interrupted = False
    futures = {}
    if display.progress:
        display.progress.start()
    try:
        futures = {executor.submit(task): name for name, task in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed = True
                errors.append("[{}] {}".format(name, exc))
                display.update(name, "FAILED: " + str(exc))
    except KeyboardInterrupt:
        interrupted = True
        downloader.cancel.set()
        # Further interrupts must not interrupt child-process cleanup.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        for future in futures:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if display.progress:
            display.progress.stop()
            for message in errors:
                print(message, file=sys.stderr)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 130 if interrupted else int(failed)


if __name__ == "__main__":
    sys.exit(main())
