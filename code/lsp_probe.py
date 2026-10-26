#!/usr/bin/env python
"""Probe Python language servers without editor mappings or result filtering.

Run with the vLLM environment (no extra dependencies):
    cd ~/ai-infra-dev/code/vllm-workspace/vllm
    uv run --no-sync --offline --python .venv/bin/python python ../../lsp_probe.py \
        --file vllm/v1/engine/output_processor.py --line 551 \
        --symbol OutputProcessor --open vllm/v1/engine/async_llm.py

Positions are 1-based. Relative file paths are relative to the working directory.
The default root is the vllm-workspace/vllm checkout beside this script.
Use --root to compare workspace roots; Pyrefly selects config per source file.
The script never edits source/configuration. JSON goes to stdout; server stderr
is retained at the printed log path. An empty references list is a valid result.
"""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time


class LspClient:
    def __init__(self, command, cwd, log):
        self.process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log,
        )
        self.messages = queue.Queue()
        self.next_id = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        raise EOFError("Language server closed stdout")
                    if line in (b"\r\n", b"\n"):
                        break
                    key, value = line.decode("ascii").split(":", 1)
                    headers[key.lower()] = value.strip()
                size = int(headers["content-length"])
                body = bytearray()
                while len(body) < size:
                    chunk = self.process.stdout.read(size - len(body))
                    if not chunk:
                        raise EOFError("Truncated LSP response")
                    body.extend(chunk)
                self.messages.put(json.loads(body))
        except Exception as error:
            self.messages.put(error)

    def send(self, **message):
        body = json.dumps({"jsonrpc": "2.0", **message}).encode()
        self.process.stdin.write(
            f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        self.process.stdin.flush()

    def handle_server_message(self, message):
        if isinstance(message, Exception):
            raise message
        if "method" not in message or "id" not in message:
            return
        if message["method"] == "workspace/configuration":
            result = [{} for _ in message.get("params", {}).get("items", [])]
            self.send(id=message["id"], result=result)
        elif message["method"] in (
            "client/registerCapability", "client/unregisterCapability",
            "window/workDoneProgress/create",
        ):
            self.send(id=message["id"], result=None)
        else:
            self.send(id=message["id"], error={
                "code": -32601, "message": "Unsupported probe client method",
            })

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                message = self.messages.get(timeout=remaining)
            except queue.Empty:
                return
            self.handle_server_message(message)

    def request(self, method, params, timeout):
        self.next_id += 1
        request_id = self.next_id
        self.send(id=request_id, method=method, params=params)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {method}")
            try:
                message = self.messages.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"Timed out waiting for {method}") from None
            self.handle_server_message(message)
            if "method" not in message and message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"]))
                return message["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "vllm-workspace" / "vllm")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    position = parser.add_mutually_exclusive_group(required=True)
    position.add_argument("--symbol", help="Unique substring on the selected line")
    position.add_argument("--column", type=int, help="1-based Unicode character column")
    parser.add_argument("--action", choices=["references", "definition", "typeDefinition", "hover"], default="references")
    parser.add_argument("--open", type=Path, action="append", default=[], help="Additional file to open; repeatable")
    parser.add_argument("--wait", type=float, default=5, help="Seconds before querying (default: 5); not an index-completion guarantee")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--exclude-declaration", action="store_true")
    parser.add_argument("--server", default="pyrefly", help="Language server executable")
    parser.add_argument("--server-subcommand", choices=["lsp", "server"], default="lsp",
                        help="Server entrypoint: lsp for Pyrefly, server for ty")
    parser.add_argument("--server-arg", action="append", default=[], help="Extra server argument; use --server-arg=--flag")
    parser.add_argument("--log", type=Path, help="Server stderr path (default: unique /tmp file)")
    args = parser.parse_args()
    if args.wait < 0 or args.timeout <= 0:
        parser.error("--wait must be nonnegative and --timeout must be positive")
    return parser, args


def prepare_query(parser, args):
    root, file = args.root.resolve(), args.file.resolve()
    text = file.read_text()
    lines = text.splitlines()
    if not 1 <= args.line <= len(lines):
        parser.error("--line is outside the file")
    line = lines[args.line - 1]
    if args.symbol is not None:
        if not args.symbol or line.count(args.symbol) != 1:
            parser.error("--symbol must occur exactly once on the selected line")
        column = line.index(args.symbol)
    else:
        column = args.column - 1
        if not 0 <= column < len(line):
            parser.error("--column is outside the selected line")
    params = {"textDocument": {"uri": file.as_uri()}, "position": {
        "line": args.line - 1,
        "character": len(line[:column].encode("utf-16-le")) // 2,
    }}
    if args.action == "references":
        params["context"] = {"includeDeclaration": not args.exclude_declaration}
    files = {file: text}
    for extra in args.open:
        extra = extra.resolve()
        files[extra] = extra.read_text()
    return root, files, params


def open_log(path):
    if path:
        return path.open("w")
    return tempfile.NamedTemporaryFile(
        mode="w", prefix="lsp-probe-", suffix=".log", delete=False,
    )


def initialize_client(client, root, timeout):
    initialized = client.request("initialize", {
        "processId": os.getpid(), "rootUri": root.as_uri(),
        "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
        "capabilities": {"general": {"positionEncodings": ["utf-16"]},
                         "workspace": {"configuration": True}},
    }, timeout)
    encoding = initialized.get("capabilities", {}).get("positionEncoding", "utf-16")
    if encoding != "utf-16":
        raise RuntimeError(f"Unsupported negotiated position encoding: {encoding}")
    client.send(method="initialized", params={})
    return initialized


def open_documents(client, files):
    for path, content in files.items():
        client.send(method="textDocument/didOpen", params={"textDocument": {
            "uri": path.as_uri(), "languageId": "python", "version": 1,
            "text": content,
        }})


def run_query(client, args, root, files, params):
    initialized = initialize_client(client, root, args.timeout)
    open_documents(client, files)
    client.wait(args.wait)
    result = client.request("textDocument/" + args.action, params, args.timeout)
    result_count = len(result) if isinstance(result, list) else int(result is not None)
    return {"serverInfo": initialized.get("serverInfo"),
            "action": args.action, "params": params,
            "result_count": result_count, "result": result}


def main():
    parser, args = parse_arguments()
    root, files, params = prepare_query(parser, args)
    log = open_log(args.log)
    command = [args.server, args.server_subcommand, *args.server_arg]
    print(f"Python: {sys.executable}\nServer: {command}\nRoot: {root}\nLog: {log.name}", file=sys.stderr)
    client = None
    try:
        client = LspClient(command, root, log)
        response = run_query(client, args, root, files, params)
        print(json.dumps(response, indent=2, ensure_ascii=False))
    finally:
        if client is not None:
            client.close()
        log.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TimeoutError, EOFError) as error:
        print(f"LSP probe failed: {error}", file=sys.stderr)
        sys.exit(1)
