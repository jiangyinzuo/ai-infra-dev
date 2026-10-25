#!/usr/bin/env python3
"""运行可信 Python 配置，采集选定函数的入口状态与调用栈。"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import CodeType, FrameType
from typing import Any, TextIO

# 目录入口由 Python 直接执行时，sys.path[0] 是工具目录；加入仓库根目录以加载完整包名。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trace_python.call_stack_analysis import Target, TraceConfig
from trace_python.call_stack_analysis.call_tree import CallTree


def snapshot(value: Any, depth: int = 0) -> Any:
    """不调用 repr 或搬运 tensor；容器有界，未知对象只保存类型。"""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if depth >= 3:
        return {"type": type(value).__qualname__, "truncated": True}
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [snapshot(x, depth + 1) for x in value[:16]],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "length": len(value),
            "items": {
                str(k): snapshot(v, depth + 1) for k, v in list(value.items())[:16]
            },
        }
    if all(hasattr(value, attr) for attr in ("shape", "dtype", "device")):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return {"type": type(value).__qualname__}


def capture_fields(frame: FrameType, fields: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for path in fields:
        parts = path.split(".")
        try:
            value = frame.f_locals[parts[0]]
            for part in parts[1:]:
                value = getattr(value, part)
            result[path] = snapshot(value)
        except (KeyError, AttributeError) as exc:
            result[path] = {"unavailable": type(exc).__name__}
    return result


def locate(
    root: Path, targets: tuple[Target, ...]
) -> dict[tuple[str, int, str], Target]:
    """编译但不执行源码；按实际 code object 的限定名验证目标。"""
    found = {}
    for target in targets:
        if target.limit < 0 or any(
            not all(p.isidentifier() for p in f.split(".")) for f in target.fields
        ):
            raise ValueError(f"无效 limit 或字段路径：{target}")
        path = (root / target.file).resolve()
        code = compile(path.read_bytes(), str(path), "exec")
        matches = []

        def walk(node: CodeType) -> None:
            if node.co_qualname == target.function:
                matches.append(node)
            for child in node.co_consts:
                if isinstance(child, CodeType):
                    walk(child)

        walk(code)
        if len(matches) != 1:
            raise ValueError(
                f"{target.file}:{target.function} 匹配到 {len(matches)} 个函数"
            )
        # 运行时以文件、首行及限定名匹配，避免 code equality 涉及整段字节码。
        key = (str(path), matches[0].co_firstlineno, target.function)
        if key in found:
            raise ValueError(f"重复采集目标：{target.file}:{target.function}")
        found[key] = target
    return found


class Writer:
    """输出trace结果"""

    def __init__(self, machine: TextIO, human: TextIO,
                 tree_json: TextIO, tree_text: TextIO):
        self.machine, self.human = machine, human
        self.tree_json, self.tree_text = tree_json, tree_text
        self.tree = CallTree()

    def emit(self, record: dict[str, Any]) -> None:
        if record["type"] == "event":
            self.tree.add(record)
        elif record["type"] == "summary":
            self.tree.write_json(self.tree_json, record)
            self.tree.write_text(self.tree_text, record)
        print(
            json.dumps(record, ensure_ascii=False, allow_nan=False),
            file=self.machine,
            flush=True,
        )
        kind = record["type"]
        if kind == "event":
            print(
                f"\n[{record['sequence']}] {record['target']}  hit={record['hit']}",
                file=self.human,
            )
            for name, value in record["values"].items():
                print(
                    f"  {name} = {json.dumps(value, ensure_ascii=False)}",
                    file=self.human,
                )
            print("  调用栈（外层 → 内层）：", file=self.human)
            for f in record["stack"]:
                print(f"    {f['file']}:{f['line']}  {f['function']}", file=self.human)
        elif kind == "summary":
            print(
                f"\n结束：{record['status']}；记录 {record['events']} 个事件",
                file=self.human,
            )
            for name, count in record["counts"].items():
                print(f"  {count:6d}  {name}", file=self.human)
            if record.get("error"):
                print(record["error"], file=self.human)
        else:
            print(f"{kind}: {json.dumps(record, ensure_ascii=False)}", file=self.human)
        self.human.flush()


class Tracer:
    def __init__(self, config: TraceConfig, writer: Writer):
        self.config, self.writer = config, writer
        self.targets = locate(config.root, config.targets)
        self.counts: Counter[str] = Counter()
        self.sequence = 0
        self.active = False

    def start(self) -> None:
        """配置入口在初始化结束、提交请求前显式开始采集。"""
        self.active = True

    def trace(self, frame: FrameType, event: str, arg: Any) -> None:
        # 不跟踪逐行事件；只观察当前线程函数调用，不接管子进程。
        if event != "call" or not self.active:
            return
        code = frame.f_code
        target = self.targets.get(
            (code.co_filename, code.co_firstlineno, code.co_qualname)
        )
        if target is None:
            return
        name = f"{target.file}:{target.function}"
        self.counts[name] += 1
        if self.counts[name] > target.limit:
            return
        stack = []
        parent = frame
        while parent:
            path = Path(parent.f_code.co_filename)
            filename = (
                str(path.relative_to(self.config.root))
                if path.is_relative_to(self.config.root)
                else str(path)
            )
            stack.append(
                {
                    "file": filename,
                    "line": parent.f_lineno,
                    "function": parent.f_code.co_qualname,
                }
            )
            parent = parent.f_back
        self.sequence += 1
        self.writer.emit(
            {
                "type": "event",
                "sequence": self.sequence,
                "target": name,
                "hit": self.counts[name],
                "values": capture_fields(frame, target.fields),
                "stack": stack[::-1],
            }
        )
        return


def load_config(path: Path) -> TraceConfig:
    spec = importlib.util.spec_from_file_location("trace_user_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    if not isinstance(config, TraceConfig):
        raise ValueError("配置必须导出 TraceConfig 类型的 config")
    if not config.root.is_absolute():
        raise ValueError("config.root 必须为绝对路径；可相对于配置的 __file__ 计算")
    return config


@contextlib.contextmanager
def target_output():
    """把被追踪程序的标准输出转到 stderr，避免它的日志混进追踪结果，尤其是 JSONL。"""
    # 同时隔离 Python print、原生扩展和继承 fd 1 的子进程输出。
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 1)
        os.close(saved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--config", required=True, type=Path, help="可信 Python 配置，import 会执行代码"
    )
    parser.add_argument("--output-prefix", required=True, type=Path,
                        help="统一输出前缀；覆盖写入 .jsonl、.txt、.tree.json、.tree.txt")
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    return args


@contextlib.contextmanager
def open_writer(args: argparse.Namespace) -> Iterator[Writer]:
    with contextlib.ExitStack() as stack:
        def open_output(suffix: str) -> TextIO:
            path = Path(str(args.output_prefix) + suffix)
            return stack.enter_context(path.open("w", encoding="utf-8"))

        yield Writer(open_output(".jsonl"), open_output(".txt"),
                     open_output(".tree.json"), open_output(".tree.txt"))


@contextlib.contextmanager
def execution_context(config: TraceConfig, tracer: Tracer) -> Iterator[None]:
    previous_trace = sys.gettrace()
    previous_cwd = Path.cwd()
    previous_env = {key: os.environ.get(key) for key in config.env}
    try:
        os.chdir(config.root)
        os.environ.update(config.env)
        with target_output():
            sys.settrace(tracer.trace)
            yield
    finally:
        sys.settrace(previous_trace)
        os.chdir(previous_cwd)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_trace(config: TraceConfig, writer: Writer, config_path: Path) -> int:
    tracer = Tracer(config, writer)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    writer.emit(
        {
            "type": "run",
            "started_at": started_at,
            "config": str(config_path),
            "root": str(config.root),
            "python": sys.executable,
            "scope": "current-thread Python calls; no CUDA kernel tracing",
        }
    )
    start = time.monotonic()
    status, error, exit_code = "ok", None, 0
    try:
        with execution_context(config, tracer):
            config.run(tracer.start)
    except BaseException as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        traceback.print_exc(file=sys.stderr)
    finally:
        writer.emit(
            {
                "type": "summary",
                "started_at": started_at,
                "status": status,
                "error": error,
                "events": tracer.sequence,
                "counts": dict(tracer.counts),
                "elapsed_seconds": time.monotonic() - start,
            }
        )
    return exit_code


def main() -> int:
    args = parse_args()
    with target_output():
        trace_config: TraceConfig = load_config(args.config)
    with open_writer(args) as writer:
        return run_trace(trace_config, writer, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
