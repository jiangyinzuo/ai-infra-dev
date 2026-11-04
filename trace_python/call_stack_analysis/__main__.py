#!/usr/bin/env python3
"""运行可信 Python 配置，采集选定函数的入口状态与调用栈。"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path

# 目录入口由 Python 直接执行时，sys.path[0] 是工具目录；加入仓库根目录以加载完整包名。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trace_python.call_stack_analysis import TraceConfig, TraceSession
from trace_python.call_stack_analysis.configuration import load_config


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
def execution_context(config: TraceConfig) -> Iterator[None]:
    previous_cwd = Path.cwd()
    previous_env = {key: os.environ.get(key) for key in config.env}
    try:
        os.chdir(config.root)
        os.environ.update(config.env)
        with target_output():
            yield
    finally:
        os.chdir(previous_cwd)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_trace(config: TraceConfig, output_prefix: Path, config_path: Path) -> int:
    session = TraceSession(config.root, config.targets, output_prefix,
                           config_path=config_path)
    try:
        with session.capture(active=False, output_prefix=output_prefix) as tracer:
            with execution_context(config):
                config.run(tracer.start)
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


def main() -> int:
    args = parse_args()
    with target_output():
        trace_config: TraceConfig = load_config(args.config)
    return run_trace(trace_config, args.output_prefix, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
