"""Shared process-wide startup tracing for vLLM CLI configurations."""

from __future__ import annotations

import atexit
import os
import signal
from pathlib import Path

from trace_python.call_stack_analysis import TraceSession
from trace_python.call_stack_analysis.configuration import load_config


def _exit_on_sigterm(signum, _frame) -> None:
    # 只展开 Python 栈；不能在信号处理器中加 tracer 锁或写文件。
    raise SystemExit(128 + signum)


def install() -> None:
    config_path = Path(os.environ["TRACE_VLLM_STARTUP_CONFIG"]).resolve()
    output_prefix = Path(os.environ["TRACE_VLLM_STARTUP_PREFIX"]).resolve()
    config = load_config(config_path)
    session = TraceSession(
        root=config.root,
        targets=config.targets,
        output_prefix=output_prefix,
        run_id=os.environ["TRACE_VLLM_STARTUP_RUN_ID"],
        config_path=config_path,
        process_label=os.environ.get("TRACE_PROCESS_LABEL", config.process_label),
    )
    capture = session.capture()
    capture.__enter__()
    atexit.register(capture.__exit__, None, None, None)
    if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
