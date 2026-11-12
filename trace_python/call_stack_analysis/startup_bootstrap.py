"""Shared process-wide startup tracing for vLLM CLI configurations."""

from __future__ import annotations

import atexit
import os
from pathlib import Path

from trace_python.call_stack_analysis import TraceSession
from trace_python.call_stack_analysis.configuration import load_config


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
    )
    capture = session.capture()
    capture.__enter__()
    atexit.register(capture.__exit__, None, None, None)
