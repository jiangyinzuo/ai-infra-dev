"""Trace the launcher, EngineCore, Worker, and two API server processes."""

from pathlib import Path

from trace_python.call_stack_analysis.config.startup_common import make_config

config = make_config("multi-api", Path(__file__).resolve())
