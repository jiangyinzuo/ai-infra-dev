"""Trace headless EngineCore and Worker startup without API processes."""

from pathlib import Path

from trace_python.call_stack_analysis.config.startup_common import make_config

config = make_config("headless", Path(__file__).resolve())
