"""Trace a single API process with separately spawned EngineCore and Worker."""

from pathlib import Path

from trace_python.call_stack_analysis.config.startup_common import make_config

config = make_config("single-api", Path(__file__).resolve())
