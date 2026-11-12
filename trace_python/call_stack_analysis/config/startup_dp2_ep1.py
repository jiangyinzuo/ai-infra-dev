"""Trace process and engine initialization for the DP=2, EP=1 recipe."""

from pathlib import Path

from trace_python.call_stack_analysis.config.startup_common import make_config

config = make_config("dp2-ep1", Path(__file__).resolve(), process_label="dp2-ep1")
