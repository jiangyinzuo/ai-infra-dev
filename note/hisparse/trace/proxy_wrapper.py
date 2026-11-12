#!/usr/bin/env python3
"""Start toy proxy with an in-process trace session."""

import os
from pathlib import Path
import runpy
import sys

from trace_python.call_stack_analysis import TraceSession
from trace_python.call_stack_analysis.configuration import load_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "trace_python/call_stack_analysis/config/pd_hisparse_attach.py"


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: proxy_wrapper.py PROXY_SCRIPT [proxy args ...]")
    proxy = Path(sys.argv[1]).resolve()
    sys.argv = [str(proxy), *sys.argv[2:]]
    config = load_config(CONFIG)
    prefix = Path(os.environ["TRACE_PROXY_OUTPUT_PREFIX"])
    session = TraceSession(
        config.root,
        config.targets,
        prefix,
        config_path=CONFIG,
    )
    with session.capture():
        runpy.run_path(str(proxy), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
