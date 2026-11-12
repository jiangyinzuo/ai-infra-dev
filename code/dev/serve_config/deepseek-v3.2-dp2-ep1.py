"""Tiny random DeepSeek-V3.2 with two DP replicas and no expert sharding."""
# ruff: noqa: N999

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve_config import ServeConfig

BASE = runpy.run_path(str(Path(__file__).with_name("deepseek-v3.2-hisparse.py")))["config"]
# EP=1：不传 --enable-expert-parallel；每个 DP rank 保留完整专家集合。

config = ServeConfig(
    model=BASE.model,
    device_count=2,
    serve_command=[
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        *BASE.serve_command[1:],
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
    ],
)
