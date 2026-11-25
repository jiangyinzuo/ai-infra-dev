"""双卡 DCP：dense MLA、TP=2、DCP=2，观察 KV 分片和 LSE 归并。"""

from pathlib import Path

from trace_python.call_stack_analysis.config.context_parallel_common import make_config

config = make_config("dcp", Path(__file__).resolve())
