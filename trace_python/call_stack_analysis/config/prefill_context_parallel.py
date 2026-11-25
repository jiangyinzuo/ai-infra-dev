"""双卡 PCP：TP=1、PCP=2、DCP=1，观察头尾分块和 KV/hidden 收集。"""

from pathlib import Path

from trace_python.call_stack_analysis.config.context_parallel_common import make_config

config = make_config("pcp", Path(__file__).resolve())
