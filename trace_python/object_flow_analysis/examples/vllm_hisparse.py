"""真实 vLLM 单请求，复用仓库已有 HiSparse dummy 权重配置。"""
import ast
from argparse import Namespace
import os
from pathlib import Path
import runpy
import sys
from typing import cast

from trace_python.object_flow_analysis import FlowConfig, Origin

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "code/vllm-workspace/vllm"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"


def run():
    # uni executor 与禁用 engine 多进程让实际 Scheduler 留在受观察进程内。
    previous = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    old_path = sys.path[:]
    engine = None
    try:
        sys.path.insert(0, str(REPO / "code/dev"))
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.outputs import RequestOutput
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.v1.engine.llm_engine import LLMEngine

        serve = runpy.run_path(str(REPO / "code/dev/serve_config/deepseek-v3.2-hisparse.py"))["config"]
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        parsed = cast(Namespace, parser.parse_args([
            "--model", serve.model, *serve.serve_command[3:],
            "--distributed-executor-backend", "uni",
        ]))
        engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(parsed))
        engine.add_request(
            "object-flow-hisparse", {"prompt_token_ids": [100] * 2304},
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        )
        while engine.has_unfinished_requests():
            for output in engine.step():
                output = cast(RequestOutput, output)
                if output.finished:
                    print("REQUEST_RESULT", output.request_id,
                          "generated_tokens", len(output.outputs[0].token_ids),
                          "finish_reason", output.outputs[0].finish_reason)
    finally:
        if engine is not None:
            engine.engine_core.shutdown()
        sys.path[:] = old_path
        if previous is None:
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        else:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = previous


# 选择 schedule 中第一个 allocate_slots 赋值；记录真实源码位置，不写死行号。
source = ast.parse((ROOT / SCHEDULER).read_bytes())
line = min(node.lineno for node in ast.walk(source)
           if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
           and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "allocate_slots"
           and any(isinstance(target, ast.Name) and target.id == "new_blocks" for target in node.targets))
config = FlowConfig(
    root=ROOT,
    origin=Origin(SCHEDULER, "Scheduler.schedule", line, "new_blocks"),
    run=run,
    include=(SCHEDULER, "vllm/v1/core/kv_cache_manager.py", "vllm/v1/core/sched/output.py"),
    max_events=3000,
    max_objects=1000,
)
