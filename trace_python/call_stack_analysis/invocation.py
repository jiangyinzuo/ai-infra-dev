"""命令行采集的本机复现信息；只记录明确允许的环境变量。"""

import json
import os
import shlex
import sys
from pathlib import Path

_ENVIRONMENT_KEYS = (
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "TRACE_PROCESS_LABEL",
    "TRACE_VLLM_STARTUP_DEVICES",
    "TRACE_VLLM_STARTUP_OUTPUT_PREFIX",
    "TRACE_VLLM_STARTUP_SECONDS",
    "TRACE_VLLM_STARTUP_TENSOR_PARALLEL_SIZE",
    "VLLM_NIXL_SIDE_CHANNEL_HOST",
    "VLLM_NIXL_SIDE_CHANNEL_PORT",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "UCX_NET_DEVICES",
    "HF_HUB_OFFLINE",
    "MOONCAKE_CONFIG_PATH",
    "TRACE_KV_POOLING_OUTPUT_DIRECTORY",
    "TRACE_KV_POOLING_PORT",
    "TRACE_KV_POOLING_READY_SECONDS",
    "TRACE_EXPERT_PORT",
)


def write_command(prefix: Path, *, argv: list[str] | None = None) -> None:
    """在执行配置前记录入口命令；None 环境值表示复现时应取消设置。"""
    command = list(sys.orig_argv if argv is None else argv)
    cwd = str(Path.cwd())
    environment = {key: os.environ.get(key) for key in _ENVIRONMENT_KEYS}
    shell = ["env"]
    for key, value in environment.items():
        if value is None:
            shell.extend(["-u", key])
    shell.extend(f"{key}={value}" for key, value in environment.items()
                 if value is not None)
    shell.extend(command)
    record = {
        "argv": command,
        "command": shlex.join(command),
        "cwd": cwd,
        "env": environment,
        "replay_command": f"cd {shlex.quote(cwd)} && {shlex.join(shell)}",
    }
    Path(str(prefix.resolve()) + ".command.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
