"""Shared real two-GPU tiny-MoE runners for the EP/EPLB trace cases.

The service is a child process on purpose: its ``sitecustomize`` installs the
startup tracer before vLLM imports.  Requests are sent over HTTP so the trace
contains the same API/EngineCore/worker process topology as a deployment.
"""

from __future__ import annotations

import json
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[3]
VLLM_ROOT = REPOSITORY / "code/vllm-workspace/vllm"
RECIPE = REPOSITORY / "code/dev/serve_config/deepseek-v3.2-hisparse.py"


def _phase(name: str, request_id: str | None = None) -> None:
    print(
        "TRACE_PHASE", json.dumps({"name": name, "request_id": request_id}), flush=True
    )


def _remove_flag(arguments: list[str], flag: str) -> None:
    """Remove a flag and its JSON/string value from a recipe command."""
    while flag in arguments:
        index = arguments.index(flag)
        del arguments[index : index + 2]


def command(
    *,
    tp_size: int,
    dp_size: int,
    enable_eplb: bool = False,
    eplb_config: dict[str, Any] | None = None,
    port: int,
) -> list[str]:
    """Build a deployment command from the checked-in tiny random MoE recipe."""
    sys.path.insert(0, str(REPOSITORY / "code/dev"))
    recipe = runpy.run_path(str(RECIPE))["config"]
    arguments = list(recipe.serve_command[3:])

    # These examples are ordinary MoE/EP runs, not HiSparse or KV transfer.
    _remove_flag(arguments, "--kv-transfer-config")
    attention_index = arguments.index("--attention-config") + 1
    attention = json.loads(arguments[attention_index])
    attention.pop("hisparse_config", None)
    arguments[attention_index] = json.dumps(attention)

    tensor_index = arguments.index("--tensor-parallel-size") + 1
    arguments[tensor_index] = str(tp_size)
    # The base tiny recipe has eight heads for TP=1.  TP=2 must retain eight
    # heads per rank, hence a total of sixteen heads.
    overrides_index = arguments.index("--hf-overrides") + 1
    overrides = json.loads(arguments[overrides_index])
    overrides["num_attention_heads"] = 8 * tp_size
    arguments[overrides_index] = json.dumps(overrides)

    arguments.extend(
        [
            "--data-parallel-size",
            str(dp_size),
            "--data-parallel-size-local",
            str(dp_size),
            "--enable-expert-parallel",
            "--all2all-backend",
            "allgather_reducescatter",
            "--distributed-executor-backend",
            "mp",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--shutdown-timeout",
            "30",
        ]
    )
    if enable_eplb:
        assert eplb_config is not None
        arguments.extend(["--enable-eplb", "--eplb-config", json.dumps(eplb_config)])

    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        recipe.model,
        *arguments,
    ]


def _request(port: int, model: str, prompt: str, request_id: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "user": request_id,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"{request_id}: HTTP {response.status}")
        result = json.loads(response.read())
    generated_tokens = result.get("usage", {}).get("completion_tokens")
    if generated_tokens != 8 or not result.get("choices"):
        raise RuntimeError(
            f"{request_id}: expected 8 generated tokens, got {generated_tokens}"
        )
    return {
        "request_id": request_id,
        "response_id": result.get("id"),
        "generated_tokens": generated_tokens,
        "finish_reason": result["choices"][0].get("finish_reason"),
    }


def _wait_ready(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 300
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise subprocess.CalledProcessError(process.returncode or 1, process.args)
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"vLLM API did not become ready at {url}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=75)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()


def run_service(
    start,
    *,
    config_path: Path,
    scenario: str,
    tp_size: int,
    dp_size: int,
    requests: tuple[tuple[str, str], ...],
    enable_eplb: bool = False,
    eplb_config: dict[str, Any] | None = None,
) -> None:
    """Run one finite HTTP workload under startup tracing."""
    start()
    _phase("scenario_start", scenario)
    port = int(os.environ.get("TRACE_EXPERT_PORT", "18080"))
    run_id = uuid.uuid4().hex
    prefix = (
        Path(
            os.environ.get(
                "TRACE_VLLM_STARTUP_OUTPUT_PREFIX",
                str(Path.home() / "vllm-trace-python-log" / "startup-trace" / scenario),
            )
        )
        .resolve()
        .with_name(f"{scenario}-{run_id[:8]}")
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    devices = env.get("TRACE_VLLM_STARTUP_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices
    env.update(
        {
            "TRACE_VLLM_STARTUP_CONFIG": str(config_path.resolve()),
            "TRACE_VLLM_STARTUP_PREFIX": str(prefix),
            "TRACE_VLLM_STARTUP_RUN_ID": run_id,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    existing_pythonpath = env.get("PYTHONPATH")
    with tempfile.TemporaryDirectory(prefix="vllm-expert-trace-") as bootstrap_dir:
        Path(bootstrap_dir, "sitecustomize.py").write_text(
            "from trace_python.call_stack_analysis.startup_bootstrap import install\n"
            "install()\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (bootstrap_dir, str(REPOSITORY), existing_pythonpath)
            if value
        )
        process = subprocess.Popen(
            command(
                tp_size=tp_size,
                dp_size=dp_size,
                enable_eplb=enable_eplb,
                eplb_config=eplb_config,
                port=port,
            ),
            cwd=VLLM_ROOT,
            env=env,
            start_new_session=True,
        )
        try:
            _wait_ready(port, process)
            _phase("engine_ready")
            model = "deepseek-ai/DeepSeek-V3.2"
            for request_id, prompt in requests:
                _phase("request", request_id)
                result = _request(port, model, prompt, request_id)
                print("REQUEST_RESULT", json.dumps(result), flush=True)
                _phase("request_finished", request_id)
        finally:
            _stop(process)
    if process.returncode not in (0, 130, -signal.SIGINT):
        raise subprocess.CalledProcessError(process.returncode, process.args)
    _phase("scenario_finished", scenario)
