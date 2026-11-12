"""Run vLLM startup under process-wide function-entry tracing."""

from __future__ import annotations

import json
import os
import runpy
import shlex
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from trace_python.call_stack_analysis import Target, TraceConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = REPOSITORY_ROOT / "code/vllm-workspace/vllm"
SERVE_CONFIG = REPOSITORY_ROOT / "code/dev/serve_config/deepseek-v3.2-hisparse.py"
sys.path.insert(0, str(REPOSITORY_ROOT / "code/dev"))

TARGETS = (
    Target(
        "vllm/entrypoints/cli/serve.py",
        "ServeSubcommand.cmd",
        ("args.headless", "args.api_server_count"),
    ),
    Target("vllm/entrypoints/cli/serve.py", "run_headless"),
    Target("vllm/entrypoints/launchers/api_server/entry.py", "run_server"),
    Target("vllm/entrypoints/launchers/api_server/entry.py", "run_server_worker"),
    Target(
        "vllm/entrypoints/launchers/api_server/entry.py",
        "build_async_engine_client_from_engine_args",
    ),
    Target("vllm/v1/engine/async_llm.py", "AsyncLLM.from_vllm_config"),
    Target("vllm/v1/engine/core_client.py", "EngineCoreClient.make_async_mp_client"),
    Target("vllm/v1/engine/utils.py", "launch_core_engines"),
    Target("vllm/v1/engine/utils.py", "CoreEngineProcManager.__init__"),
    Target("vllm/v1/engine/core.py", "EngineCoreProc.run_engine_core"),
    Target("vllm/v1/engine/core.py", "EngineCoreProc.__init__"),
    Target("vllm/v1/engine/core.py", "EngineCore.__init__"),
    Target("vllm/v1/engine/core.py", "EngineCore._initialize_kv_caches"),
    Target("vllm/v1/executor/multiproc_executor.py", "MultiprocExecutor.__init__"),
    Target("vllm/v1/executor/multiproc_executor.py", "WorkerProc.__init__"),
    Target("vllm/v1/executor/multiproc_executor.py", "WorkerProc.worker_main"),
    Target("vllm/v1/worker/worker_base.py", "WorkerWrapperBase.init_worker"),
    Target("vllm/v1/worker/worker_base.py", "WorkerWrapperBase.init_device"),
    Target("vllm/v1/worker/worker_base.py", "WorkerWrapperBase.initialize_from_config"),
    Target("vllm/v1/worker/gpu_worker.py", "Worker.init_device"),
    Target("vllm/v1/worker/gpu_worker.py", "Worker.load_model"),
    Target("vllm/v1/worker/gpu_worker.py", "Worker.initialize_from_config"),
)


def make_config(mode: str, config_path: Path) -> TraceConfig:
    def run(start) -> None:
        run_startup(start, mode, config_path)

    return TraceConfig(root=VLLM_ROOT, run=run, targets=TARGETS)


def _command(mode: str) -> list[str]:
    serve_config = runpy.run_path(str(SERVE_CONFIG))["config"]
    tokens = [
        token.replace("{model}", serve_config.model)
        for token in serve_config.serve_command[1:]
    ]
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", *tokens]
    tensor_parallel_size = os.environ.get(
        "TRACE_VLLM_STARTUP_TENSOR_PARALLEL_SIZE", "2"
    )
    try:
        tensor_parallel_index = command.index("--tensor-parallel-size")
    except ValueError:
        command.extend(("--tensor-parallel-size", tensor_parallel_size))
    else:
        command[tensor_parallel_index + 1] = tensor_parallel_size
    if mode == "multi-api":
        command.extend(("--api-server-count", "2"))
    elif mode == "headless":
        command.append("--headless")
    return command


def _write_timeline(prefix: Path) -> Path:
    events = []
    for path in prefix.parent.glob(prefix.name + ".*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("type") == "event":
                events.append(record)
    events.sort(
        key=lambda record: (record["monotonic_ns"], record["pid"], record["sequence"])
    )
    timeline = prefix.with_suffix(".timeline.txt")
    if events:
        origin = events[0]["monotonic_ns"]
        lines = [
            f"+{(event['monotonic_ns'] - origin) / 1_000_000:10.3f} ms "
            f"pid={event['pid']} {event['process_name']} #{event['sequence']} "
            f"{event['target']}"
            for event in events
        ]
        timeline.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n初始化入口时间线：{timeline}", flush=True)
        print(timeline.read_text(encoding="utf-8"), end="", flush=True)
    else:
        timeline.write_text("没有捕获到初始化入口事件。\n", encoding="utf-8")
        print(f"没有捕获到初始化入口事件；检查进程日志：{prefix.parent}", flush=True)
    return timeline


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_startup(start, mode: str, config_path: Path) -> None:
    if mode not in {"single-api", "multi-api", "headless"}:
        raise ValueError(f"unsupported startup mode: {mode}")
    start()
    run_id = uuid.uuid4().hex
    base_prefix = Path(
        os.environ.get(
            "TRACE_VLLM_STARTUP_OUTPUT_PREFIX", f"runtime/startup-trace/{mode}"
        )
    ).resolve()
    prefix = base_prefix.with_name(f"{base_prefix.name}-{run_id[:8]}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    with tempfile.TemporaryDirectory(prefix="vllm-startup-trace-") as bootstrap_dir:
        Path(bootstrap_dir, "sitecustomize.py").write_text(
            "from trace_python.call_stack_analysis.startup_bootstrap import install\n"
            "install()\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (bootstrap_dir, str(REPOSITORY_ROOT), existing_pythonpath)
            if value
        )
        env["TRACE_VLLM_STARTUP_CONFIG"] = str(config_path.resolve())
        env["TRACE_VLLM_STARTUP_PREFIX"] = str(prefix)
        env["TRACE_VLLM_STARTUP_RUN_ID"] = run_id
        # Spawn reruns sitecustomize in every API, EngineCore, and Worker process.
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        command = _command(mode)
        print("命令：" + shlex.join(command), flush=True)
        process = subprocess.Popen(
            command, cwd=VLLM_ROOT, env=env, start_new_session=True
        )
        seconds = float(os.environ.get("TRACE_VLLM_STARTUP_SECONDS", "0"))
        try:
            if seconds > 0:
                try:
                    process.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    print(f"采集 {seconds:g} 秒到期，正在停止服务进程组。", flush=True)
                    _stop(process)
            else:
                process.wait()
        except KeyboardInterrupt:
            print("收到 Ctrl-C，正在停止服务进程组。", flush=True)
            _stop(process)
        finally:
            if process.poll() is None:
                _stop(process)
    _write_timeline(prefix)
