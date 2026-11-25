"""真实 Mooncake Store：冷写入，只清本地 APC，两次读回同一前缀。"""

from __future__ import annotations

import json
import os
import runpy
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from trace_python.call_stack_analysis import Target, TraceConfig
from trace_python.call_stack_analysis.config.startup_common import _stop

REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = REPOSITORY / "code/vllm-workspace/vllm"
STORE = "vllm/distributed/kv_transfer/kv_connector/v1/mooncake_store/"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
MANAGER = "vllm/v1/core/kv_cache_manager.py"
REQUEST_FIELDS = ("request.request_id", "request.num_computed_tokens")


def _command(port: int, namespace: str) -> list[str]:
    sys.path.insert(0, str(REPOSITORY / "code/dev"))
    recipe = runpy.run_path(
        str(REPOSITORY / "code/dev/serve_config/deepseek-v3.2-hisparse.py")
    )["config"]
    arguments = list(recipe.serve_command[3:])
    index = arguments.index("--attention-config") + 1
    attention = json.loads(arguments[index])
    attention.pop("hisparse_config")
    arguments[index] = json.dumps(attention)
    index = arguments.index("--kv-transfer-config") + 1
    arguments[index] = json.dumps(
        {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "cache_prefix": namespace,
                "load_async": True,
                "lookup_async": True,
                "lookup_prefetch_limit": 4,
                "save_decode_cache": False,
            },
        }
    )
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        recipe.model,
        *arguments,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "kv-pooling-teaching",
        "--enable-prompt-tokens-details",
        "--shutdown-timeout",
        "30",
    ]


def _http(base: str, path: str, payload: dict | None = None) -> str:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base + path,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=300) as response:
        return response.read().decode()


def _metric(text: str, metric: str, operation: str, status: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(metric + "{"):
            continue
        labels, value = line.rsplit("} ", 1)
        if f'operation="{operation}"' in labels and f'status="{status}"' in labels:
            total += float(value.split()[0])
    return total


def _wait_metrics(
    base: str,
    process: subprocess.Popen,
    operation: str,
    previous: float,
    artifact: Path,
) -> tuple[str, float]:
    # API 响应早于统计周期；等真实计数发布，不能用固定 sleep 冒充已完成传输。
    deadline = time.monotonic() + 45
    while process.poll() is None and time.monotonic() < deadline:
        text = _http(base, "/metrics")
        artifact.write_text(text, encoding="utf-8")
        for op in ("save_put", "load_get"):
            failed = sum(
                _metric(text, "vllm:mooncake_store_operation_total", op, status)
                for status in ("error", "partial_failure")
            )
            if failed:
                raise RuntimeError(f"{op}: Store 报告传输失败，见 {artifact}")
        value = _metric(
            text, "vllm:mooncake_store_operation_bytes_total", operation, "ok"
        )
        if value > previous:
            return text, value
        time.sleep(0.5)
    raise RuntimeError(f"没有观察到 {operation} 成功字节数增长，见 {artifact}")


def _reset_local(base: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 45
    while process.poll() is None and time.monotonic() < deadline:
        # reset_external=False：绝不删除 Store；异步保存引用释放前，reset 可返回 False。
        result = json.loads(_http(base, "/reset_prefix_cache?reset_external=false", {}))
        if result.get("success") is True:
            print("LOCAL_APC_RESET", json.dumps(result), flush=True)
            return
        time.sleep(0.2)
    raise RuntimeError("本地 APC 未成功重置；不能把后续命中归因于 Store")


def run(start) -> None:
    config_path = Path(os.environ["MOONCAKE_CONFIG_PATH"]).expanduser().resolve()
    mooncake_config = json.loads(config_path.read_text(encoding="utf-8"))
    devices = os.environ.get("TRACE_VLLM_STARTUP_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not devices or len(devices.split(",")) != 1:
        raise ValueError("本场景要求通过 TRACE_VLLM_STARTUP_DEVICES 指定一个 GPU UUID")
    run_id = uuid.uuid4().hex
    namespace = "trace-kv-pooling-" + run_id
    output_root = Path(os.environ.get(
        "TRACE_KV_POOLING_OUTPUT_DIRECTORY", str(Path.home() / "vllm-trace-python-log")
    )).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="kv-pooling-", dir=output_root))
    prefix = output_dir / "store"
    port = int(os.environ.get("TRACE_KV_POOLING_PORT", "8134"))
    base = f"http://127.0.0.1:{port}"
    command = _command(port, namespace)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "HF_HUB_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_SERVER_DEV_MODE": "1",
            "MOONCAKE_CONFIG_PATH": str(config_path),
            "TRACE_VLLM_STARTUP_CONFIG": str(Path(__file__).resolve()),
            "TRACE_VLLM_STARTUP_PREFIX": str(prefix),
            "TRACE_VLLM_STARTUP_RUN_ID": run_id,
        }
    )
    start()
    print("POOLING_ARTIFACTS", output_dir, flush=True)
    print("COMMAND", shlex.join(command), flush=True)
    with tempfile.TemporaryDirectory(prefix="bootstrap-", dir=output_dir) as bootstrap:
        Path(bootstrap, "sitecustomize.py").write_text(
            "from trace_python.call_stack_analysis.startup_bootstrap import install\n"
            "install()\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (
                    bootstrap,
                    str(REPOSITORY),
                    env.get("PYTHONPATH"),
                ),
            )
        )
        evidence = {
            "argv": command,
            "cwd": str(ROOT),
            "cache_prefix": namespace,
            "mooncake_config": mooncake_config,
            "environment": {
                key: env[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "HF_HUB_OFFLINE",
                    "MOONCAKE_CONFIG_PATH",
                    "VLLM_WORKER_MULTIPROC_METHOD",
                    "VLLM_SERVER_DEV_MODE",
                    "TRACE_VLLM_STARTUP_CONFIG",
                    "TRACE_VLLM_STARTUP_PREFIX",
                    "TRACE_VLLM_STARTUP_RUN_ID",
                )
            },
            "requests": [],
        }
        manifest = output_dir / "scenario.json"
        manifest.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        with (output_dir / "service.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + float(
                    os.environ.get("TRACE_KV_POOLING_READY_SECONDS", "600")
                )
                while True:
                    if process.poll() is not None:
                        raise subprocess.CalledProcessError(process.returncode, command)
                    try:
                        _http(base, "/health")
                        break
                    except (URLError, TimeoutError, ConnectionError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"vLLM 未就绪，见 {output_dir / 'service.log'}"
                            )
                        time.sleep(1)
                previous_load = 0.0
                for scenario in ("cold-save", "store-hit-1", "store-hit-2"):
                    if scenario != "cold-save":
                        _reset_local(base, process)
                    response = json.loads(
                        _http(
                            base,
                            "/v1/completions",
                            {
                                "model": "kv-pooling-teaching",
                                "prompt": [100] * 2303,
                                "max_tokens": 8,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        )
                    )
                    (output_dir / f"{scenario}.response.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    usage = response["usage"]
                    cached = usage["prompt_tokens_details"]["cached_tokens"]
                    record = {"scenario": scenario, "usage": usage}
                    evidence["requests"].append(record)
                    manifest.write_text(
                        json.dumps(evidence, indent=2), encoding="utf-8"
                    )
                    print("REQUEST_RESULT", json.dumps(record), flush=True)
                    if usage["completion_tokens"] != 8:
                        raise RuntimeError(f"{scenario}: 没有生成 8 token")
                    if scenario == "cold-save":
                        if cached != 0:
                            raise RuntimeError("独立 namespace 冷请求发生意外命中")
                        _, saved = _wait_metrics(
                            base,
                            process,
                            "save_put",
                            0,
                            output_dir / f"{scenario}.metrics.txt",
                        )
                        evidence["cold_save_bytes"] = saved
                    else:
                        if cached <= 0:
                            raise RuntimeError(f"{scenario}: 未复用前缀")
                        _, loaded = _wait_metrics(
                            base,
                            process,
                            "load_get",
                            previous_load,
                            output_dir / f"{scenario}.metrics.txt",
                        )
                        record["store_load_bytes_delta"] = loaded - previous_load
                        previous_load = loaded
                    manifest.write_text(
                        json.dumps(evidence, indent=2), encoding="utf-8"
                    )
                _reset_local(base, process)
                print(
                    "POOLING_PROOF",
                    json.dumps(
                        {
                            "cold_save_bytes": evidence["cold_save_bytes"],
                            "repeated_store_load_bytes": previous_load,
                            "local_apc_only_reset": True,
                        }
                    ),
                    flush=True,
                )
            finally:
                _stop(process)
    if process.returncode not in (0, -2, 130):
        raise subprocess.CalledProcessError(process.returncode, command)


config = TraceConfig(
    root=ROOT,
    run=run,
    env={"HF_HUB_OFFLINE": "1"},
    process_label=os.environ.get("TRACE_PROCESS_LABEL", "kv-pooling"),
    targets=(
        Target(STORE + "connector.py", "MooncakeStoreConnector.__init__", ("role",), 4),
        Target(
            STORE + "scheduler.py",
            "MooncakeStoreScheduler.start_lookup_prefetch",
            REQUEST_FIELDS,
            3,
            note="请求到达时发起真实 Store 查找",
        ),
        Target(
            STORE + "scheduler.py",
            "MooncakeStoreScheduler.get_num_new_matched_tokens",
            REQUEST_FIELDS + ("num_computed_tokens",),
            128,
        ),
        Target(STORE + "worker.py", "MooncakeStoreWorker.lookup", ("num_tokens",), 12),
        Target(
            STORE + "coordinator.py",
            "MooncakeStoreCoordinator.find_longest_cache_hit",
            ("max_length", "block_hashes"),
            32,
        ),
        Target(
            MANAGER,
            "KVCacheManager.allocate_slots",
            REQUEST_FIELDS
            + (
                "num_new_tokens",
                "num_new_computed_tokens",
                "num_external_computed_tokens",
            ),
            64,
            note="分配本地物理 block，并非分配 Store 地址",
        ),
        Target(
            STORE + "scheduler.py",
            "MooncakeStoreScheduler.update_state_after_alloc",
            REQUEST_FIELDS + ("num_external_tokens",),
            6,
        ),
        Target(STORE + "worker.py", "MooncakeStoreWorker.register_kv_caches", (), 2),
        Target(
            STORE + "layout.py",
            "RankLocalStoreLayout.prepare_values",
            ("chunks", "block_ids", "shard_ids"),
            64,
            note="逻辑 chunk 映射为本地缓冲区地址和长度",
        ),
        Target(STORE + "worker.py", "MooncakeStoreWorker.start_load_kv", (), 64),
        Target(
            STORE + "worker.py",
            "KVCacheStoreRecvingThread._handle_request",
            (
                "req_meta.req_id",
                "req_meta.load_spec.kvpool_cached_tokens",
                "req_meta.load_spec.vllm_cached_tokens",
            ),
            8,
            note="真实 GET 线程入口",
        ),
        Target(STORE + "worker.py", "MooncakeStoreWorker.wait_for_save", (), 64),
        Target(
            STORE + "scheduler.py",
            "MooncakeStoreScheduler._reference_save_blocks",
            (),
            64,
            note="保存任务为源 block 增加引用",
        ),
        Target(
            STORE + "worker.py",
            "KVCacheStoreSendingThread._handle_request",
            (
                "req_meta.req_id",
                "req_meta.store_job_id",
                "req_meta.token_len_chunk",
            ),
            64,
            note="等待 CUDA 事件后执行真实 PUT",
        ),
        Target(
            STORE + "worker.py",
            "KVTransferThread._record_operation",
            (
                "operation",
                "num_keys",
                "num_bytes",
                "status",
                "num_failed_keys",
            ),
            256,
            note="PUT/GET 返回后的成功与失败证据",
        ),
        Target(
            STORE + "worker.py",
            "KVCacheStoreSendingThread.finish_store_job",
            (
                "req_meta.req_id",
                "req_meta.store_job_id",
            ),
            64,
        ),
        Target(
            STORE + "worker.py",
            "MooncakeStoreWorker.build_connector_worker_meta",
            (),
            128,
        ),
        Target(
            STORE + "scheduler.py",
            "MooncakeStoreScheduler.update_connector_output",
            (),
            128,
            note="全部 worker 完成后释放保存引用",
        ),
        Target(
            STORE + "worker.py", "MooncakeStoreWorker.get_transfer_results", (), 128
        ),
        Target(SCHEDULER, "Scheduler._free_request", REQUEST_FIELDS, 3),
        Target(MANAGER, "KVCacheManager.free", REQUEST_FIELDS, 3),
        Target(
            MANAGER,
            "KVCacheManager.reset_prefix_cache",
            (),
            24,
            note="只使本地 APC 失效，Store 内容保留",
        ),
    ),
)
