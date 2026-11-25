"""PCP/DCP 教学共用场景：真实双卡服务、启动前采集、单请求与完整退出。"""

import json
import os
import runpy
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from urllib.request import Request, urlopen

from trace_python.call_stack_analysis import Target, TraceConfig
from trace_python.call_stack_analysis.config.startup_common import (
    REPOSITORY_ROOT,
    TARGETS,
    VLLM_ROOT,
    _stop,
    _write_timeline,
)

PCP = "vllm/v1/worker/gpu/pcp_manager.py"
DCP = "vllm/v1/attention/ops/dcp.py"
GROUP = "vllm/distributed/parallel_state.py"
TABLE = "vllm/v1/worker/gpu/block_table.py"
COMM_FIELDS = ("self.unique_name", "self.world_size", "self.rank_in_group", "input_")


def _command(mode: str) -> list[str]:
    sys.path.insert(0, str(REPOSITORY_ROOT / "code/dev"))
    recipe = runpy.run_path(
        str(REPOSITORY_ROOT / "code/dev/serve_config/deepseek-v3.2-hisparse.py")
    )["config"]
    args = list(recipe.serve_command[3:])
    index = args.index("--kv-transfer-config")
    del args[index : index + 2]
    attention_index = args.index("--attention-config") + 1
    overrides_index = args.index("--hf-overrides") + 1
    overrides = json.loads(args[overrides_index])
    model = recipe.model
    parallel = ["--prefill-context-parallel-size", "2"]
    if mode == "pcp":
        args[attention_index] = json.dumps({"backend": "FLASHINFER_MLA_SPARSE_SM120"})
    elif mode == "dcp":
        # SM120 sparse MLA 不返回 LSE；DCP 用真实支持 LSE 的 dense MLA 路径。
        model = "deepseek-ai/DeepSeek-V3"
        overrides.pop("index_topk")
        overrides["num_attention_heads"] = 16
        args[attention_index] = json.dumps(
            {
                "backend": "TRITON_MLA",
                "mla_prefill_backend": "FLASH_ATTN",
            }
        )
        args[args.index("--kv-cache-dtype") + 1] = "auto"
        args[args.index("--tensor-parallel-size") + 1] = "2"
        # 请求仅使用 token IDs；不依赖本地未缓存的 V3 tokenizer 文件。
        args.append("--skip-tokenizer-init")
        parallel = [
            "--decode-context-parallel-size",
            "2",
            "--dcp-comm-backend",
            "ag_rs",
        ]
    else:
        raise ValueError(mode)
    args[overrides_index] = json.dumps(overrides)
    args[args.index("--max-model-len") + 1] = "4096"
    args[args.index("--max-num-batched-tokens") + 1] = "512"
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        *args,
        *parallel,
        "--seed",
        "42",
        "--served-model-name",
        "tiny-cp",
        "--host",
        "127.0.0.1",
        "--port",
        "8350",
        "--shutdown-timeout",
        "30",
    ]


def _run(start, mode: str, config_path: Path) -> None:
    start()
    devices = os.environ.get("TRACE_VLLM_STARTUP_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not devices or len(devices.split(",")) != 2:
        raise ValueError("PCP/DCP 教学需要两张 GPU，设置 TRACE_VLLM_STARTUP_DEVICES")
    prefix = Path(
        os.environ.get(
            "TRACE_VLLM_STARTUP_OUTPUT_PREFIX",
            str(
                Path.home() / "vllm-trace-python-log" / f"{mode}-{uuid.uuid4().hex[:8]}"
            ),
        )
    ).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = _command(mode)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices)
    env.update(
        TRACE_VLLM_STARTUP_CONFIG=str(config_path.resolve()),
        TRACE_VLLM_STARTUP_PREFIX=str(prefix),
        TRACE_VLLM_STARTUP_RUN_ID=uuid.uuid4().hex,
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TRACE_PROCESS_LABEL=mode,
    )
    payload = {
        "model": "tiny-cp",
        "prompt": [100] * 1024,
        "max_tokens": 8,
        "temperature": 0,
        "ignore_eos": True,
    }
    result = {"mode": mode, "command": command, "request": payload}
    print("BUSINESS_COMMAND", shlex.join(command), flush=True)
    with tempfile.TemporaryDirectory(prefix="cp-startup-") as bootstrap:
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
                    str(REPOSITORY_ROOT),
                    env.get("PYTHONPATH"),
                ),
            )
        )
        process = subprocess.Popen(
            command, cwd=VLLM_ROOT, env=env, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise subprocess.CalledProcessError(process.returncode, command)
                try:
                    with urlopen("http://127.0.0.1:8350/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("context-parallel API 启动超时") from None
                    time.sleep(1)
            request = Request(
                "http://127.0.0.1:8350/v1/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=120) as response:
                body = json.load(response)
            result["response"] = body
            if body["usage"]["completion_tokens"] != 8:
                raise RuntimeError(f"生成结果不符：{body}")
            print("REQUEST_RESULT", json.dumps(body), flush=True)
            _stop(process)
            result["exit_code"] = process.returncode
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                _stop(process)
            prefix.with_name(prefix.name + ".result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_timeline(prefix)


def make_config(mode: str, config_path: Path) -> TraceConfig:
    targets = (
        *(replace(target, limit=4096) for target in TARGETS),
        Target(
            TABLE,
            "BlockTables.compute_slot_mappings",
            (
                "self.cp_size",
                "self.cp_rank",
                "self.cp_interleave",
                "positions",
            ),
            128,
            note="按 CP rank 选择本地 KV 写入槽",
        ),
        Target(
            GROUP,
            "GroupCoordinator.all_gather",
            COMM_FIELDS + ("dim",),
            4096,
            note="收集组内张量；分组名区分 TP/PCP/DCP",
        ),
        Target(
            GROUP,
            "GroupCoordinator.reduce_scatter",
            COMM_FIELDS + ("dim",),
            4096,
            note="归约部分结果并切分输出",
        ),
        Target(
            GROUP,
            "GroupCoordinator.all_reduce",
            COMM_FIELDS,
            4096,
            note="归约组内部分结果",
        ),
    )
    if mode == "pcp":
        targets += (
            Target(
                PCP,
                "PCPManager.__init__",
                (
                    "pcp_world_size",
                    "pcp_rank",
                    "dcp_world_size",
                    "dcp_rank",
                    "shard_decode_requests",
                ),
                2,
                note="建立 PCP token 分工",
            ),
            Target(
                PCP,
                "PCPManager.partition_batch",
                (
                    "self.pcp_rank",
                    "input_batch.num_tokens",
                    "input_batch.req_ids",
                ),
                64,
                note="全局 batch 切为 rank 本地头尾分块",
            ),
            Target(
                PCP,
                "PCPManager._iter_rank_chunks",
                ("rank",),
                128,
                note="枚举头尾配对的 Query 分块",
            ),
            Target(
                PCP,
                "PCPManager.prepare_attn",
                (
                    "self.pcp_rank",
                    "input_batch.num_tokens",
                    "input_batch.positions",
                ),
                64,
                note="准备本地 Attention 与共享 KV 映射",
            ),
            Target(
                PCP,
                "PCPManager.restore_hidden_states",
                (
                    "self.pcp_rank",
                    "hidden_states",
                ),
                64,
                note="AllGather 后恢复全局 token 顺序",
            ),
            Target(
                "vllm/v1/attention/ops/pcp.py",
                "_gather_prefill_cache_inputs",
                (
                    "slot_mapping",
                    "num_decode_tokens",
                    "shard_decode_requests",
                ),
                128,
                note="收集 PCP rank 的 KV，保留缓存副本",
            ),
        )
    elif mode == "dcp":
        targets += (
            Target(
                "vllm/v1/worker/gpu/cp_utils.py",
                "prepare_dcp_local_seq_lens",
                (
                    "dcp_size",
                    "dcp_rank",
                    "cp_interleave",
                    "seq_lens",
                ),
                64,
                note="计算本 rank 持有的历史 KV 长度",
            ),
            Target(
                DCP,
                "MLADCPManager._gather_query",
                (
                    "self.group.world_size",
                    "self.group.rank_in_group",
                    "query",
                ),
                128,
                note="各 KV 分片收集所需 Query heads",
            ),
            Target(
                "vllm/v1/attention/backends/mla/triton_mla.py",
                "TritonMLAImpl.forward_mqa",
                (
                    "self.dcp_world_size",
                    "q",
                    "attn_metadata.decode.seq_lens",
                ),
                128,
                note="对本地 KV 分片计算输出与 LSE",
            ),
            Target(
                DCP,
                "cp_lse_ag_out_rs",
                (
                    "cp_group.world_size",
                    "cp_group.rank_in_group",
                    "cp_attn_out",
                    "cp_attn_lse",
                    "is_lse_base_on_e",
                ),
                128,
                note="收集 LSE、校正局部输出、ReduceScatter",
            ),
            Target(
                DCP,
                "correct_attn_out",
                (
                    "cp_rank",
                    "out",
                    "lses",
                    "is_lse_base_on_e",
                ),
                128,
                note="以全局 softmax 分母重新加权",
            ),
        )
    else:
        raise ValueError(mode)

    def run(start):
        _run(start, mode, config_path)

    return TraceConfig(root=VLLM_ROOT, run=run, targets=targets, process_label=mode)
