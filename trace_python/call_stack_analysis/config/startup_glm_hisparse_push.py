"""复现 GLM-5.3 HiSparse + push 的 Decode KV 注册失败，不启动 P/Proxy。"""

from pathlib import Path

from trace_python.call_stack_analysis import Target, TraceConfig
from trace_python.call_stack_analysis.config.startup_common import (
    TARGETS,
    VLLM_ROOT,
    run_startup,
)


def run(start):
    run_startup(start, "glm-hisparse-push-decode", Path(__file__).resolve())


_NIXL = "vllm/distributed/kv_transfer/kv_connector/v1/nixl/"

config = TraceConfig(
    root=VLLM_ROOT,
    run=run,
    process_label="D",
    env={
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "5601",
        "UCX_NET_DEVICES": "all",
    },
    targets=(
        *TARGETS,
        Target(
            _NIXL + "push_worker.py",
            "NixlPushConnectorWorker.register_kv_caches",
            ("self.engine_id", "self._mixed_mem_types"),
        ),
        Target(
            _NIXL + "base_worker.py",
            "NixlBaseConnectorWorker.register_kv_caches",
            ("self.engine_id", "self.use_mla"),
        ),
        Target(
            _NIXL + "base_worker.py",
            "NixlBaseConnectorWorker.register_local_xfer_handler",
            (
                "self.engine_id",
                "self._mixed_mem_types",
                "self.region_mem_types",
                "self.region_names",
                "self.region_num_blocks",
            ),
        ),
    ),
)
