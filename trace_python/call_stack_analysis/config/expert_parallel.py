"""TP=2, EP=2 tiny-MoE trace: logical routing and local expert compute.

This is the clean expert-parallel case: two GPUs form one TP/EP group and
there is no DP replica.  It deliberately keeps EPLB off so routing, physical
expert lookup, and compute can be read without a rebalance interleaving them.
"""

import os
from pathlib import Path

from trace_python.call_stack_analysis import Target, TraceConfig
from trace_python.call_stack_analysis.config.expert_parallel_common import (
    run_service,
)

ROOT = Path(__file__).resolve().parents[3] / "code/vllm-workspace/vllm"


ROUTER = "vllm/model_executor/layers/fused_moe/router/base_router.py"
RUNNER = "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
EXPERTS = "vllm/model_executor/layers/fused_moe/routed_experts.py"
CUDA = "vllm/distributed/device_communicators/cuda_communicator.py"
PARALLEL = "vllm/distributed/parallel_state.py"


def run(start) -> None:
    run_service(
        start,
        config_path=Path(__file__).resolve(),
        scenario="expert-parallel-tp2",
        tp_size=2,
        dp_size=1,
        requests=(
            ("ep-tp2-a", "Explain expert parallelism with a short answer. " * 32),
            ("ep-tp2-b", "Explain tensor parallelism with a short answer. " * 32),
        ),
    )


config = TraceConfig(
    root=ROOT,
    run=run,
    env={"HF_HUB_OFFLINE": "1"},
    process_label=os.environ.get("TRACE_PROCESS_LABEL", "expert-parallel-tp2"),
    targets=(
        Target(
            str(Path(__file__).with_name("expert_parallel_common.py")),
            "_phase",
            ("name", "request_id"),
            32,
        ),
        Target(
            ROUTER,
            "BaseRouter._select_experts",
            (
                "hidden_states",
                "router_logits",
                "topk_indices_dtype",
            ),
            4096,
            note="由 router logits 选出每个 token 的 top-k 逻辑专家",
        ),
        Target(
            ROUTER,
            "BaseRouter._apply_eplb_mapping",
            ("topk_ids",),
            4096,
            note="EPLB 关闭时保持逻辑 ID；开启时转成物理 ID",
        ),
        Target(
            RUNNER,
            "MoERunner._forward_impl",
            (
                "hidden_states",
                "router_logits",
                "input_ids",
            ),
            4096,
            note="MoE 一层的路由、专家计算和输出合并总入口",
        ),
        Target(
            RUNNER,
            "MoERunner._maybe_dispatch",
            (
                "hidden_states",
                "router_logits",
            ),
            4096,
            note="DP/序列并行才会发生的显式 dispatch",
        ),
        Target(
            RUNNER,
            "MoERunner._maybe_combine",
            ("hidden_states",),
            4096,
            note="DP/序列并行才会发生的显式 combine",
        ),
        Target(
            EXPERTS,
            "RoutedExperts.forward_modular",
            (
                "x",
                "topk_weights",
                "topk_ids",
            ),
            4096,
            note="按物理 expert map 执行两次 GEMM 与激活",
        ),
        # TP=2, DP=1 does not need an all-to-all.  These targets make that
        # distinction observable: a DP=2 scenario should hit them.
        Target(
            CUDA,
            "CudaCommunicator.dispatch",
            (
                "hidden_states",
                "topk_weights",
                "topk_ids",
            ),
            4096,
            note="跨 DP/EP rank 分发 token 与 top-k 元数据",
        ),
        Target(
            CUDA,
            "CudaCommunicator.combine",
            ("hidden_states",),
            4096,
            note="跨 DP/EP rank 合并专家输出",
        ),
        Target(
            PARALLEL,
            "GroupCoordinator.dispatch",
            (
                "hidden_states",
                "topk_weights",
                "topk_ids",
            ),
            32,
        ),
        Target(PARALLEL, "GroupCoordinator.combine", ("hidden_states",), 32),
    ),
)
