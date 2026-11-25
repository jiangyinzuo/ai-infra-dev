"""DP=2, EP=2 tiny-MoE trace: finite requests that trigger synchronous EPLB.

Both GPUs are one-token/attention DP replica each, while routed experts are
sharded across the two EP ranks.  This topology exercises explicit dispatch
and combine as well as the EPLB map/weight migration path.
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
PREPARE = "vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py"
CUDA = "vllm/distributed/device_communicators/cuda_communicator.py"
PARALLEL = "vllm/distributed/parallel_state.py"
STATE = "vllm/distributed/eplb/eplb_state.py"
POLICY = "vllm/distributed/eplb/policy/default.py"
MIGRATE = "vllm/distributed/eplb/rebalance_execute.py"
COMM = "vllm/distributed/eplb/eplb_communicator.py"


# Four steps is finite and intentionally small for this teaching capture:
# EplbState starts at 3/4, records the four-step window, then rebalances.
EPLB = {
    "window_size": 4,
    "step_interval": 4,
    "num_redundant_experts": 2,
    "log_balancedness": True,
    "log_balancedness_interval": 1,
    "use_async": False,
    "policy": "default",
    "communicator": "torch_nccl",
}


def run(start) -> None:
    # Repeated but non-empty prompts create several engine steps and a stable
    # routing workload; the trace, not this input choice, is the proof of load.
    prompt = "route this repeated token through the tiny MoE " * 96
    run_service(
        start,
        config_path=Path(__file__).resolve(),
        scenario="expert-load-balance-dp2",
        tp_size=1,
        dp_size=2,
        enable_eplb=True,
        eplb_config=EPLB,
        requests=tuple(
            (f"eplb-{index:02d}", prompt + f" request {index}") for index in range(8)
        ),
    )


config = TraceConfig(
    root=ROOT,
    run=run,
    env={"HF_HUB_OFFLINE": "1"},
    process_label=os.environ.get("TRACE_PROCESS_LABEL", "expert-load-balance-dp2"),
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
            note="根据 router logits 选出 top-k 逻辑专家",
        ),
        Target(
            ROUTER,
            "BaseRouter._apply_eplb_mapping",
            ("topk_ids",),
            4096,
            note="把逻辑 ID 映射为可执行的物理副本 ID",
        ),
        Target(
            PREPARE,
            "MoEPrepareAndFinalizeNaiveDPEPModular.prepare",
            (
                "a1",
                "topk_weights",
                "topk_ids",
                "num_experts",
            ),
            4096,
            note="量化并 dispatch hidden states、权重和 expert IDs",
        ),
        Target(
            PREPARE,
            "MoEPrepareAndFinalizeNaiveDPEPModular.finalize",
            (
                "output",
                "fused_expert_output",
                "topk_weights",
                "topk_ids",
            ),
            4096,
            note="对专家输出加 top-k 权重并 combine 回原 token",
        ),
        Target(
            RUNNER,
            "MoERunner._maybe_dispatch",
            (
                "hidden_states",
                "router_logits",
            ),
            4096,
            note="DP/EP dispatch 总入口",
        ),
        Target(
            RUNNER,
            "MoERunner._maybe_combine",
            ("hidden_states",),
            4096,
            note="DP/EP combine 总入口",
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
            note="路由、专家计算与最终输出的框架入口",
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
            note="物理 expert 上执行两次 GEMM 与激活",
        ),
        Target(
            CUDA,
            "CudaCommunicator.dispatch",
            (
                "hidden_states",
                "topk_weights",
                "topk_ids",
            ),
            4096,
            note="把 token 和路由元数据送到持有目标专家的 rank",
        ),
        Target(
            CUDA,
            "CudaCommunicator.combine",
            ("hidden_states",),
            4096,
            note="把各 EP rank 专家输出合并回 token",
        ),
        Target(
            PARALLEL,
            "GroupCoordinator.dispatch",
            (
                "hidden_states",
                "topk_weights",
                "topk_ids",
            ),
            4096,
        ),
        Target(
            PARALLEL,
            "GroupCoordinator.combine_into_output",
            (
                "hidden_states",
                "output",
            ),
            4096,
            note="将专家输出归约回请求的输出缓冲区",
        ),
        Target(
            CUDA,
            "CudaCommunicator.combine_into_output",
            (
                "hidden_states",
                "output",
            ),
            4096,
            note="执行 combine collective",
        ),
        # EPLB state: a valid window is recorded before step_interval and
        # synchronous rearrange executes on all EP ranks at the boundary.
        Target(
            STATE,
            "EplbState.prepare_forward",
            (
                "model_config",
                "num_unpadded_tokens",
            ),
            4096,
        ),
        Target(
            STATE,
            "EplbState.step",
            (
                "is_dummy",
                "is_profile",
                "log_stats",
                "self.expert_rearrangement_step",
                "self.expert_rearrangement_step_interval",
            ),
            4096,
            note="推进滑动窗口和重平衡计数器",
        ),
        Target(
            STATE,
            "EplbState.rearrange",
            (
                "is_profile",
                "rank_mapping",
                "use_last_expert_load",
            ),
            4096,
            note="由窗口负载计算新 physical-to-logical map",
        ),
        Target(
            POLICY,
            "DefaultEplbPolicy.rebalance_experts",
            (
                "stats.global_expert_load_window",
                "stats.num_replicas",
                "stats.num_groups",
                "stats.num_nodes",
                "stats.num_gpus",
                "old_global_expert_indices",
            ),
            4096,
            note="默认层次 packing/replication 策略",
        ),
        Target(
            STATE,
            "EplbModelState._commit_eplb_maps",
            ("new_physical_to_logical_map",),
            4096,
            note="提交新映射，供后续 router 使用",
        ),
        # Entry into rearrange is not migration proof.  These targets expose
        # old/new maps and the actual per-rank communicator copy operations.
        Target(
            MIGRATE,
            "rearrange_expert_weights_inplace",
            (
                "old_global_expert_indices",
                "new_global_expert_indices",
                "is_profile",
                "rank_mapping",
            ),
            4096,
            note="按 old/new map 计划真实权重搬运",
        ),
        Target(
            MIGRATE,
            "transfer_layer",
            (
                "old_layer_indices",
                "new_layer_indices",
                "layer_idx",
                "is_profile",
            ),
            4096,
            note="逐 MoE layer 交换 physical expert 权重",
        ),
        Target(
            MIGRATE,
            "move_to_buffer",
            (
                "old_indices",
                "new_indices",
                "layer_idx",
                "ep_rank",
            ),
            4096,
            note="先做本地临时拷贝，再收发跨 rank 权重",
        ),
        Target(
            MIGRATE,
            "move_from_buffer",
            (
                "new_indices",
                "ep_rank",
                "transfer_metadata",
            ),
            4096,
            note="把收到的权重写回目标 physical slot",
        ),
        Target(
            COMM,
            "TorchDistNcclEplbCommunicator.add_send",
            (
                "dst_rank",
                "expert_id",
                "tensors",
            ),
            4096,
        ),
        Target(
            COMM,
            "TorchDistNcclEplbCommunicator.add_recv",
            (
                "src_rank",
                "expert_id",
                "tensors",
            ),
            4096,
        ),
        Target(
            COMM,
            "TorchDistNcclEplbCommunicator.execute",
            ("self._p2p_ops",),
            4096,
            caller_fields=("send_count", "recv_count", "layer_idx", "ep_rank"),
            note="提交非空权重传输队列并等待完成",
        ),
    ),
)
