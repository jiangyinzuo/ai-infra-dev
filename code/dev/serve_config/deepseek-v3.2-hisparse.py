"""DeepSeek-V3.2 随机权重、单卡 SM120 HiSparse 学习配置。"""

import json

from serve_config import ServeConfig

# 缩小层数、FFN 和专家数，但保留真实 MLA/DSA 路径；不能用它评价回答质量。
# SM120 sparse MLA 要求 top-k=2048、KV rank=512、RoPE=64，不能照搬测试的 top-k=128。
# 原 checkpoint 带 FP8 权重量化声明；dummy 学习使用 BF16 随机权重，不继承该声明。
_hf_overrides = {
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_topk": 2048,
    "quantization_config": None,
}

config = ServeConfig(
    model="deepseek-ai/DeepSeek-V3.2",
    device_count=1,
    serve_command=[
        "vllm",
        "serve",
        "{model}",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(_hf_overrides),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8_ds_mla",
        "--attention-config",
        json.dumps({
            "backend": "FLASHINFER_MLA_SPARSE_SM120",
            # 显式 HiSparse 配置会自动选择 Model Runner V2，无需改运行时环境变量。
            # 2048 个选中行之外保留一份 LRU 余量，并容纳最新 token，按 64 行对齐。
            "hisparse_config": {"device_buffer_size": 4096},
        }),
        "--kv-transfer-config",
        json.dumps({
            "kv_connector": "HiSparseConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"host_pool_gib": 1},
        }),
        "--tensor-parallel-size",
        "1",
        "--block-size",
        "64",
        # 上下文必须超过 2048 才能观察 top-k 选择；8192 也超过热缓冲容量。
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.8",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--scheduler-reserve-full-isl",
        # 先走 eager、无 speculative 的最小路径，避免把图捕获和 MTP 混入学习。
        "--enforce-eager",
    ],
)
