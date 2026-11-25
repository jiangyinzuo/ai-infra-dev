"""Qwen3-Next tiny random：观察 Full Attention / GDN 分组；无需远程权重和 tokenizer。"""
# ruff: noqa: N999

from pathlib import Path

from serve_config import ServeConfig

config = ServeConfig(
    model=str(Path(__file__).with_suffix("")),
    device_count=1,
    serve_command=[
        "vllm", "serve", "{model}",
        "--load-format", "dummy",
        "--skip-tokenizer-init",
        "--served-model-name", "tiny-qwen3-next-kv-groups",
        "--dtype", "bfloat16",
        "--kv-cache-dtype", "auto",
        "--seed", "42",
        "--tensor-parallel-size", "1",
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "1024",
        "--max-num-seqs", "1",
        # 固定小池，避免为学习用例按整卡空闲显存扩大 KV 分配。
        "--num-gpu-blocks-override", "128",
        "--disable-extensible-kv-cache",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enforce-eager",
        "--shutdown-timeout", "30",
    ],
)
