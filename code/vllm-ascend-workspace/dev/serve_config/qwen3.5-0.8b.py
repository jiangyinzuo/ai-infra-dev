"""Qwen3.5-0.8B single-device configuration, imported by dev serve."""

from serve_config import ServeConfig

config = ServeConfig(
    device_count=1,
    serve_command=["vllm", "serve", "{model}", "--tensor-parallel-size", "1"],
)
