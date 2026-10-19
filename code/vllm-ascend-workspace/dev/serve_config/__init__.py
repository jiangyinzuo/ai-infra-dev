"""User-authored serve configuration; no implicit vLLM parameter defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServeConfig:
    device_count: int
    serve_command: list[str]
