#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/vllm"
uv run --no-sync --offline --python .venv/bin/python \
  python ../../lsp_probe.py \
  --file vllm/v1/engine/output_processor.py \
  --line 551 --symbol OutputProcessor \
  | uv run --no-sync --offline --python .venv/bin/python python -c '
import json
import sys

response = json.load(sys.stdin)
count = response["result_count"]
if count != 32:
    sys.exit(f"FAIL: expected 32 references, got {count}")
print(f"PASS: {count} references")
'
