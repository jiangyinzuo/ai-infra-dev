#!/usr/bin/env bash
set -euo pipefail
script_dir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
root=$(realpath -- "$script_dir/../../..")
exec "$root/code/vllm-workspace/vllm/.venv/bin/python" "$script_dir/run.py" "$@"
