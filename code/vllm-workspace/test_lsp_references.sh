#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/vllm"
results=$(mktemp -d)
trap 'rm -rf -- "$results"' EXIT

for server in pyrefly ty; do
  case "$server" in
    pyrefly) subcommand=lsp ;;
    ty) subcommand=server ;;
  esac
  uv run --no-sync --offline --python .venv/bin/python \
    python ../../lsp_probe.py \
    --server "$server" --server-subcommand "$subcommand" \
    --file vllm/v1/engine/output_processor.py \
    --line 551 --symbol OutputProcessor \
    > "$results/$server.json"
done

pyrefly_count=$(jq -er '.result_count' "$results/pyrefly.json")
ty_count=$(jq -er '.result_count' "$results/ty.json")
printf 'Pyrefly: %s references\nty: %s references\n' "$pyrefly_count" "$ty_count"
if [[ "$pyrefly_count" != "$ty_count" ]]; then
  printf 'FAIL: reference counts differ\n' >&2
  exit 1
fi
printf 'PASS: reference counts match (locations not compared)\n'
