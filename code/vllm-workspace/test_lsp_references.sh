#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/vllm"
results=$(mktemp -d)
trap 'rm -rf -- "$results"' EXIT

target_symbol=OutputProcessor
mapfile -t target_files < <(
  rg -l --glob '*.py' "^class ${target_symbol}([(: ]|$)" vllm
)
if (( ${#target_files[@]} != 1 )); then
  printf 'FAIL: expected one %s declaration, found %s\n' \
    "$target_symbol" "${#target_files[@]}" >&2
  exit 1
fi
target_file=${target_files[0]}
mapfile -t target_lines < <(
  rg -n --no-filename "^class ${target_symbol}([(: ]|$)" "$target_file" |
    cut -d: -f1
)
if (( ${#target_lines[@]} != 1 )); then
  printf 'FAIL: expected one %s declaration in %s, found %s\n' \
    "$target_symbol" "$target_file" "${#target_lines[@]}" >&2
  exit 1
fi
target_line=${target_lines[0]}

for server in pyrefly ty; do
  case "$server" in
    pyrefly) subcommand=lsp ;;
    ty) subcommand=server ;;
  esac
  if ! uv run --no-sync --offline --python .venv/bin/python \
    python ../../lsp_probe.py \
    --server "$server" --server-subcommand "$subcommand" \
    --file "$target_file" \
    --line "$target_line" --symbol "$target_symbol" \
    --log "$results/$server.log" \
    > "$results/$server.json" 2> "$results/$server.err"; then
    printf 'FAIL: %s probe failed\n' "$server" >&2
    cat "$results/$server.err" >&2
    cat "$results/$server.log" >&2
    exit 1
  fi
done

pyrefly_count=$(jq -er '.result_count' "$results/pyrefly.json")
ty_count=$(jq -er '.result_count' "$results/ty.json")
printf 'Pyrefly: %s references\nty: %s references\n' "$pyrefly_count" "$ty_count"
if [[ "$pyrefly_count" != "$ty_count" ]]; then
  printf 'FAIL: reference counts differ\n' >&2
  exit 1
fi
printf 'PASS: reference counts match (locations not compared)\n'
