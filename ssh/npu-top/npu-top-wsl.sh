#!/usr/bin/env bash
set -euo pipefail

# 从 ssh/hosts.json 的 xiaoqiaoling 分组读取连接规则，不依赖调用目录。
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1

hosts_dir=$(mktemp -d)
trap 'rm -rf -- "$hosts_dir"' EXIT
python3 "$script_dir/../ssh_hosts.py" \
  --group xiaoqiaoling --inventory-group npu_servers > "$hosts_dir/hosts.json"

bash "$script_dir/npu-top-run.sh" -i "$hosts_dir/hosts.json" "$script_dir/npu-smi.yml" "$@"
