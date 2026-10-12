#!/usr/bin/env bash
set -euo pipefail

# 在 WSL 中运行；需设置 USG1_IP，并提前安装 Windows 公钥以便免密登录。
# 从 ssh/hosts.json 的 lianqiuhu 分组读取连接规则。
# USAGE: ./npu-top-usg1.sh [-f 20] [--limit lianqiuhu-950pr-193]
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${USG1_IP:?Error: USG1_IP is not set}"

hosts_dir=$(mktemp -d)
trap 'rm -rf -- "$hosts_dir"' EXIT
python3 "$script_dir/../ssh_hosts.py" \
  --group lianqiuhu --inventory-group npu_servers > "$hosts_dir/hosts.json"

bash "$script_dir/npu-top-run.sh" -i "$hosts_dir/hosts.json" "$script_dir/npu-smi.yml" "$@"
