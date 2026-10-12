#!/usr/bin/env bash
set -euo pipefail

# 在 WSL 中运行；需设置 USG0_IP，并提前安装 Windows 公钥以便免密登录。
# 动态查询 Windows SSH config 中 suzhou 开头的具体 Host，复用 WSL 的汇总界面。
# USAGE: ./npu-top-usg0.sh [-f 20] [--limit suzhou-a3-129]
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${USG0_IP:?Error: USG0_IP is not set}"
ssh_program=$(command -v ssh-usg0)
windows_profile=$(cd /mnt/c && cmd.exe /d /c 'chcp 65001 >nul & echo %USERPROFILE%')
windows_profile=${windows_profile//$'\r'/}
config="$(wslpath -u "$windows_profile")/.ssh/config"
windows_config=$(wslpath -w "$config")

hosts_dir=$(mktemp -d)
trap 'rm -rf -- "$hosts_dir"' EXIT
python3 "$script_dir/../../../scripts/ssh_hosts.py" \
  --config "$config" --ssh-config "$windows_config" --ssh "$ssh_program" \
  --host-glob 'suzhou*' --group npu_servers > "$hosts_dir/hosts.json"

bash "$script_dir/npu-top-run.sh" -i "$hosts_dir/hosts.json" "$script_dir/npu-smi.yml" "$@"
