#!/usr/bin/env bash
set -euo pipefail

# 在 WSL 中运行；需设置 USG1_IP，并提前安装 Windows 公钥以便免密登录。
# 动态查询 Windows SSH config 中 lianqiuhu 开头的具体 Host，复用 WSL 的汇总界面。
# USAGE: ./npu-top-usg1.sh [-f 20] [--limit lianqiuhu-950pr-193]
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${USG1_IP:?Error: USG1_IP is not set}"
ssh_program=$(command -v ssh-usg1)
windows_profile=$(cd /mnt/c && cmd.exe /d /c 'chcp 65001 >nul & echo %USERPROFILE%')
windows_profile=${windows_profile//$'\r'/}
config="$(wslpath -u "$windows_profile")/.ssh/config"
windows_config=$(wslpath -w "$config")

hosts_dir=$(mktemp -d)
trap 'rm -rf -- "$hosts_dir"' EXIT
python3 "$script_dir/../ssh_hosts.py" \
  --config "$config" --ssh-config "$windows_config" --ssh "$ssh_program" \
  --host-glob 'lianqiuhu*' --group npu_servers > "$hosts_dir/hosts.json"

bash "$script_dir/npu-top-run.sh" -i "$hosts_dir/hosts.json" "$script_dir/npu-smi.yml" "$@"
