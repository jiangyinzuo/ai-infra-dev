#!/usr/bin/env bash
set -euo pipefail

# 从 WSL ~/.ssh/config 筛选 xiaoqiaoling*，不依赖调用目录。
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1

hosts_dir=$(mktemp -d)
trap 'rm -rf -- "$hosts_dir"' EXIT
python3 "$script_dir/../../../scripts/ssh_hosts.py" \
  --config "$HOME/.ssh/config" --ssh-config "$HOME/.ssh/config" --ssh /usr/bin/ssh \
  --host-glob 'xiaoqiaoling*' --group npu_servers > "$hosts_dir/hosts.json"

export ANSIBLE_CALLBACK_PLUGINS="$script_dir/callback_plugins${ANSIBLE_CALLBACK_PLUGINS:+:$ANSIBLE_CALLBACK_PLUGINS}"
export ANSIBLE_STDOUT_CALLBACK=npu_top

ansible-playbook -i "$hosts_dir/hosts.json" "$script_dir/npu-smi.yml" "$@"
