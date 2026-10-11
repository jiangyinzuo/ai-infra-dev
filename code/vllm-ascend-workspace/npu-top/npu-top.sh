#!/usr/bin/env bash

# 按脚本所在目录定位文件，不依赖调用时的工作目录。
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1

export ANSIBLE_CALLBACK_PLUGINS="$script_dir/callback_plugins${ANSIBLE_CALLBACK_PLUGINS:+:$ANSIBLE_CALLBACK_PLUGINS}"
export ANSIBLE_STDOUT_CALLBACK=npu_top

exec ansible-playbook -i "$script_dir/npu-hosts.ini" "$script_dir/npu-smi.yml" "$@"
