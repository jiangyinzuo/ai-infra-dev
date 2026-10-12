#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
args=()
for arg in "$@"; do
  if [[ "$arg" == --containers ]]; then
    args+=(-e npu_containers=true)
  else
    args+=("$arg")
  fi
done
export ANSIBLE_CALLBACK_PLUGINS="$script_dir/callback_plugins${ANSIBLE_CALLBACK_PLUGINS:+:$ANSIBLE_CALLBACK_PLUGINS}"
export ANSIBLE_STDOUT_CALLBACK=npu_top
exec ansible-playbook "${args[@]}"
