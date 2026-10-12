#!/usr/bin/env bash
set -euo pipefail

# USAGE: bash ssh/install-ssh-keys-wsl.sh [--dry-run]；不依赖调用目录。
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/install-ssh-keys.py" \
  --group xiaoqiaoling --timeout 120 "$@"
