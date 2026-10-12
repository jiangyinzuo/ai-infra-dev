#!/usr/bin/env bash
set -euo pipefail

# USAGE: bash ssh/install-ssh-keys-wsl.sh [--dry-run]；不依赖调用目录。
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/install-ssh-keys.py" \
  --config ~/.ssh/config \
  --ssh /usr/bin/ssh \
  --public-key ~/.ssh/id_ed25519.pub \
  --host-glob 'xiaoqiaoling*' \
  --timeout 120 "$@"
