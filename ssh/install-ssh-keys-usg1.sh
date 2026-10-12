#!/usr/bin/env bash
set -euo pipefail
# Select the lianqiuhu connection profile; extra arguments filter or preview it.
python3 "$(dirname -- "${BASH_SOURCE[0]}")/install-ssh-keys.py" \
  --group lianqiuhu --timeout 120 "$@"
