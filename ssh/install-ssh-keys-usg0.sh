#!/usr/bin/env bash
set -euo pipefail
# Select the suzhou connection profile; extra arguments filter or preview it.
python3 "$(dirname -- "${BASH_SOURCE[0]}")/install-ssh-keys.py" \
  --group suzhou --timeout 120 "$@"
