#!/usr/bin/env bash
set -euo pipefail
: "${USG_IP:?error: USG_IP is not set}"
help=$(nc --help 2>&1 || true)
case $help in
    *Ncat*) exec nc --proxy-type http --proxy "$USG_IP:3128" "$@" ;;
    *) exec nc -X connect -x "$USG_IP:3128" "$@" ;;
esac
