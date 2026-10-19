#!/usr/bin/env bash
set -euo pipefail
if [[ -n ${DEVTOOLS_HTTP_PROXY:-} ]]; then
    proxy=$(python3 -c 'import sys; from urllib.parse import urlsplit; p=urlsplit(sys.argv[1]); assert p.scheme == "http" and p.hostname and not p.username and not p.password and p.path in ("", "/") and not p.query and not p.fragment, "invalid HTTP CONNECT proxy"; host="["+p.hostname+"]" if ":" in p.hostname else p.hostname; print(host+":"+str(p.port or 80))' "$DEVTOOLS_HTTP_PROXY")
else
    echo 'error: 请设置 DEVTOOLS_HTTP_PROXY' >&2
    exit 1
fi
help=$(nc --help 2>&1 || true)
case $help in
    *Ncat*) exec nc --proxy-type http --proxy "$proxy" "$@" ;;
    *) exec nc -X connect -x "$proxy" "$@" ;;
esac
