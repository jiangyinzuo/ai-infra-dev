# Restore the SSH session proxy after system/user shell initialization.
if [ -n "${DEVTOOLS_HTTP_PROXY:-}" ]; then
    export http_proxy="$DEVTOOLS_HTTP_PROXY"
    export https_proxy="$DEVTOOLS_HTTP_PROXY"
    export HTTP_PROXY="$DEVTOOLS_HTTP_PROXY"
    export HTTPS_PROXY="$DEVTOOLS_HTTP_PROXY"
fi
