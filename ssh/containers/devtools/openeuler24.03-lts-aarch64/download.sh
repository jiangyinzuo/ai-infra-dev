#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"


command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "error: sha256sum is required" >&2; exit 1; }
command -v wget >/dev/null || { echo "error: wget is required" >&2; exit 1; }

download_if_changed() {
    local file=$1 url=$2 expected=$3 actual temp

    if [[ -f $file ]]; then
        actual=$(sha256sum "$file" | cut -d ' ' -f1)
        if [[ $actual == "$expected" ]]; then
            printf 'skip %s (sha256 matches)\n' "$file"
            return
        fi
        printf 'replace %s (sha256 mismatch)\n' "$file"
    else
        printf 'download %s\n' "$file"
    fi

    temp=$(mktemp "${file}.tmp.XXXXXX")
    trap 'rm -f -- "$temp"' RETURN
    wget -O "$temp" "$url"
    actual=$(sha256sum "$temp" | cut -d ' ' -f1)
    if [[ $actual != "$expected" ]]; then
        echo "error: sha256 mismatch for $file" >&2
        echo "       expected: $expected" >&2
        echo "       actual:   $actual" >&2
        return 1
    fi
    mv -f -- "$temp" "$file"
    trap - RETURN
}

github_asset_sha256() {
    local api_url=$1 asset=$2
    curl -fsSL "$api_url" | jq -er --arg asset "$asset" \
        '.assets[] | select(.name == $asset) | .digest | sub("^sha256:"; "")'
}

node_sha256=$(curl -fsSL https://nodejs.org/dist/v24.21.0/SHASUMS256.txt \
    | awk '$2 == "node-v24.21.0-linux-arm64.tar.xz" { print $1 }')
[[ $node_sha256 =~ ^[[:xdigit:]]{64}$ ]] || {
    echo "error: cannot obtain Node.js checksum" >&2
    exit 1
}

nvim_sha256=$(github_asset_sha256 \
    https://api.github.com/repos/neovim/neovim/releases/tags/nightly \
    nvim-linux-arm64.tar.gz)
omp_sha256=$(github_asset_sha256 \
    https://api.github.com/repos/can1357/oh-my-pi/releases/tags/v18.10.3 \
    omp-linux-arm64)
codex_sha256=$(github_asset_sha256 \
    https://api.github.com/repos/openai/codex/releases/tags/rust-v0.162.1 \
    codex-package-aarch64-unknown-linux-musl.tar.gz)

download_if_changed nvim-linux-arm64.tar.gz \
    https://github.com/neovim/neovim/releases/download/nightly/nvim-linux-arm64.tar.gz \
    "$nvim_sha256"
download_if_changed node-v24.21.0-linux-arm64.tar.xz \
    https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-arm64.tar.xz \
    "$node_sha256"
download_if_changed omp \
    https://github.com/can1357/oh-my-pi/releases/download/v18.10.3/omp-linux-arm64 \
    "$omp_sha256"
download_if_changed codex-package-aarch64-unknown-linux-musl.tar.gz \
    https://github.com/openai/codex/releases/download/rust-v0.162.1/codex-package-aarch64-unknown-linux-musl.tar.gz \
    "$codex_sha256"

chmod +x omp
