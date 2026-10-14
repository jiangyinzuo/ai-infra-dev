#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"

# Usage: download.sh [--jobs N]; each worker verifies SHA-256 before publishing.
jobs=8
while (($#)); do
    case $1 in
        --jobs)
            (($# >= 2)) || { echo 'error: --jobs requires a positive integer' >&2; exit 2; }
            jobs=$2
            shift 2
            ;;
        -h|--help)
            echo 'Usage: download.sh [--jobs N] (default: 8)'
            exit 0
            ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo 'error: --jobs requires a positive integer' >&2; exit 2; }
command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "error: sha256sum is required" >&2; exit 1; }
command -v wget >/dev/null || { echo "error: wget is required" >&2; exit 1; }

download_if_changed() (
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
    trap 'rm -f -- "$temp"' EXIT
    wget -O "$temp" "$url"
    actual=$(sha256sum "$temp" | cut -d ' ' -f1)
    if [[ $actual != "$expected" ]]; then
        echo "error: sha256 mismatch for $file" >&2
        echo "       expected: $expected" >&2
        echo "       actual:   $actual" >&2
        return 1
    fi
    mv -f -- "$temp" "$file"
    trap - EXIT
)

# Download a GitHub release asset; optionally override its local filename.
download_github_asset() {
    local repo=$1 tag=$2 asset=$3 file=${4:-$3} expected
    expected=$(curl -fsSL "https://api.github.com/repos/$repo/releases/tags/$tag" \
        | jq -er --arg asset "$asset" \
            '.assets[] | select(.name == $asset) | .digest | sub("^sha256:"; "")')
    download_if_changed "$file" \
        "https://github.com/$repo/releases/download/$tag/$asset" "$expected"
}

download_node() {
    local node_sha256
    node_sha256=$(curl -fsSL https://nodejs.org/dist/v24.21.0/SHASUMS256.txt \
        | awk '$2 == "node-v24.21.0-linux-arm64.tar.xz" { print $1 }')
    [[ $node_sha256 =~ ^[[:xdigit:]]{64}$ ]] || {
        echo "error: cannot obtain Node.js checksum" >&2
        exit 1
    }
    download_if_changed node-v24.21.0-linux-arm64.tar.xz \
        https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-arm64.tar.xz \
        "$node_sha256"
}

download_omp() {
    download_github_asset can1357/oh-my-pi v18.10.3 omp-linux-arm64 omp
    chmod +x omp
}

# Wait for explicit PIDs so every worker failure is observed, even if it exits early.
pids=()
failed=0
submit() {
    (set -e; "$@") &
    pids+=("$!")
    if ((${#pids[@]} >= jobs)); then
        wait "${pids[0]}" || failed=1
        pids=("${pids[@]:1}")
    fi
}

submit download_github_asset neovim/neovim nightly nvim-linux-arm64.tar.gz
submit download_node

submit download_omp
submit download_github_asset openai/codex rust-v0.162.1 codex-package-aarch64-unknown-linux-musl.tar.gz
submit download_github_asset BurntSushi/ripgrep 15.2.0 ripgrep-15.2.0-aarch64-unknown-linux-musl.tar.gz
submit download_github_asset sharkdp/fd v10.5.0 fd-v10.5.0-aarch64-unknown-linux-musl.tar.gz
submit download_github_asset dandavison/delta 0.20.1 delta-0.20.1-aarch64-unknown-linux-gnu.tar.gz
submit download_github_asset tree-sitter/tree-sitter v0.27.1 tree-sitter-linux-arm64.gz

for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if ((failed)); then
    echo 'error: one or more tool downloads failed' >&2
    exit 1
fi
