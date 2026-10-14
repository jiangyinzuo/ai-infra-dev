#!/usr/bin/env bash
set -euo pipefail

# Usage: extract-nvim-plugin.sh [--jobs N] [--host USER@HOST]
# On WSL, require an SSH target and run the already-synced ~/devtools script.
# Run up to N tar processes concurrently (default: 8).
# Read sibling nvim-plugins/*.tar.gz, stripping the GitHub archive's root directory.
# Install plugins into ~/plugged/; lazy.nvim uses ~/.local/share/nvim/lazy/lazy.nvim.
# Overwrite matching files; retain archives and existing destination-only files.
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
jobs=8
remote_target=
while (($#)); do
    case $1 in
        --jobs)
            (($# >= 2)) || { echo 'error: --jobs requires a positive integer' >&2; exit 2; }
            jobs=$2
            shift 2
            ;;
        --host)
            (($# >= 2)) || { echo 'error: --host requires USER@HOST' >&2; exit 2; }
            remote_target=$2
            shift 2
            ;;
        -h|--help)
            echo 'Usage: extract-nvim-plugin.sh [--jobs N] [--host USER@HOST] (default jobs: 8)'
            exit 0
            ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo 'error: --jobs requires a positive integer' >&2; exit 2; }

source "$script_dir/remote-exec.sh"
remote_extract_if_wsl "$remote_target" extract-nvim-plugin.sh --jobs "$jobs"
command -v tar >/dev/null || { echo 'error: tar is required' >&2; exit 1; }
shopt -s nullglob
archives=("$script_dir"/nvim-plugins/*.tar.gz)

if ((${#archives[@]} == 0)); then
    echo 'No plugin archives found.'
    exit 0
fi

extract_plugin() (
    archive=$1
    name=${archive##*/}
    name=${name%.tar.gz}
    if [[ $name == lazy.nvim ]]; then
        destination="$HOME/.local/share/nvim/lazy/lazy.nvim"
    else
        destination="$HOME/plugged/$name"
    fi
    mkdir -p -- "$destination"
    printf 'extract %s -> %s\n' "$archive" "$destination"
    tar -xf "$archive" --strip-components=1 -C "$destination"
)

pids=()
failed=0
for archive in "${archives[@]}"; do
    extract_plugin "$archive" &
    pids+=("$!")
    if ((${#pids[@]} >= jobs)); then
        wait "${pids[0]}" || failed=1
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if ((failed)); then
    echo 'error: one or more plugin archives failed to extract' >&2
    exit 1
fi
