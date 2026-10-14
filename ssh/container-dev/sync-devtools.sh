#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ssh_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
devtools_dir="$script_dir/devtools"
remote_user=jiangyinzuo
remote_dir='~/devtools'
apply=false

usage() {
    cat <<'EOF'
同步`devtools/`目录（Docker镜像除外）

Usage: sync-devtools.sh [options] PLATFORM HOST [HOST ...]

Example: sync-devtools.sh --apply openeuler24.03-lts-aarch64 suzhou-a3-130

Options:
  -u, --user USER       Remote SSH user (default: jiangyinzuo)
  -d, --remote-dir DIR  Remote destination directory
                        (default: ~/devtools)
      --apply           Perform synchronization; default is --dry-run
                        Creates missing remote parent directories before transfer
      --dry-run         Show planned changes without synchronizing (default)
  -h, --help            Show this help

SSH wrapper 从 ssh/hosts.json 选择:
  suzhou*       ssh-usg0
  lianqiuhu*    ssh-usg1
  xiaoqiaoling* ssh
EOF
}

source "$devtools_dir/remote-exec.sh"

targets=()
while (($#)); do
    case $1 in
        -u|--user)
            (($# >= 2)) || { echo "error: missing argument for $1" >&2; exit 2; }
            remote_user=$2
            shift 2
            ;;
        -d|--remote-dir)
            (($# >= 2)) || { echo "error: missing argument for $1" >&2; exit 2; }
            remote_dir=$2
            shift 2
            ;;
        --apply)
            apply=true
            shift
            ;;
        --dry-run)
            apply=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            targets+=("$@")
            break
            ;;
        -* )
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
        *)
            targets+=("$1")
            shift
            ;;
    esac
done

if ((${#targets[@]} < 2)); then
    echo "error: a platform and at least one remote host are required" >&2
    usage >&2
    exit 2
fi

platform=${targets[0]}
hosts=("${targets[@]:1}")
[[ $platform =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && $platform != nvim-plugins && $platform != images ]] || {
    echo "error: invalid platform '$platform'" >&2
    exit 2
}
source_dir="$devtools_dir/$platform"
[[ -d $source_dir ]] || { echo "error: source directory not found: $source_dir" >&2; exit 1; }
# Keep an empty common plugin directory syncable before the first download.
mkdir -p -- "$devtools_dir/nvim-plugins"
shopt -s nullglob
sources=("$source_dir" "$devtools_dir/nvim-plugins" "$devtools_dir"/*.py "$devtools_dir"/*.sh)
sources+=("$devtools_dir/images")
command -v rsync >/dev/null || { echo "error: rsync is required" >&2; exit 1; }

# Quote for the remote POSIX shell, preserving spaces and literal metacharacters.
shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

for host in "${hosts[@]}"; do
    ssh_command=$(ssh_command_for_host "$host")
    if $apply; then
        mode=sync
        rsync_args=(-av --info=progress2 -e "$ssh_command")
        # Expand only a leading ~/ against the remote user's home directory.
        if [[ $remote_dir == '~/'* ]]; then
            mkdir_path="\"\$HOME\"/$(shell_quote "${remote_dir:2}")"
        elif [[ $remote_dir == '~' ]]; then
            mkdir_path='"$HOME"'
        else
            mkdir_path=$(shell_quote "$remote_dir")
        fi
        rsync_args+=(--rsync-path "mkdir -p -- $mkdir_path && exec rsync")
    else
        mode=dry-run
        rsync_args=(-av --info=progress2 --dry-run -e "$ssh_command")
    fi
    # Image trees contain large archives; sync only directories and scripts.
    rsync_args+=(--include='/images/**/' --include='/images/**.sh' --include='/images/**.py' --exclude='/images/**')
    # Reset the green status color before rsync prints its own output.
    printf '\033[32m%s %s + common -> %s@%s:%s\033[0m\n' "$mode" "$source_dir" "$remote_user" "$host" "$remote_dir"
    rsync "${rsync_args[@]}" "${sources[@]}" "$remote_user@$host:$remote_dir/"
done
