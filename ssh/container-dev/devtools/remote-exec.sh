#!/usr/bin/env bash
# Sourced by sync/extract scripts. WSL extraction runs the already-synced script
# under ~/devtools on a chosen server; it never transfers or extracts locally.
ssh_command_for_host() {
    local ssh_dir
    ssh_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
    case $1 in
        suzhou*) printf '%s/ssh-usg0' "$ssh_dir" ;;
        lianqiuhu*) printf '%s/ssh-usg1' "$ssh_dir" ;;
        *) command -v ssh ;;
    esac
}

remote_extract_if_wsl() {
    local target=$1 relative_script=$2 host ssh_command command argument
    shift 2
    if [[ $(uname -r) != *[Mm]icrosoft* ]]; then
        [[ -z $target ]] || { echo 'error: --host is only supported on WSL' >&2; exit 2; }
        return
    fi
    if [[ -z $target ]]; then
        read -r -p 'WSL: remote SSH target (USER@HOST; default user jiangyinzuo): ' target || {
            echo 'error: WSL requires a remote SSH target; use --host USER@HOST' >&2
            exit 2
        }
    fi
    [[ $target =~ ^([a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || {
        echo 'error: invalid SSH target' >&2
        exit 2
    }
    [[ $target == *@* ]] || target="jiangyinzuo@$target"
    host=${target##*@}
    ssh_command=$(ssh_command_for_host "$host")
    # Quote relative script and arguments for the remote POSIX shell.
    command="bash \"\$HOME\"/devtools/'${relative_script//\'/\'\\\'\'}'"
    for argument in "$@"; do
        command+=" '${argument//\'/\'\\\'\'}'"
    done
    printf 'extract on %s using ~/devtools/%s\n' "$target" "$relative_script"
    exec "$ssh_command" "$target" "$command"
}
