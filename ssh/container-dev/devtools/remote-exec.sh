#!/usr/bin/env bash
# Sourced by sync/extract/update scripts. Remote scripts must already be synced
# under ~/devtools; dispatch never transfers scripts.
ssh_command_for_host() {
    local ssh_dir
    ssh_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
    case $1 in
        suzhou*) printf '%s/ssh-usg0%s' "$ssh_dir" "${2:-}" ;;
        lianqiuhu*) printf '%s/ssh-usg1%s' "$ssh_dir" "${2:-}" ;;
        *) command -v ssh ;;
    esac
}

remote_extract_if_wsl() {
    remote_run_if_wsl prompt '' "$@"
}

# Policy and wrapper suffix are internal parameters, not remote script arguments.
remote_run_if_wsl() {
    local policy=$1 suffix=$2 target=$3 relative_script=$4 host ssh_command command argument
    shift 4
    if [[ $(uname -r) != *[Mm]icrosoft* ]]; then
        [[ -z $target ]] || { echo 'error: --host is only supported on WSL' >&2; exit 2; }
        return
    fi
    if [[ -z $target ]]; then
        if [[ $policy == required ]]; then
            echo 'error: WSL requires --host USER@HOST' >&2
            exit 2
        fi
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
    ssh_command=$(ssh_command_for_host "$host" "$suffix")
    # Quote relative script and arguments for the remote POSIX shell.
    command="bash \"\$HOME\"/devtools/'${relative_script//\'/\'\\\'\'}'"
    for argument in "$@"; do
        command+=" '${argument//\'/\'\\\'\'}'"
    done
    printf 'run on %s using ~/devtools/%s\n' "$target" "$relative_script"
    exec "$ssh_command" "$target" "$command"
}
