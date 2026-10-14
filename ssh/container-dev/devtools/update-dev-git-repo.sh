#!/usr/bin/env bash
set -euo pipefail

# Usage: update-dev-git-repo.sh [--host USER@HOST]
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote_target=
while (($#)); do
    case $1 in
        --host)
            (($# >= 2)) || { echo 'error: --host requires USER@HOST' >&2; exit 2; }
            remote_target=$2
            [[ -n $remote_target ]] || { echo 'error: --host requires USER@HOST' >&2; exit 2; }
            shift 2
            ;;
        -h|--help)
            echo 'Usage: update-dev-git-repo.sh [--host USER@HOST] (required on WSL)'
            exit 0
            ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
source "$script_dir/remote-exec.sh"
remote_run_if_wsl required -proxy "$remote_target" update-dev-git-repo.sh

command -v git >/dev/null || { echo 'error: git is required' >&2; exit 1; }
command -v ssh >/dev/null || { echo 'error: ssh is required' >&2; exit 1; }
command -v nc >/dev/null || { echo 'error: nc (Ncat or OpenBSD netcat) is required' >&2; exit 1; }
: "${USG_IP:?error: USG_IP must be provided by the SSH wrapper}"
proxy_address="$USG_IP:3128"
nc_help=$(nc --help 2>&1 || true)
if [[ $nc_help == *Ncat* ]]; then
    proxy_command="nc --proxy-type http --proxy $proxy_address %h %p"
else
    proxy_command="nc -X connect -x $proxy_address %h %p"
fi
# Git's SSH transport ignores http_proxy; explicitly tunnel SSH over HTTP CONNECT.
# This environment assignment is scoped to each Git invocation below.
# GitHub's SSH endpoint on 443 avoids the gateway block on github.com:22.
# Keep github.com as the host-key identity; do not disable host verification.
git_ssh_command="ssh -o Hostname=ssh.github.com -p 443 -o HostKeyAlias=github.com -o 'ProxyCommand=$proxy_command'"
for repo in ai-infra-dev vimrc; do
    destination="$HOME/$repo"
    if [[ -e $destination || -L $destination ]]; then
        [[ -d $destination/.git || -f $destination/.git ]] || {
            echo "error: $destination exists but is not a Git repository" >&2
            exit 1
        }
        printf 'pull %s\n' "$destination"
        GIT_SSH_COMMAND="$git_ssh_command" git -C "$destination" pull --ff-only
    else
        printf 'clone %s\n' "$destination"
        GIT_SSH_COMMAND="$git_ssh_command" git clone "git@github.com:jiangyinzuo/$repo.git" "$destination"
    fi
done
