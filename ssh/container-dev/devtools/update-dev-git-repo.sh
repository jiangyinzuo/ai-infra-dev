#!/usr/bin/env bash
# 服务器需要 Git、OpenSSH 和 nc
set -euo pipefail

# Usage: update-dev-git-repo.sh [--host USER@HOST] [--debug]
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote_target=
debug_args=()
while (($#)); do
    case $1 in
        --host)
            (($# >= 2)) || { echo 'error: --host requires USER@HOST' >&2; exit 2; }
            remote_target=$2
            [[ -n $remote_target ]] || { echo 'error: --host requires USER@HOST' >&2; exit 2; }
            shift 2
            ;;
        --debug)
            debug_args=(--debug)
            shift
            ;;
        -h|--help)
            echo 'Usage: update-dev-git-repo.sh [--host USER@HOST] [--debug] (host required on WSL)'
            exit 0
            ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
source "$script_dir/remote-exec.sh"
remote_run_if_wsl required -proxy "$remote_target" update-dev-git-repo.sh "${debug_args[@]}"

command -v git >/dev/null || { echo 'error: git is required' >&2; exit 1; }
command -v ssh >/dev/null || { echo 'error: ssh is required' >&2; exit 1; }
command -v nc >/dev/null || { echo 'error: nc (Ncat or OpenBSD netcat) is required' >&2; exit 1; }
: "${USG_IP:?error: 未设置USG_IP环境变量，请检查 SSH wrapper}"
proxy_address="$USG_IP:3128"

# 自动识别 Nmap Ncat与 OpenBSD netcat
nc_help=$(nc --help 2>&1 || true)
if [[ $nc_help == *Ncat* ]]; then
    proxy_command="nc --proxy-type http --proxy $proxy_address %h %p"
else
    proxy_command="nc -X connect -x $proxy_address %h %p"
fi

# Git SSH 不读取 http_proxy；仅本次 Git 命令使用 HTTP CONNECT 代理。
# 用 ssh.github.com:443 绕过 22 端口拦截。
# 首次自动信任（不核验官方指纹、不询问 yes），已有密钥变化则拒绝；不复制密钥。
git_ssh_command="ssh -o StrictHostKeyChecking=accept-new -o Hostname=ssh.github.com -p 443 -o HostKeyAlias=github.com -o 'ProxyCommand=$proxy_command'"
if ((${#debug_args[@]})); then
    git_ssh_command="${git_ssh_command/ssh /ssh -vvv }"
    git_ssh_command="${git_ssh_command/nc /nc -v }"
    export GIT_TRACE=1
fi
printf 'HTTP CONNECT proxy=%s; SSH endpoint=ssh.github.com:443\n' "$proxy_address" >&2
for repo in ai-infra-dev vimrc; do
    destination="$HOME/$repo"
    if [[ -e $destination || -L $destination ]]; then
        [[ -d $destination/.git || -f $destination/.git ]] || {
            echo "error: $destination exists but is not a Git repository" >&2
            exit 1
        }
        git_args=(-C "$destination" pull --ff-only)
    else
        git_args=(clone "git@github.com:jiangyinzuo/$repo.git" "$destination")
    fi
    printf 'GIT_SSH_COMMAND=%q git' "$git_ssh_command"
    printf ' %q' "${git_args[@]}"
    printf '\n'
    GIT_SSH_COMMAND="$git_ssh_command" git "${git_args[@]}"
done
