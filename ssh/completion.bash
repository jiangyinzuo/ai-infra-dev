# 在 ~/.bashrc 中 source；合并 WSL 和 Windows SSH Host，供 ssh2 和相关工具补全。
_ssh_usg_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} prefix= host line alias profile
    local config=${_ssh_usg_config:-} words=() configs=("$@")
    COMPREPLY=()
    [[ $current != -* ]] || return 0
    if [[ $current == *@* ]]; then
        prefix=${current%@*}@
        current=${current##*@}
    fi
    if [[ -z $config ]]; then
        profile=$(cmd.exe /d /c 'chcp 65001 >nul & echo %USERPROFILE%' 2>/dev/null) || profile=
        profile=${profile//$'\r'/}
        if [[ -n $profile ]]; then
            config=$(wslpath -u "$profile\\.ssh\\config" 2>/dev/null) || config=
        fi
        _ssh_usg_config=$config
    fi
    [[ -z $config ]] || configs+=("$config")
    for config in "${configs[@]}"; do
    [[ -r $config ]] || continue
    while IFS= read -r line || [[ -n $line ]]; do
        line=${line%$'\r'}
        line=${line%%#*}
        if [[ $line =~ ^[[:space:]]*[Hh][Oo][Ss][Tt]([[:space:]]*=[[:space:]]*|[[:space:]]+)(.*)$ ]]; then
            read -r -a words <<< "${BASH_REMATCH[2]}"
            for alias in "${words[@]}"; do
                alias=${alias#\"}; alias=${alias%\"}
                alias=${alias#\'}; alias=${alias%\'}
                [[ $alias =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.:-]*$ ]] || continue
                [[ $alias == "$current"* ]] || continue
                host=$prefix$alias
                [[ " ${COMPREPLY[*]} " == *" $host "* ]] || COMPREPLY+=("$host")
            done
        fi
    done < "$config"
    done
    return 0
}
_ssh2_complete() { _ssh_usg_complete "$HOME/.ssh/config"; }
complete -F _ssh2_complete ssh2

_sync_devtools_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} previous=${COMP_WORDS[COMP_CWORD-1]}
    COMPREPLY=()
    case $previous in
        -u|--user|-d|--remote-dir) return 0 ;;
    esac
    case $current in
        -*)
            _complete_values -h --help -u --user -d --remote-dir --apply --dry-run
            ;;
        *)
            _ssh_usg_complete "$HOME/.ssh/config"
            if [[ -z $current ]]; then
                COMPREPLY+=(-h --help -u --user -d --remote-dir --apply --dry-run)
            fi
            ;;
    esac
}
complete -F _sync_devtools_complete sync-devtools.py

_container_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} previous=${COMP_WORDS[COMP_CWORD-1]}
    COMPREPLY=()
    case $previous in
        --host)
            _ssh_usg_complete "$HOME/.ssh/config"
            [[ localhost != "$current"* ]] || COMPREPLY+=(localhost)
            return ;;
        --npu-platform) _complete_values a3; return ;;
        --date|--version|--image|--name) return ;;
    esac
    local options='-h --help --host --npu-platform'
    case ${COMP_WORDS[0]} in
        *download-container.py) options+=' --date --version' ;;
        *create-container.py) options+=' --image --name' ;;
    esac
    _complete_values $options
}
complete -F _container_complete download-container.py create-container.py

_update_dev_git_repo_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} previous=${COMP_WORDS[COMP_CWORD-1]}
    COMPREPLY=()
    case $previous in
        --host) _ssh_usg_complete "$HOME/.ssh/config" ;;
        *) _complete_values -h --help --host --debug ;;
    esac
}
complete -F _update_dev_git_repo_complete update-dev-git-repo.sh

_ssh_completion_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 候选值逐个匹配，保留路径中的空格。
_complete_values() {
    local value
    COMPREPLY=()
    for value in "$@"; do
        [[ $value != "${COMP_WORDS[COMP_CWORD]}"* ]] || COMPREPLY+=("$value")
    done
}

_profile_cli_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} previous=${COMP_WORDS[COMP_CWORD-1]}
    local profiles="$_ssh_completion_dir/hosts.json" group command=${COMP_WORDS[0]}
    local -a options=(-h --help --group --host-glob)
    # ./cli.py 按当前目录区分；带路径调用按入口目录区分。
    case $command in
        */*) command=${command%/*} ;;
        *) command=$PWD ;;
    esac
    [[ $command != . ]] || command=$PWD
    case ${command##*/} in
        createuser) options+=(--apply --check --workers) ;;
        npu-top) options+=(-i --inventory --containers -f --forks --limit --become) ;;
        *) COMPREPLY=(); return ;;
    esac
    COMPREPLY=()
    case $previous in
        --group)
            while IFS= read -r group; do
                [[ $group != "$current"* ]] || COMPREPLY+=("$group")
            done < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["groups"]))' "$profiles" 2>/dev/null)
            ;;
        --host-glob|--limit|-l) _ssh_usg_complete "$HOME/.ssh/config" ;;
        -i|--inventory)
            mapfile -t COMPREPLY < <(compgen -f -- "$current")
            compopt -o filenames
            ;;
        --workers|-f|--forks|-e|--extra-vars) return ;;
        *) _complete_values "${options[@]}" ;;
    esac
}
complete -F _profile_cli_complete cli.py
for _completion_cli in npu-top createuser; do
    complete -F _profile_cli_complete "./$_completion_cli/cli.py" "$_completion_cli/cli.py" "$_ssh_completion_dir/$_completion_cli/cli.py"
done
unset _completion_cli
