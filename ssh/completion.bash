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
_ssh2_complete() {
    _ssh_usg_complete "$HOME/.ssh/config"
    local current=${COMP_WORDS[COMP_CWORD]}
    if [[ $current != *@* && jiangyinzuo@ == "$current"* ]]; then
        COMPREPLY+=(jiangyinzuo@)
        compopt -o nospace 2>/dev/null || true
    fi
}
complete -F _ssh2_complete ssh2


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
    local group command=${_cli_completion_command:-${COMP_WORDS[0]}}
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
            done < <(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from ssh_common.config import HOST_GROUPS; print("\n".join(HOST_GROUPS))' "$_ssh_completion_dir" 2>/dev/null)
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
_devtools_cli_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} previous=${COMP_WORDS[COMP_CWORD-1]}
    local start=${_cli_completion_start:-1} group= action= word i
    local -a options=(-h --help)
    COMPREPLY=()
    for ((i=start; i<COMP_CWORD; i++)); do
        word=${COMP_WORDS[i]}
        case $word in
            --no-hints|-h|--help) continue ;;
            -*) break ;;
        esac
        if [[ -z $group ]]; then group=$word; else action=$word; break; fi
    done
    if [[ -z $group ]]; then
        _complete_values -h --help --no-hints git-repo binaries containers
        return
    fi
    case $group in
        binaries|containers|git-repo)
            if [[ -z $action ]]; then
                case $group in
                    binaries) _complete_values -h --help download extract sync sync-all ;;
                    containers) _complete_values -h --help create delete download ls ;;
                    git-repo) _complete_values -h --help update ;;
                esac
                return
            fi ;;
        *) return ;;
    esac
    options+=(--host)
    case $group:$action in
        git-repo:update) options+=(--debug) ;;
        binaries:download) options+=(--lock --output --repos --jobs) ;;
        binaries:extract) options+=(--output --jobs) ;;
        binaries:sync) options+=(--apply --dry-run) ;;
        binaries:sync-all) options+=(--jobs) ;;
        containers:create) options+=(--image --name --no-bindfs --share --test-home) ;;
        containers:delete) options+=(--name --keep-bindfs --remove-bindfs --cleanup-bindfs --yes) ;;
        containers:download) options+=(--npu-platform --date --version) ;;
        containers:ls) options+=(--json) ;;
    esac
    case $previous in
        --host)
            _ssh_usg_complete "$HOME/.ssh/config"
            [[ localhost != "$current"* || " ${COMPREPLY[*]} " == *' localhost '* ]] || COMPREPLY+=(localhost)
            ;;
        --npu-platform) _complete_values a3 ;;
        --lock|--output|--repos|--share|--test-home)
            mapfile -t COMPREPLY < <(compgen -f -- "$current")
            compopt -o filenames 2>/dev/null || true
            ;;
        --jobs|--image|--name|--date|--version) return ;;
        *) _complete_values "${options[@]}" ;;
    esac
}

_cli_complete() {
    local command=${_cli_completion_command:-${COMP_WORDS[0]}} directory
    case $command in
        */*) directory=${command%/*} ;;
        *) directory=$PWD ;;
    esac
    [[ $directory != . ]] || directory=$PWD
    case ${directory##*/} in
        devtools) _devtools_cli_complete ;;
        createuser|npu-top) _profile_cli_complete ;;
        *) COMPREPLY=() ;;
    esac
}

_cli_python_complete() {
    local _cli_completion_command=${COMP_WORDS[1]:-} _cli_completion_start=2
    local directory=${_cli_completion_command%/*}
    [[ $_cli_completion_command == */* && $directory != . ]] || directory=$PWD
    if ((COMP_CWORD > 1)) && [[ ${_cli_completion_command##*/} == cli.py ]]; then
        case ${directory##*/} in
            devtools|createuser|npu-top) _cli_complete; return ;;
        esac
    fi
    if [[ -n ${_cli_python_original_function:-} ]]; then
        "$_cli_python_original_function" "$@"
    elif ((${#_cli_python_original_args[@]})); then
        mapfile -t COMPREPLY < <(compgen "${_cli_python_original_args[@]}" -- "${COMP_WORDS[COMP_CWORD]}")
    else
        mapfile -t COMPREPLY < <(compgen -f -- "${COMP_WORDS[COMP_CWORD]}")
        compopt -o filenames 2>/dev/null || true
    fi
}

complete -F _cli_complete cli.py
for _completion_cli in npu-top createuser container-dev/devtools; do
    complete -F _cli_complete "./$_completion_cli/cli.py" "$_completion_cli/cli.py" "$_ssh_completion_dir/$_completion_cli/cli.py"
done
complete -F _cli_complete ./devtools/cli.py devtools/cli.py "$HOME/ai-infra-dev/ssh/container-dev/devtools/cli.py"
# 保留已加载的 Python 补全函数及其选项；其他脚本仍交给原补全处理。
_cli_python_spec=$(complete -p python3 2>/dev/null) || _cli_python_spec=
if [[ $_cli_python_spec != *' -F _cli_python_complete '* ]]; then
    _cli_python_original_function=
    _cli_python_original_args=()
    if [[ $_cli_python_spec =~ ' -F '([^[:space:]]+) ]]; then
        _cli_python_original_function=${BASH_REMATCH[1]}
        eval "${_cli_python_spec/ -F $_cli_python_original_function / -F _cli_python_complete }"
    elif [[ -z $_cli_python_spec ]]; then
        complete -o default -F _cli_python_complete python3
    else
        _cli_capture_python_spec() {
            _cli_python_original_args=("${@:1:$#-1}")
        }
        eval "${_cli_python_spec/#complete /_cli_capture_python_spec }"
        unset -f _cli_capture_python_spec
        complete -o default -F _cli_python_complete python3
    fi
fi
unset _completion_cli _cli_python_spec
