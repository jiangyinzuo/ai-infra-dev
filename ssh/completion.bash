# 在 ~/.bashrc 中 source；同步工具合并本地和 Windows SSH Host，SSH 包装器仅用 Windows。
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
complete -F _ssh_usg_complete ssh-usg0 ssh-usg1 ssh-usg0-proxy ssh-usg1-proxy

_sync_devtools_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} previous=${COMP_WORDS[COMP_CWORD-1]}
    COMPREPLY=()
    case $previous in
        -u|--user|-d|--remote-dir) return 0 ;;
    esac
    case $current in
        -*)
            mapfile -t COMPREPLY < <(compgen -W '-h --help -u --user -d --remote-dir --apply --dry-run' -- "$current")
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
