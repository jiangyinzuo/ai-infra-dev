# Source this file from ~/.bashrc. Complete Windows SSH Host aliases, not DNS.
_ssh_usg_complete() {
    local current=${COMP_WORDS[COMP_CWORD]} prefix= host line alias profile
    local config=${_ssh_usg_config:-} words=()
    COMPREPLY=()
    [[ $current != -* ]] || return 0
    if [[ $current == *@* ]]; then
        prefix=${current%@*}@
        current=${current##*@}
    fi
    if [[ -z $config ]]; then
        profile=$(cmd.exe /d /c 'chcp 65001 >nul & echo %USERPROFILE%' 2>/dev/null) || return 0
        profile=${profile//$'\r'/}
        config=$(wslpath -u "$profile\\.ssh\\config" 2>/dev/null) || return 0
        _ssh_usg_config=$config
    fi
    [[ -r $config ]] || return 0
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
    return 0
}
complete -F _ssh_usg_complete ssh-usg0 ssh-usg1
