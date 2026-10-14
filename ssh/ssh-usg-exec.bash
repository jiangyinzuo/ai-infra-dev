#!/usr/bin/env bash
# Sourced by USG wrappers with the interface address and proxy mode first.
ssh_usg_exec() {
    local ip=$1 proxy=$2 option quoted_ip command environment
    shift 2
    local -a connection=()
    local subsystem=0 no_command=0 terminal=0
    while (($#)); do
        option=$1
        case $option in
            --) connection+=("$1"); shift; break ;;
            -[BbcDEeFIiJLlmOopQRSWw])
                connection+=("$1")
                shift
                if (($# == 0)); then
                    exec /mnt/c/Windows/System32/OpenSSH/ssh.exe -b "$ip" "${connection[@]}"
                fi
                connection+=("$1")
                shift
                ;;
            -*)
                # Flags may be combined; options with attached values are opaque.
                if [[ $option != -[BbcDEeFIiJLlmOopQRSWw]* ]]; then
                    [[ $option != *s* ]] || subsystem=1
                    [[ $option != *N* ]] || no_command=1
                    [[ $option != *[Tt]* ]] || terminal=1
                fi
                connection+=("$1")
                shift
                ;;
            *) break ;;
        esac
    done
    if (($# == 0 || subsystem || no_command)); then
        exec /mnt/c/Windows/System32/OpenSSH/ssh.exe -b "$ip" "${connection[@]}" "$@"
    fi
    connection+=("$1")
    shift
    # POSIX-shell quoting, including values containing a single quote.
    quoted_ip="'${ip//\'/\'\\\'\'}'"
    environment="export USG_IP=$quoted_ip; "
    if ((proxy)); then
        environment+='export http_proxy="http://$USG_IP:3128" https_proxy="http://$USG_IP:3128" all_proxy="http://$USG_IP:3128" HTTP_PROXY="http://$USG_IP:3128" HTTPS_PROXY="http://$USG_IP:3128" ALL_PROXY="http://$USG_IP:3128"; '
    fi
    if (($#)); then
        # OpenSSH itself joins remote command arguments with spaces.
        command="${environment}$*"
    else
        command="${environment}exec \"\${SHELL:-/bin/sh}\" -l"
        # Supplying a command disables SSH's automatic login TTY allocation.
        if ((terminal == 0)) && [[ -t 0 ]]; then
            connection=(-t "${connection[@]}")
        fi
    fi
    exec /mnt/c/Windows/System32/OpenSSH/ssh.exe -b "$ip" "${connection[@]}" "$command"
}
