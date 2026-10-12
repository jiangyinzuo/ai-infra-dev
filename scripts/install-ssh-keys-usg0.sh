#!/usr/bin/env bash
set -euo pipefail
# USAGE: ./install-ssh-keys-usg0.sh --host-glob 'suzhou*' --host-glob '其他前缀*' [--dry-run]
# 默认 suzhou*；显式 --host-glob 覆盖默认值，可重复指定。
# NOTE: ssh-usg0需先设置 USG0_IP环境变量

host_args=(--host-glob 'suzhou*')
for arg in "$@"; do
  case "$arg" in
    --host-glob|--host-glob=*) host_args=(); break ;;
  esac
done

# 避免 CMD 在 UNC 工作目录启动的警告，并使用 UTF-8 输出用户目录。
windows_profile=$(cd /mnt/c && cmd.exe /d /c 'chcp 65001 >nul & echo %USERPROFILE%')
windows_profile=${windows_profile//$'\r'/}
windows_ssh_home="$(wslpath -u "$windows_profile")/.ssh"

echo "windows_ssh_home: $windows_ssh_home"
# Windows SSH 客户端（例如 ssh-usg0）：--config 和 --public-key 使用 WSL 路径，
# 用 --ssh-config 'C:\\Users\\用户名\\.ssh\\config' 指定客户端配置路径。
#
# windows-askpass：自动输入 Password，并转换 ssh-copy-id 探测用的私钥路径。
python3 "$(dirname -- "${BASH_SOURCE[0]}")/install-ssh-keys.py" \
  --config "$windows_ssh_home/config" \
  --ssh-config "$(wslpath -w "$windows_ssh_home/config")" \
  --ssh ssh-usg0 \
  --password-method windows-askpass \
  --public-key "$windows_ssh_home/id_ed25519.pub" \
  --timeout 120 "${host_args[@]}" "$@"
