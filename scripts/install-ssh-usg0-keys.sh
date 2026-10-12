#!/usr/bin/env bash
set -euo pipefail

# 在 WSL 中通过 ssh-usg0 上传 Windows 当前用户的公钥；需先设置 USG0_IP。
# 避免 CMD 在 UNC 工作目录启动的警告，并使用 UTF-8 输出用户目录。
windows_profile=$(cd /mnt/c && cmd.exe /d /c 'chcp 65001 >nul & echo %USERPROFILE%')
windows_profile=${windows_profile//$'\r'/}
windows_ssh_home="$(wslpath -u "$windows_profile")/.ssh"

echo "windows_ssh_home: $windows_ssh_home"
# Windows SSH 客户端（例如 ssh-usg0）：--config 和 --public-key 使用 WSL 路径，
# 用 --ssh-config 'C:\\Users\\用户名\\.ssh\\config' 指定客户端配置路径。
#
# --password-method windows-askpass: 使用 Windows SSH_ASKPASS 自动输入 Password
python3 "$(dirname -- "${BASH_SOURCE[0]}")/install-ssh-keys.py" \
  --config "$windows_ssh_home/config" \
  --ssh-config "$(wslpath -w "$windows_ssh_home/config")" \
  --ssh ssh-usg0 \
  --password-method windows-askpass \
  --public-key "$windows_ssh_home/id_ed25519.pub" \
  --host-glob 'suzhou*' \
  --timeout 120 "$@"
