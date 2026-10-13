# `sshfs.sh` 使用 `ssh-usg1` 挂载远端目录；运行前需导出 `USG1_IP`，并在本目录执行 `sh sshfs.sh`。
# `ssh_command` 必须指向实际可执行文件，不能使用 shell alias/function；使用绝对路径避免 SSHFS 改变工作目录后找不到包装器。
# 远端目录使用绝对路径，不依赖 SFTP 展开 `~`。卸载：`fusermount3 -u ./jiangyinzuo-lianqiuhu-950pr-193-a5-example`。

sshfs jiangyinzuo@lianqiuhu-950pr-193:/home/jiangyinzuo/a5-example /home/j00958433/ai-infra-dev/ascendc/a5-example \
  -o ssh_command="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/ssh-usg1" \
  -o dir_cache=no,attr_timeout=0,entry_timeout=0,negative_timeout=0,direct_io \
  -o reconnect \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o ConnectTimeout=10
