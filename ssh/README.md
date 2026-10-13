## Install Dependencies

```bash
python3 -m pip install sshconf==0.2.7
uv tool install --force --with ansible ansible-core
```

## ssh-usg0 / ssh-usg1

一些服务器需要通过windows的ssh.exe -b <IP> 才能连接，`ssh-usg0`和`ssh-usg1`就是对这一命令的封装脚本。

## hosts.json

在wsl中，同时支持`~/.ssh/config`和依赖bind address的Windows `.../.ssh/config`，
详见 `hosts.json` 和 `ssh_hosts.py`。
