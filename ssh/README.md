# SSH连接相关脚本

## Install Dependencies

```bash
python3 -m pip install sshconf==0.2.7
uv tool install --force --with ansible ansible-core
```

## ssh-usg0 / ssh-usg1 / ssh-usg0-proxy / ssh-usg1-proxy

一些服务器需要通过windows的ssh.exe -b <IP> 才能连接，`ssh-usg0`和`ssh-usg1`就是对这一命令的封装脚本。

```bash
ssh-usg0 suzhou-a3-130
ssh-usg1 lianqiuhu-950pr-193
```

`ssh-usg0-proxy` / `ssh-usg1-proxy`，会在远程服务器导出 `http_proxy`、`https_proxy`、`all_proxy`、`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`，值均为 `http://$USG_IP:3128`。
相当于`ssh-usg0`登录后，`source $HOME/devtools/proxy.env`

```bash
ssh-usg0-proxy suzhou-a3-130
ssh-usg1-proxy lianqiuhu-950pr-193 'curl -k -I https://github.com'
```

Bash completion:
```bash
source ~/ai-infra-dev/ssh/ssh-usg-completion.bash
```

## hosts.json

在wsl中，同时支持`~/.ssh/config`和依赖bind address的Windows `.../.ssh/config`，
详见 `hosts.json` 和 `ssh_hosts.py`。
