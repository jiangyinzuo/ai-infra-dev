# SSH连接相关脚本

以下命令均在当前 `ssh/` 目录执行。

## Install Dependencies

```bash
python3 -m pip install sshconf==0.2.7
uv tool install --force --with ansible ansible-core
```

## ssh连接基础脚本

### ssh-usg0 / ssh-usg1 / ssh-usg0-proxy / ssh-usg1-proxy

一些服务器需要通过windows的ssh.exe -b <IP> 才能连接，`ssh-usg0`和`ssh-usg1`就是对这一命令的封装脚本。

```bash
./ssh-usg0 suzhou-a3-130
./ssh-usg1 lianqiuhu-950pr-193
```

`ssh-usg0-proxy` / `ssh-usg1-proxy`，会在远程服务器导出 `http_proxy`、`https_proxy`、`all_proxy`、`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`，值均为 `http://$USG_IP:3128`。
相当于`ssh-usg0`登录后，`source $HOME/devtools/proxy.env`

```bash
./ssh-usg0-proxy suzhou-a3-130
./ssh-usg1-proxy lianqiuhu-950pr-193 'curl -k -I https://github.com'
```

Bash completion:
```bash
source ./completion.bash
```

### `hosts.json` 和 `ssh_hosts.py`

在wsl中，同时支持
- ssh + `~/.ssh/config`
- ssh.exe -b <ip> + `/mnt/c/.../.ssh/config`

## machine-info: 查看机器基本情况

输出到 `machine-info/`：`machines-output.md` 保存本次报告，`machines-output.csv` 按 Host 更新 OS、Arch，供同步工具选择平台；未查询或采集失败的数据保留。

## npu-top: 查看npu使用率

## createuser: 批量创建用户

## container-dev: 初始化容器开发环境
