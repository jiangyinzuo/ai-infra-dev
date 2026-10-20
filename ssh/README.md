# SSH连接相关脚本

以下命令均在当前 `ssh/` 目录执行。

## Install Dependencies

本目录脚本使用 Python 3.9+；请确保 `python3 --version` 满足要求。

```bash
python3 -m pip install -r requirements-wsl.txt
uv tool install --force --with ansible ansible-core
sudo apt install sshpass。
```

## sshconfig示例

`$HOME/.ssh/config`
```sshconfig
IgnoreUnknown Password
Host xiaoqiaoling-a3-xx
    Hostname 10.246.63.xx
    User root
    PassWord xxxxx
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    ProxyCommand ncat --proxy 172.23.125.152:3128 --proxy-type http %h %p
```

`/mnt/c/.../.ssh/config`
```sshconfig
IgnoreUnknown Password
Host suzhou-a3-xx
    HostName 80.5.9.xx
    User root
    PassWord  xxxxx
Host suzhou-a3-xx
    HostName 80.5.17.xx
    User root
    PassWord  xxxxx
Host lianqiuhu-950pr-xx
    HostName 141.61.41.xx
    User root
    PassWord  xxxxx
Host lianqiuhu-950dt-xx
    HostName 141.61.141.xx
    User root
    PassWord  xxxxx
```

## ssh连接基础脚本

### ssh2

Python 3.9+ 入口，按 Host 前缀选择客户端，配置维护在 `ssh_common/config.py`：
- `suzhou*`：Windows SSH，绑定 `$USG0_IP`。
- `lianqiuhu*`：Windows SSH，绑定 `$USG1_IP`。
- `xiaoqiaoling*`：WSL `/usr/bin/ssh`，不绑定网卡。

```bash
./ssh2 suzhou-a3-130
./ssh2 lianqiuhu-950pr-193 'curl -k -I https://github.com'
./ssh2 xiaoqiaoling-a3-24
```

网关连接保留 `USG_IP`，并由它生成 `DEVTOOLS_HTTP_PROXY=http://<USG_IP>:3128`；WSL 原生 SSH 透传本地 `DEVTOOLS_HTTP_PROXY`。非空时统一覆盖六个大小写 HTTP/HTTPS/ALL 代理变量；未设置时仍透传已有代理变量。两者均透传已设置的 `SUDOCODE_API_KEY`、`OPENAI_API_KEY`，未设置的不覆盖远端值。SFTP、仅转发及配置查询不注入环境。

GitHub SSH 通过 createuser 部署的 `~/.ssh/config` 和 Python 代理读取 `DEVTOOLS_HTTP_PROXY`，支持 URL 编码的用户名和密码；远端需 ncat，首次克隆前需完成 createuser 配置。认证信息可能出现在进程参数中，请勿分享相关调试输出。

密钥会进入 SSH 进程参数，勿分享包含远端命令的调试日志。


Bash completion:
```bash
source ~/ai-infra-dev/completion.bash
```

### 共享配置

`ssh_common/config.py` 统一维护主机分组、Windows/WSL SSH 路由、配置路径、公钥、远端用户、平台和环境变量名单。组名加 `*` 匹配主机；所有组通过 `ssh2` 连接。

SSH argv 统一由 `ssh_common.remote.ssh_arguments()` 构造，rsync 传输入口复用 `ssh_common.transfer.rsync_command()`；调用方仍决定传输权限、部分文件策略和进程执行方式。bindfs 查询与额度修复共用身份识别，提权修复保持严格读取及源/目标匹配。workspace dev 的 Docker exec 参数由 `dev/container_command.py` 统一构造，不改变服务环境或生命周期。

## 模型发现

使用 `container-dev/devtools/cli.py models discovery --host HOST --json`，参见 [容器开发工具](container-dev/README.md)。


## machine-info: 查看机器基本情况

输出到 `machine-info/`：`machines-output.md` 保存本次报告，`machines-output.csv` 按 Host 更新 `OS,Arch,NPUPlatform`，保留未查询或采集失败的数据。NPU 平台通过 `sudo -n npu-smi info -t board -i 0` 的 PCI Device ID 判断：`0xD803 → a3`、`0xD802 → a2`，其他为 `unknown`；需免密 sudo，缺少 npu-smi 不算采集错误。

## npu-top: 查看npu使用率

```bash
./npu-top/cli.py --group suzhou --host-glob '*-130'
./npu-top/cli.py --group xiaoqiaoling --containers
# -f、--limit、--become 等参数直接传给 Ansible
./npu-top/cli.py --group lianqiuhu -f 20
# 自定义 inventory
./npu-top/cli.py -i /path/to/inventory --containers
```

## createuser: 批量创建用户

## container-dev: 初始化容器开发环境

参见 [开发环境说明](container-dev/README.md)。先推送本地提交到远端可访问的上游，再运行 `./container-dev/devtools/cli.py git-repo update --host HOST`，之后才同步资源。服务器入口为 `python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py`；两端资源均默认保存在仓库的 devtools 目录，sync 不传管理脚本。远程 CLI（引导更新除外）要求两端 HEAD 一致；不一致时先重新更新仓库，再重试。
