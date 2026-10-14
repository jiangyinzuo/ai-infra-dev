# Docker容器开发环境

用于向远程服务器的宿主机同步
- Neovim插件
- 命令行工具
- Docker镜像
- ai-infra-dev仓和vllm-ascend仓？

## 更新并同步Neovim插件

读取 `$HOME/.config/nvim/lazy-lock.json`，WSL 本地下载源码归档至 `devtools/nvim-plugins/`

WSL运行：
```bash
./devtools/download-nvim-plugins.py
./sync-devtools.sh openeuler24.03-lts-aarch64 suzhou-a3-130 --apply
# 如果判断运行环境是WSL，则要求输入host。在远程服务器则不需要
./devtools/extract-nvim-plugin.sh --host suzhou-a3-130 --jobs 8
```

## 更新开发 Git 仓库

在服务器的 `~/ai-infra-dev`、`~/vimrc` 克隆 `jiangyinzuo` 的同名 GitHub 仓库；已有仓库执行 `git pull --ff-only`，不自动丢弃本地修改或合并分叉。

```bash
# WSL：先同步 devtools，必须显式指定 --host
./sync-devtools.sh openeuler24.03-lts-aarch64 suzhou-a3-130 --apply
./devtools/update-dev-git-repo.sh --host suzhou-a3-130

# 服务器：直接执行，不指定 --host
bash ~/devtools/update-dev-git-repo.sh
```

Git SSH 通过临时 `GIT_SSH_COMMAND` 使用 `$USG_IP:3128` 的 HTTP CONNECT 代理，不读取其他代理变量或修改全局配置。服务器需要 Git、OpenSSH 和 `nc`；自动识别 Nmap Ncat（`--proxy-type http --proxy`）与 OpenBSD netcat（`-X connect -x`）。`USG_IP` 由 SSH 包装器传入，远端需已配置 GitHub SSH 密钥和主机信任。

保留 `git@github.com:jiangyinzuo/...` 的仓库 URL，但临时将 SSH 连接目标指定为 `ssh.github.com:443`，避免网关对 `github.com:22` 返回 HTTP 403 拦截页。使用 `HostKeyAlias=github.com` 保留 GitHub 主机信任身份，不关闭主机密钥校验；远端缺少已核验的主机密钥或 GitHub 身份密钥时，仍需先配置，脚本不会自动接受未知主机或复制私钥。

## 更新并同步命令行工具

WSL运行：
```bash
# 默认并发 8，逐文件校验 SHA-256
./devtools/openeuler24.03-lts-aarch64/download.sh --jobs 8
./sync-devtools.sh openeuler24.03-lts-aarch64 suzhou-a3-130 --apply
# 如果判断运行环境是WSL，则要求输入host。在远程服务器则不需要
./devtools/openeuler24.03-lts-aarch64/extract.sh --host suzhou-a3-130
```

## 创建Docker容器

```bash
# 查看当前服务器镜像
ssh-usg0 suzhou-a3-130 docker images vllm-ascend
```

```bash
# 本地WSL或服务器下载
./devtools/images/a3/download-a3-container.sh
```

创建容器:
```bash
docker load -i ~/devtools/images/a3/vllm-ascend_dev-26.2.0.day20261012-800I-A3-py311-openEuler24.03-lts-aarch64.tar.gz
# 未设置 IMAGE_NAME 时按创建时间从新到旧列出镜像，回车选最新。
# 未设置 CONTAINER_NAME 时提示输入，默认 jiangyinzuo-dev<最大现有编号+1>。
./devtools/images/a3/run-a3-container.sh
# 环境变量可分别跳过对应选择提示：
# IMAGE_NAME=9824ebc97c4d CONTAINER_NAME=jiangyinzuo-dev1 ./devtools/images/a3/run-a3-container.sh
```

