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

连接失败时加 `--debug`
```bash
# 服务器
bash ~/devtools/update-dev-git-repo.sh --debug
# WSL（先同步更新后的脚本）
./devtools/update-dev-git-repo.sh --host suzhou-a3-130 --debug
```

本地 smoke 测试（临时 Git 仓库，不访问 GitHub、不修改用户目录）：

```bash
uv run --with pytest python -m pytest smoketest -q
```

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

