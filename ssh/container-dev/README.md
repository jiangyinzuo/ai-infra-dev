# Docker容器开发环境

NOTE: 同步前需要在 `ssh/` 运行 `./machine-info/machine-info.py` 生成 `machine-info/machines-output.csv`。

按 Host 查找 OS/Arch，当前仅支持 openEuler + aarch64/arm64，自动选择 `openeuler24.03-lts-aarch64`。

用于向远程服务器的宿主机同步
- vimrc和ai-infra-dev仓库
- Neovim插件
- 命令行工具
- Docker镜像
- vllm-ascend仓？

## 更新开发 Git 仓库

在服务器的 `~/ai-infra-dev`、`~/vimrc` 克隆 `jiangyinzuo` 的同名 GitHub 仓库；已有仓库执行 `git pull --ff-only`，不自动丢弃本地修改或合并分叉。

`vimrc` 首次克隆成功后，在仓库目录执行 `./install/setup.sh`；已有仓库 pull 后不重复安装。安装失败时返回其退出码，仓库保留，需要手动重跑 setup。原 `init-vimrc.sh` 已删除，不再执行其中的 `openeuler-ascend.sh`。

```bash
# WSL：先同步 devtools，必须显式指定 --host
./sync-devtools.py --apply suzhou-a3-130
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

## 更新并同步Neovim插件

读取 `$HOME/.config/nvim/lazy-lock.json`，WSL 本地下载源码归档至 `devtools/nvim-plugins/`

WSL运行：
```bash
./devtools/download-nvim-plugins.py
./sync-devtools.py --apply suzhou-a3-130
# 如果判断运行环境是WSL，则要求输入host。在远程服务器则不需要
./devtools/extract-nvim-plugin.sh --host suzhou-a3-130 --jobs 8
```

## 更新并同步命令行工具

WSL运行：
```bash
# 默认并发 8，逐文件校验 SHA-256
./devtools/openeuler24.03-lts-aarch64/download.sh --jobs 8
./sync-devtools.py --apply suzhou-a3-130
# 如果判断运行环境是WSL，则要求输入host。在远程服务器则不需要
# 解压并在 ~/.bashrc 配置平台目录下的 path.env，重复运行不重复添加。
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

