# Docker容器开发环境

NOTE: 同步前需要在 `ssh/` 运行 `./machine-info/machine-info.py` 生成 `machine-info/machines-output.csv`。

按 Host 查找 OS/Arch，当前仅支持 openEuler + aarch64/arm64，自动选择 `openeuler24.03-lts-aarch64`。

用于向远程服务器的宿主机同步
- vimrc和ai-infra-dev仓库
- Neovim插件
- 命令行工具
- Docker镜像
- vllm-ascend仓？pip install?

## 更新开发 Git 仓库

在服务器的 `~/ai-infra-dev`、`~/vimrc` 克隆 `jiangyinzuo` 的同名 GitHub 仓库；已有仓库执行 `git pull --ff-only`。

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

fzf 下载最新稳定 release 的 Linux ARM64 程序及同版本 `fzf-preview.sh`、`fzf-tmux`，解压安装到 `$HOME/plugged/fzf/bin/`。

WSL运行：
```bash
# 默认并发 8，逐文件校验 SHA-256
./devtools/openeuler24.03-lts-aarch64/download.sh --jobs 8
./sync-devtools.py --apply suzhou-a3-130
# 如果判断运行环境是WSL，则要求输入host。在远程服务器则不需要
# 解压并在 ~/.bashrc 配置平台目录下的 path.env，重复运行不重复添加。
./devtools/openeuler24.03-lts-aarch64/extract.sh --host suzhou-a3-130
```

## 构建nvim插件

经过上面两步后，服务器环境上同时有了nvim和插件。
```bash
nvim --headless "+Lazy! build telescope-fzf-native.nvim" +qa
```

## 容器镜像与创建

目前支持 NPU 平台 `a3`。WSL 必须指定 `--host`；远程平台读取 `ssh/machine-info/machines-output.csv` 的 `NPUPlatform`，先用 machine-info 采集。远程脚本需先同步到服务器。

Python 脚本兼容 3.9 及以上版本，本地无需切换解释器。

```bash
# 查看服务器已有镜像
ssh2 suzhou-a3-130 docker images vllm-ascend

# WSL 本地下载，必须指定平台；归档保存到 images/
./devtools/images/download-container.py --host localhost --npu-platform a3 --date 20261012 --version 26.2.0
# 下载到 SSH Host 对应服务器，平台由 CSV 选择
./devtools/images/download-container.py --host suzhou-a3-130
# 远程创建，默认交互选择镜像与容器名（不会启动容器）
./devtools/images/create-container.py --host suzhou-a3-130
# 服务器直接执行
python3 ~/devtools/images/download-container.py
docker load -i ~/devtools/images/vllm-ascend_dev-26.2.0.day20261012-800I-A3-py311-openEuler24.03-lts-aarch64.tar.gz
python3 ~/devtools/images/create-container.py --image vllm-ascend:TAG --name jiangyinzuo-dev1
```

创建时继承已设置的 `SUDOCODE_API_KEY`、`OPENAI_API_KEY`、`USG_IP` 及大小写 HTTP/HTTPS/ALL 代理变量。Docker 环境可被有 Docker 权限的用户通过 inspect 查看，勿分享该输出。

服务器本地通过 `sudo -n npu-smi info -t board -i 0` 的 PCI Device ID 探测平台，需免密 sudo；与 machine-info 共用 `devtools/machine_probe.py`。`0xD803` 为 A3，`0xD802` 为 A2；当前容器工具仅支持 A3，可用 `--npu-platform a3` 手动指定。WSL 本地下载仍需指定平台，禁止本机创建容器。
