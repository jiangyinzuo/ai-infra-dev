# Docker容器开发环境

用于向远程服务器的宿主机同步
- vimrc和ai-infra-dev仓库
- Neovim插件
- 命令行工具
- Docker镜像
- vllm-ascend仓？pip install?

同步目标固定为 `jiangyinzuo@HOST:~/devtools/`，默认仅预览，使用 `--apply` 执行。

自动根据OS + Arch同步对应文件：
- 同步前需要在 `ssh/` 运行 `./machine-info/machine-info.py` 生成 `machine-info/machines-output.csv`。
- 当前仅支持 openEuler + aarch64/arm64，自动选择 `openeuler24.03-lts-aarch64`。

## 更新开发 Git 仓库

在服务器的 `~/ai-infra-dev`、`~/vimrc` 克隆 `jiangyinzuo` 的同名 GitHub 仓库；已有仓库执行 `git pull --ff-only`。
先运行 createuser 配置用户的 GitHub SSH；仓库更新直接使用 `~/.ssh/config`，登录需提供 `USG_IP`。`--debug` 开启 Git/SSH 日志。

`vimrc` 首次克隆成功后，在仓库目录执行 `./install/setup.sh`；已有仓库 pull 后不重复安装。安装失败时返回其退出码，仓库保留，需要手动重跑 setup。原 `init-vimrc.sh` 已删除，不再执行其中的 `openeuler-ascend.sh`。

```bash
# WSL：先同步 devtools，必须显式指定 --host
./sync-devtools.py --apply suzhou-a3-130
./devtools/update-dev-git-repo.py --host suzhou-a3-130

# 服务器：直接执行，不指定 --host
python3 ~/devtools/update-dev-git-repo.py
```

连接失败时加 `--debug`
```bash
# 服务器
python3 ~/devtools/update-dev-git-repo.py --debug
# WSL（先同步更新后的脚本）
./devtools/update-dev-git-repo.py --host suzhou-a3-130 --debug
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

# WSL 本地下载，必须指定平台；归档保存到 containers/
./devtools/containers/download.py --host localhost --npu-platform a3 --date 20261012 --version 26.2.0
# 下载到 SSH Host 对应服务器，平台由 CSV 选择
./devtools/containers/download.py --host suzhou-a3-130
# 远程创建并启动，默认交互选择镜像与容器名
./devtools/containers/create.py --host suzhou-a3-130
# 服务器直接执行
python3 ~/devtools/containers/download.py
docker load -i ~/devtools/containers/vllm-ascend_dev-26.2.0.day20261012-800I-A3-py311-openEuler24.03-lts-aarch64.tar.gz
python3 ~/devtools/containers/create.py --image vllm-ascend:TAG --name jiangyinzuo-dev1
```

HOME 仍共享；每个新容器的 `~/.local/bin`、`~/.local/lib` 使用独立空 volume，避免 Python 工具和依赖互相覆盖。容器内需重新安装工具；删除容器时用 `docker rm -v` 一并清理这些 volume，已有容器不受影响。

```bash
# 进入容器
docker exec -it jiangyinzuo-dev1 bash
# 停止并删除容器，同时清理独立 volume
docker stop jiangyinzuo-dev1
docker rm -v jiangyinzuo-dev1
# 忘记 -v 时，查看遗留 volume 并按需清理（会询问确认）
docker volume ls --filter dangling=true
docker volume prune
```

`prune` 会清理未被任何容器引用的匿名 volume，不限于本工具创建的 volume；确认数据不再需要后再执行。

创建并启动后配置与宿主机用户名、UID/GID 和 home 一致的账号，初始化独立目录权限并校验用户名；不修改共享 HOME 所有权。普通提示为绿色，进入和删除提示为黄色，镜像列表和 Docker 命令保持默认颜色。
自动探测实际 NPU 字符设备和 GID，挂载设备并为容器用户添加对应附加组，同时添加 GID 0 以访问 CANN 的 root 组可读目录；主 UID/GID 不变，不修改宿主机设备权限。GID 0 也会授予共享挂载中 root 组允许的访问权限。仅对新建容器生效；设备权限之外的驱动限制仍可能需要单独处理。
openEuler 容器自动安装 `sudo`、`nmap-ncat` 并配置当前用户免密 sudo，验证成功后才完成初始化。安装时禁用 debuginfo、source、update-source 仓库；需要网络，失败则停止。免密 sudo 授予容器内 root 权限，注意共享目录中的操作也会影响宿主机。
执行前向标准输出打印实际 Docker 命令；透传密钥仅显示变量名，不显示值。

创建时继承已设置的 `SUDOCODE_API_KEY`、`OPENAI_API_KEY`、`USG_IP` 及大小写 HTTP/HTTPS/ALL 代理变量；`SOC_VERSION` 仅由服务器宿主机传给容器，不从 WSL 经 SSH 透传。Docker 环境可被有 Docker 权限的用户通过 inspect 查看，勿分享该输出。
共享配置统一维护在 `devtools/config.py`，随 devtools 同步；仅包含配置和变量名，不包含密钥值。

服务器本地通过 `sudo -n npu-smi info -t board -i 0` 的 PCI Device ID 探测平台，需免密 sudo；与 machine-info 共用 `devtools/machine_probe.py`。`0xD803` 为 A3，`0xD802` 为 A2；当前容器工具仅支持 A3，可用 `--npu-platform a3` 手动指定。WSL 本地下载仍需指定平台，禁止本机创建容器。

## 宿主机/容器内执行命令

```bash
# 容器内执行
pip install pylint pyrefly
cd  ~/ai-infra-dev/code/vllm-ascend-workspace/vllm-ascend
git clone git@github.com:vllm-project/vllm-ascend.git --depth 1000
VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS=1 pip3 install -v -e . --no-deps --no-build-isolation
# 宿主机让容器执行
docker exec jiangyinzuo-dev1 pip install pylint pyrefly
# WSL让宿主机执行
ssh2 suzhou-a3-130 pip install pylint pyrefly
# WSL让容器执行
ssh2 suzhou-a3-130 docker exec jiangyinzuo-dev1 pip install pylint pyrefly
```

