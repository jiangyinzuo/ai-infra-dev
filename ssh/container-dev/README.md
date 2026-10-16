# Docker容器开发环境

用于向远程服务器的宿主机（Python3.9）同步
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

`download.py` 要求 **Python 3.9+ 和 curl**；默认并发 8，完成即补位。终端安装 Rich 后显示每个工具的进度、大小、速度和剩余时间：`python3 -m pip install --user 'rich<15'`。未安装 Rich 或重定向 stderr 时使用普通日志，不自动安装依赖；未知总大小不显示百分比。

连接超时 10 秒，元数据每次最多 60 秒，归档每次最多 30 分钟且连续 60 秒低于 1 KiB/s 时中止；临时传输错误最多重试 2 次。失败任务具名报告，其余任务继续，最终返回非零。SHA-256 校验成功后才原子替换；Ctrl-C 终止后台请求并清理临时文件，保留已有归档。

WSL运行：
```bash
# 默认并发 8，逐文件校验 SHA-256
python3 ./devtools/openeuler24.03-lts-aarch64/download.py --jobs 8
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

### 真实 root 与 bindfs 映射（实验性）

`root_mapped_create.py` 在服务器以普通用户运行，需要免密 sudo、Docker、bindfs。使用 rootful 特权容器，默认 HOME 环境变量、root 账号 home 及工作目录均为宿主机 HOME。HOME 和显式 `--share` 目录通过 FUSE 映射 UID/GID 0 到当前宿主机 UID/GID；新文件强制归属当前用户，chown/chgrp 返回成功但不改变底层属主，不递归修改源目录。`.local/bin`、`.local/lib` 是独立 Docker volume，不共享给宿主机。驱动路径只读挂载，未声明的 /data 等目录不自动共享。

运行时打印实际执行的 Docker、sudo/bindfs 和清理命令（按 shell 规则引用参数）；绿色显示创建/成功状态，黄色显示目录映射及操作提示。成功后输出进入容器、停止删除、卸载映射和清理空目录的可复制命令。

脚本按参数检查、bindfs 挂载、Docker 命令组装、root HOME 初始化和操作提示拆分；`create_container()` 统一管理创建顺序及异常清理，`main()` 仅调用入口流程。

```bash
# 先同步工具；本脚本只在服务器直接执行
python3 ~/devtools/containers/root_mapped_create.py --image IMAGE --name NAME
# 隔离验证，不挂真实 HOME；额外数据目录需已存在且可写
python3 ~/devtools/containers/root_mapped_create.py --image IMAGE --name NAME --test-home /path/to/test-home --share /path/to/test-data
```

映射进程和挂载必须持续存在，脚本输出挂载目录及清理命令；先删除容器及 volume，再卸载 FUSE，最后删除空挂载目录。重启后不会自动恢复挂载，不要在缺少映射挂载时启动旧容器。特权容器可绕开文件系统映射，不是安全边界。FUSE 有额外 I/O 开销；忽略 chown/chgrp 便于开发安装器运行，但依赖实际切换文件属主的服务不适用。未验证完整算子编译性能。node-97-31 隔离测试中原始 npu-smi、未关闭自动加载的 import torch、0600 文件宿主机编辑、mmap 已验证；不是原生 idmapped mount（该机器 mount_setattr 返回 ENOSYS）。

2026-10-16 在 `jiangyinzuo-bindfs-build` 挂载真实 HOME 后，以真实 root 完成 `COMPILE_CUSTOM_KERNELS=1 pip3 install -v -e . --no-deps --no-build-isolation`，退出码 0。使用原生 Python、未修改 setup.py、未关闭 Torch 后端自动加载、无 npu-smi 包装器。进一步清除 CMAKE_PREFIX_PATH、Torch_DIR 后完整构建仍成功：项目 CMakeLists.txt 通过 cmake/utils.cmake 自动执行 Python 查询 Torch 路径，root 环境没有驱动诊断污染 stdout，因此无需显式指定路径。之前独立的 find_package 实验遗漏了项目的自动查询步骤，不能据此断言项目缺少依赖发现。`import torch; import vllm_ascend` 成功。容器内两主 `.so` 为 `0:0`，宿主机为 `9011:9011` 且可写，csrc/build 和 _cann_ops_custom 未发现其他 UID 所有的文件。日志 `~/vllm-ascend-bindfs-build.log` 及 `~/vllm-ascend-bindfs-no-torch-path.log`；保留原有源码构建缓存，主扩展使用新的临时 build-temp，未做清空全部缓存的性能对比或 NPU 运算测试。

```bash
# 查看服务器已有镜像
ssh2 suzhou-a3-130 docker images vllm-ascend

# WSL 本地下载，必须指定平台；归档保存到 containers/
./devtools/containers/download.py --host localhost --npu-platform a3 --date 20261014 --version 26.2.0
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

- 仅支持 A3，在服务器创建容器；WSL 使用 `--host`，本地下载需指定 `--npu-platform a3`。自动平台探测需免密 sudo，也可手动指定平台。
- 容器沿用宿主机用户名、UID/GID 和 HOME，自动挂载 NPU 并添加设备组及 GID 0；这些配置仅对新建容器生效，不修改共享 HOME 所有权或宿主机设备权限。
- openEuler 初始化需联网安装 `sudo`、`nmap-ncat`，并为当前用户配置免密 sudo。GID 0 和 sudo 会扩大共享目录访问权限，操作会影响宿主机；设备占用等驱动限制仍需单独处理。
- 自动继承 API key、`USG_IP` 和代理变量；`SOC_VERSION` 仅由服务器传入，不从 WSL 覆盖。变量名单见 `devtools/config.py`。命令日志不打印密钥值，但 Docker inspect 可查看环境变量，勿分享敏感输出。

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

