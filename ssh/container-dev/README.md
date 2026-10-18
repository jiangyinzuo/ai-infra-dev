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
不下载或同步的插件，见 NVIM_PLUGIN_EXCLUDES。

WSL运行：
```bash
./devtools/download-nvim-plugins.py
./sync-devtools.py --apply suzhou-a3-130
# 如果判断运行环境是WSL，则要求输入host。在远程服务器则不需要
./devtools/extract-nvim-plugin.sh --host suzhou-a3-130 --jobs 8
```

## 更新并同步命令行工具及 bindfs

WSL 下载并同步工具：

```bash
python3 ./devtools/openeuler24.03-lts-aarch64/download.py --jobs 8
./sync-devtools.py --apply suzhou-a3-130
```

先在服务器更新 `vimrc`，确保编译脚本可用：

```bash
python3 ~/devtools/update-dev-git-repo.py
```

然后回到 WSL 执行解压：

```bash
./devtools/openeuler24.03-lts-aarch64/extract.sh --host suzhou-a3-130
```

`extract.sh` 会检查 bindfs：已安装则跳过；未安装时自动用 `yum` 安装编译依赖，并调用 `~/vimrc/install/build_bindfs.sh` 编译安装。首次安装后在服务器运行 `source ~/.bashrc`，用 `bindfs --version` 验证。下载的是 bindfs 官方源码压缩包，不是 Git 仓库。

## 容器镜像下载与创建

### 下载镜像

```bash
# WSL 本地下载，必须指定平台
./devtools/containers/cli.py download --host localhost --npu-platform a3 --date 20261014 --version 26.2.0
# 下载到 SSH Host 对应服务器，平台由 CSV 选择
./devtools/containers/cli.py download --host suzhou-a3-130
# 服务器直接下载
python3 ~/devtools/containers/cli.py download
```

下载归档后，在服务器用 `docker load -i <归档路径>` 导入镜像。

### 创建容器

```bash
# 服务器运行；create 不带选项时交互选择镜像，并确认容器名
python3 ~/devtools/containers/cli.py create
# 也可以明确指定镜像和容器名
python3 ~/devtools/containers/cli.py create --image vllm-ascend:TAG --name jiangyinzuo-dev1
# 性能测试：不用 bindfs，直接挂载；新文件可能归宿主机 root 所有
python3 ~/devtools/containers/cli.py create --no-bindfs
# 隔离验证，不挂真实 HOME；测试目录需已存在且可写
python3 ~/devtools/containers/cli.py create --test-home /path/to/test-home --share /path/to/test-data
```

镜像按创建时间由新到旧列出；bindfs 容器默认名为 `jiangyinzuo-bindfs-dev数字`，直接挂载为 `jiangyinzuo-dev数字`，回车即可接受。

### bindfs 目录映射

脚本通过 bindfs 将宿主机 HOME 映射到容器，`--share` 可添加其他共享目录。相同源目录、UID/GID 和映射策略自动复用；旧版无记录的随机挂载不自动接管。

容器内 root 创建的文件归属宿主机当前用户，可以共享源码和开发文件。

> FUSE 也会增加 I/O 开销。

容器仍在同一台运行中的宿主机时，可直接重启：

```bash
docker restart jiangyinzuo-dev1
```

如果宿主机重启，bindfs 挂载会丢失，但容器本身和容器内数据仍保留。先在宿主机重新执行创建时输出的每条 `sudo bindfs ...` 命令（目标目录不存在时先 `mkdir -p <目标目录>`），再启动原容器：

```bash
docker start jiangyinzuo-dev1
docker exec -it jiangyinzuo-dev1 bash
```

不要在挂载恢复前启动容器，否则容器看到的共享目录可能是空目录。无需删除或重建容器。

删除时可交互选择是否保留映射，或明确指定；两种方式均删除容器及匿名 volume，不删除共享源目录。仍被其他容器（含已停止容器）使用的映射不会卸载。

```bash
python3 ~/devtools/containers/cli.py delete
python3 ~/devtools/containers/cli.py delete --name jiangyinzuo-bindfs-dev1 --keep-bindfs
python3 ~/devtools/containers/cli.py delete --name jiangyinzuo-bindfs-dev1 --remove-bindfs
```

用 `python3 ~/devtools/containers/cli.py ls` 查看容器和映射（自动化可加 `--json`）。容器或测试源目录被手动删除后，用 `python3 ~/devtools/containers/cli.py delete --cleanup-bindfs` 交互确认清理残留；自动化可加 `--yes`。只清理可验证且无容器引用的映射，不删除源目录。

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

