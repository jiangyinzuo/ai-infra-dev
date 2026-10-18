# Docker 容器开发环境

统一入口为 `./devtools/cli.py`，同步后在服务器使用 `python3 ~/devtools/cli.py`。支持 Python 3.9。

```text
nvim-plugins download / extract
git-repo update
binaries download / extract / sync
containers create / delete / download / ls
```

各操作均支持 `--host HOST`；服务器本地执行时省略。WSL 更新仓库或解压时必须指定远程主机。使用 `--help` 查看选项，入口处加 `--no-hints` 可关闭后续步骤提示。

## 同步工具

在 `ssh/` 先运行 `./machine-info/machine-info.py` 更新机器信息。当前同步支持 openEuler aarch64/arm64，目标为 `jiangyinzuo@HOST:~/devtools/`。

默认只预览，`--apply` 才执行；多个目标重复指定 `--host`：

```bash
./devtools/cli.py binaries sync --host suzhou-a3-130
./devtools/cli.py binaries sync --host suzhou-a3-130 --host suzhou-a3-131 --apply
```

## 更新开发 Git 仓库

先配置用户的 GitHub SSH。首次克隆 `~/ai-infra-dev` 和 `~/vimrc`，已有仓库仅快进更新；首次克隆 vimrc 后自动安装。连接失败可加 `--debug`。

```bash
# WSL：先同步入口及工具，再远程更新
./devtools/cli.py binaries sync --host suzhou-a3-130 --apply
./devtools/cli.py git-repo update --host suzhou-a3-130
# 服务器本地
python3 ~/devtools/cli.py git-repo update
```

## 更新 Neovim 插件

默认读取 `~/.config/nvim/lazy-lock.json`，下载插件源码归档。WSL 执行：

```bash
./devtools/cli.py nvim-plugins download
./devtools/cli.py binaries sync --host suzhou-a3-130 --apply
./devtools/cli.py nvim-plugins extract --host suzhou-a3-130 --jobs 8
```

## 更新命令行工具

```bash
./devtools/cli.py binaries download --jobs 8
./devtools/cli.py binaries sync --host suzhou-a3-130 --apply
./devtools/cli.py git-repo update --host suzhou-a3-130
./devtools/cli.py binaries extract --host suzhou-a3-130
```

解压时会安装缺失的 bindfs。首次安装后在服务器运行 `source ~/.bashrc`。

## 容器

下载镜像归档后，用 `docker load -i <归档路径>` 导入。

```bash
# WSL 本地下载需指定平台；远程下载可由机器信息选择平台
./devtools/cli.py containers download --host localhost --npu-platform a3 --date 20261014 --version 26.2.0
./devtools/cli.py containers download --host suzhou-a3-130
# 以下命令在服务器本地执行，也可在 WSL 加 --host 远程执行
python3 ~/devtools/cli.py containers download
python3 ~/devtools/cli.py containers create
python3 ~/devtools/cli.py containers create --image vllm-ascend:TAG --name jiangyinzuo-dev1
python3 ~/devtools/cli.py containers ls
python3 ~/devtools/cli.py containers ls --json
```

创建默认交互选择镜像和容器名，并通过 bindfs 共享 HOME；容器内 root 创建的文件归宿主机当前用户。`--share PATH` 添加共享目录，`--test-home PATH` 使用隔离目录，`--no-bindfs` 改为直接挂载（文件可能归 root 所有，避免 FUSE 开销）。

删除容器及匿名 volume，不删除共享源目录；仍被其他容器使用的映射不会卸载：

```bash
python3 ~/devtools/cli.py containers delete
python3 ~/devtools/cli.py containers delete --name jiangyinzuo-dev1 --keep-bindfs
python3 ~/devtools/cli.py containers delete --name jiangyinzuo-dev1 --remove-bindfs
python3 ~/devtools/cli.py containers delete --cleanup-bindfs
```

残留映射清理默认交互确认，自动化可加 `--yes`。

宿主机未重启时可直接 `docker restart 容器名`。宿主机重启后，先重新执行创建时输出的 `sudo bindfs ...` 命令恢复映射，再 `docker start 容器名`；不要在恢复前启动容器，否则共享目录可能为空。容器及其内部数据仍保留，无需重建。

## 补全与测试

在 `ssh/` 执行 `source ./completion.bash`，支持直接运行入口和 `python3 路径/cli.py` 的命令、选项与主机名补全。

```bash
uv run --with pytest python -m pytest smoketest -q
```
