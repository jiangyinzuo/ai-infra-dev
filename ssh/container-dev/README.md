# Docker 容器开发环境

统一入口为 `./devtools/cli.py`；服务器使用 `python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py`。支持 Python 3.9。两端资源默认都保存在仓库的 `ssh/container-dev/devtools/`，不依赖运行目录；可用绝对路径 `DEVTOOLS_DATA_DIR` 覆盖本机数据目录。

```text
git-repo update
binaries download / extract / sync
containers create / delete / download / ls
models discovery
```

各操作均支持 `--host HOST`；服务器本地执行时省略。WSL 更新仓库或解压时必须指定远程主机。使用 `--help` 查看选项，入口处加 `--no-hints` 可关闭后续步骤提示。

`python3 ssh/container-dev/devtools/cli.py checkhealth --host HOST` 快速只读检查两个 Git 仓库、bindfs/nvim/fzf 版本和插件目录数量，不输出插件清单、不联网、不 sudo；WARN 时退出 1。插件仅检查目录存在且非空，不代表版本或内容一致。

## 发现模型

```bash
./devtools/cli.py models discovery --host suzhou-a3-130 --json
# 服务器本地可省略 --host；--root 可重复，覆盖默认搜索目录
./devtools/cli.py models discovery --root /data/weights
```

默认检查 `/data{,1,2,3}` 及其中的 `weight/weights/model/models`，以及 `/mnt/share/weights`，每个搜索根目录只检查自身及直接子目录，不递归深入。候选目录需有 `config.json` 和 safetensors 或 pytorch_model bin 权重；不验证模型完整性。WSL 必须指定远端，远程执行要求仓库 commit 一致；`--timeout` 默认 60 秒。

## 拉取服务器模型（仅 WSL）

```bash
python3 ssh/container-dev/devtools/cli.py models fetch \
  --host suzhou-a3-130 --output ~/models
# 已知远端路径：跳过发现和交互
python3 ssh/container-dev/devtools/cli.py models fetch \
  --host suzhou-a3-130 --model /data/models/Qwen3-1.7B --output ~/models
```

复用 `models discovery` 的发现逻辑，列出模型后输入编号选择，`q` 取消；`--root` 可重复指定扫描目录。也可用 `--model` 指定远端绝对路径直接拉取，与 `--root` 互斥。下载到 `--output/远端模型目录名`。两端需 rsync，复用 ssh2 路由及 commit 校验；显示传输进度并保留部分文件供重跑使用。解引用符号链接，下载实际权重；同名文件可能更新，不删除本地额外文件。`--timeout` 仅限制模型发现，不限制大模型传输。

## 上传模型（仅 WSL）

```bash
python3 ssh/container-dev/devtools/cli.py models upload \
  --host suzhou-a3-130 --model ~/models/Qwen3-1.7B
```

按编号选择远端父目录，默认使用 discovery 的搜索根目录；`--root` 可重复指定候选位置。每项显示目录是否非空、是否软链接、jiangyinzuo 的读写及进入权限，以及目标模型目录状态。无权限项和软链接模型目标不可选择；已有目标需再次确认。上传使用 rsync，解引用本地链接，保留远端额外文件。上传文件设置为 0644、目录为 0755，不开放公共写权限；不更改父目录权限。所有人可达标记按父目录的传统 Unix 权限位判断，ACL/安全策略可能另有限制。

## 同步工具

在 `ssh/` 先运行 `./machine-info/machine-info.py` 更新机器信息。先运行 `git-repo update`，再同步资源；同步只传资源，不传管理脚本，目标为远端仓库的 `ssh/container-dev/devtools/`。

WSL 一次完成拉取、下载至本地、同步和远程解压（任一步失败即停止）：`./devtools/cli.py sync-all --host suzhou-a3-130`。

首次构建 bindfs 时自动安装系统依赖；`sudo yum` 显式继承当前已设置的 HTTP/HTTPS/ALL 代理及 NO_PROXY（含大小写形式），不保留其他用户环境，不修改系统软件源或证书策略。

默认只预览，`--apply` 才执行；多个目标重复指定 `--host`：

```bash
./devtools/cli.py git-repo update --host suzhou-a3-130
./devtools/cli.py binaries sync --host suzhou-a3-130
./devtools/cli.py git-repo update --host suzhou-a3-131
./devtools/cli.py binaries sync --host suzhou-a3-130 --host suzhou-a3-131 --apply
```

## 更新开发 Git 仓库

先配置用户的 GitHub SSH。首次克隆 `~/ai-infra-dev` 和 `~/vimrc`，已有仓库仅快进更新；首次克隆 vimrc 后自动安装。连接失败可加 `--debug`。本地提交须先推送到远端可访问的上游，确保更新能取得相同提交；仅在本地 commit 不够。

```bash
# WSL：先更新远端仓库，再同步资源；首次无需远端已有脚本
./devtools/cli.py git-repo update --host suzhou-a3-130
./devtools/cli.py binaries sync --host suzhou-a3-130 --apply
# 服务器本地（仓库已存在）
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py git-repo update
```

除 `git-repo update` 引导更新外，远程 CLI 在执行操作前会检查远端 `~/ai-infra-dev` 的 HEAD 是否与本地一致；不一致时拒绝执行。先推送本地提交，再重新运行 `./devtools/cli.py git-repo update --host HOST`，然后重试原命令。

## 更新命令行工具和 Neovim 插件

`binaries download` 一并下载工具和插件归档，插件默认读取 `~/.config/nvim/lazy-lock.json`；`binaries extract` 一并解压。WSL 执行：

```bash
./devtools/cli.py binaries download --jobs 8
./devtools/cli.py git-repo update --host suzhou-a3-130
./devtools/cli.py binaries sync --host suzhou-a3-130 --apply
./devtools/cli.py binaries extract --host suzhou-a3-130 --jobs 8
```

插件下载在来源 URL 和归档哈希均匹配时跳过；解压在归档和目标状态均未变化时跳过，目标缺失或被修改时重新解压。

解压时会安装缺失的 bindfs。首次安装后在服务器运行 `source ~/.bashrc`。

## 容器

下载镜像归档后，用 `docker load -i <归档路径>` 导入。

```bash
# WSL 本地下载需指定平台；远程下载可由机器信息选择平台
./devtools/cli.py containers download --host localhost --npu-platform a3 --date 20261014 --version 26.2.0
./devtools/cli.py containers download --host suzhou-a3-130
# 以下命令在服务器本地执行，也可在 WSL 加 --host 远程执行
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers download
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers create
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers create --image vllm-ascend:TAG --name jiangyinzuo-dev1
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers ls
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers ls --json
```

创建默认交互选择镜像和容器名，并通过 bindfs 共享 HOME；容器内 root 创建的文件归宿主机当前用户。`--share PATH` 添加共享目录，`--test-home PATH` 使用隔离目录，`--no-bindfs` 改为直接挂载（文件可能归 root 所有，避免 FUSE 开销）。

删除容器及匿名 volume，不删除共享源目录；仍被其他容器使用的映射不会卸载：

```bash
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers delete
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers delete --name jiangyinzuo-dev1 --keep-bindfs
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers delete --name jiangyinzuo-dev1 --remove-bindfs
python3 ~/ai-infra-dev/ssh/container-dev/devtools/cli.py containers delete --cleanup-bindfs
```

残留映射清理默认交互确认，自动化可加 `--yes`。

宿主机未重启时可直接 `docker restart 容器名`。宿主机重启后，先重新执行创建时输出的 `sudo bindfs ...` 命令恢复映射，再 `docker start 容器名`；不要在恢复前启动容器，否则共享目录可能为空。容器及其内部数据仍保留，无需重建。

## 补全与测试

执行 `source ~/ai-infra-dev/completion.bash`，支持直接运行入口及 `python3` 调用的 CLI、workspace `dev/` 入口的命令、选项与主机名补全。

```bash
uv run --with pytest python -m pytest smoketest -q
```
