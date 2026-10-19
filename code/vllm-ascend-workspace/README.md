本地PC（无昇腾卡）初始化uv环境
```bash
uv venv --python 3.12 --seed --managed-python --system-certs
./uv-pip-install-for-lsp.sh
```

## 容器内构建 vllm-ascend

`dev/` 入口要求 Python >=3.11，使用容器自带的 `pip`，不创建虚拟环境、不预先卸载。默认不编译算子；首次安装需加 `--compile-ops`。

源码目录不存在时交互询问仓库 URL 和分支，默认 `git@github.com:vllm-project/vllm-ascend.git`、`main`，以 `--depth=1000` 克隆后继续安装。已有目录不覆盖、不自动切分支；远程安装需要交互终端。

```bash
# 容器内执行（自动定位 workspace 下的 vllm-ascend）
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev vllm-ascend install
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev vllm-ascend install --compile-ops

# WSL 控制远程容器
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev vllm-ascend install \
  --host suzhou-a3-130 --container jiangyinzuo-bindfs-dev1 --compile-ops
```

普通宿主机禁止执行。远程模式复用 `ssh/` 的连接和仓库 commit 检查，要求容器共享宿主机 HOME 下的仓库；先提交、推送并更新远程仓库。安装不拉取依赖、不使用 build isolation，失败会返回原退出码，不能保证保留原安装状态。

远程安装与 serve 均通过共享 SSH 接口固定登录 `jiangyinzuo`（不沿用 SSH config 的默认 User），再进入指定容器；commit 检查与 `git-repo update` 使用同一宿主机 HOME。

## 启动服务（容器内）

配置文件位于 `dev/serve_config/`，仅通过 Python import 加载，不支持 Bash。例如 `qwen3.5-0.8b.py`：

```python
from serve_config import ServeConfig

config = ServeConfig(
    device_count=1,
    serve_command=["vllm", "serve", "{model}", "--tensor-parallel-size", "1"],
)
```

```bash
python3 dev vllm-ascend serve --script qwen3.5-0.8b.py
python3 dev vllm-ascend serve --script qwen3.5-0.8b.py --model /mnt/share/weights/Qwen3.5-0.8B --devices 0
python3 dev vllm-ascend serve stop
# WSL 可追加 --host HOST --container NAME；--script 只传 dev/serve_config 下的文件名
```

dev 将 `serve_command` 参数中的 `{model}` 替换为所选路径，设置 `ASCEND_RT_VISIBLE_DEVICES`，并检查所选设备数等于 `device_count`。命令直接执行，不经 shell，不额外添加 TP、端口或其他参数；工作目录为配置文件所在目录。启动前打印最终命令。配置导入会执行 Python 顶层代码，只加载可信文件；顶层只定义配置，不启动进程或占用 NPU。

省略模型时复用 `models discovery` 搜索本机模型，使用 `~/plugged/fzf/bin/fzf` 实时模糊选择；缺少 fzf 时退回编号选择。启动前复用 npu-top 检查空闲设备，并选择可用设备。

服务后台运行，日志路径在启动后打印；后台进程启动不等于 API 已就绪。再次启动时询问是否停止旧服务并重启，`serve stop` 只停止本工具记录的实例，绝不按进程名批量杀进程。状态和日志按用户及容器 hostname 保存在 `/tmp/vllm-ascend-dev-*`，容器重建后不保证保留。空闲检查只涵盖容器可见信息，不提供跨用户独占分配，检查与启动之间仍有竞争窗口。

从 WSL 启动成功后，本地自动运行 `less -R +F` 跟随容器日志；需要本地安装 less 并使用交互终端。Ctrl-C 暂停实时跟随、进入浏览模式，`F` 恢复跟随；先 Ctrl-C 再按 `q` 退出查看。退出只断开日志流，不停止服务，并打印带实际 Host、容器及日志路径的重新查看命令。停止服务仍需执行 `serve stop`。不修改服务进程的输出缓冲行为，日志显示可能有延迟。

## 开发工具与 LSP

```bash
# 容器内安装 pyrefly>=1.3.0、pylint（仅安装工具，不构建）
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev devtools install

# WSL 驱动远程容器安装
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev devtools install \
  --host suzhou-a3-130 --container jiangyinzuo-bindfs-dev1
```

依赖统一维护在 `dev/requirements-dev.txt`。Neovim 和 code agent 均启动 `pyrefly lsp`；项目配置通过 `dev/lsp_python.py` 在 WSL 选择 workspace `.venv/bin/python`，在容器选择全局 Python，不创建虚拟环境。启动 LSP 时勿携带其他已激活的 venv/Conda 环境，以免覆盖项目解释器选择。

执行 `source ~/ai-infra-dev/completion.bash` 启用 Bash 子命令、选项及 SSH Host 补全。workspace 的 Pyrefly 分析目标为 Python 3.11，不改变 `ssh/` 工具的 Python 3.9 兼容要求。
