本地PC（无昇腾卡）初始化uv环境
```bash
uv venv --python 3.12 --seed --managed-python --system-certs
./uv-pip-install-for-lsp.sh
```

## 容器内构建 vllm-ascend

`dev/` 入口要求 Python >=3.11，使用容器自带的 `pip`，不创建虚拟环境、不预先卸载。默认不编译算子；首次安装需加 `--compile-ops`。

```bash
# 容器内执行（自动定位 workspace 下的 vllm-ascend）
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev vllm-ascend install
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev vllm-ascend install --compile-ops

# WSL 控制远程容器
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/dev vllm-ascend install \
  --host suzhou-a3-130 --container jiangyinzuo-bindfs-dev1 --compile-ops
```

普通宿主机禁止执行。远程模式复用 `ssh/` 的连接和仓库 commit 检查，要求容器共享宿主机 HOME 下的仓库；先提交、推送并更新远程仓库。安装不拉取依赖、不使用 build isolation，失败会返回原退出码，不能保证保留原安装状态。

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
