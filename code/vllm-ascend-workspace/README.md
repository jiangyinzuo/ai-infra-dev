## 更新源码与对齐版本

从任意目录执行（只依赖 Python 3 和 Git，不安装依赖）：

```bash
python3 ~/ai-infra-dev/code/vllm-ascend-workspace/update-vllm.py
```

首次运行克隆 `code/vllm` 和本工作区下的 `vllm-ascend`，后续仅 fast-forward 更新两个仓库的 `main`。根据 `vllm-ascend/.github/vllm-main-verified.commit` 创建或对齐本工作区下的 `vllm` worktree，保持 detached HEAD；`code/vllm` 主仓库的位置不变。

已有主仓库必须位于 `main` 且 origin 为官方 HTTPS URL（可用 Git `insteadOf` 重写为镜像）。所有受管理仓库和 worktree 必须无本地修改，包括未跟踪文件。失败可能保留已完成的更新，修复原因后可重新运行，不会重置本地提交或覆盖本地修改。

本地PC（无昇腾卡）初始化uv环境
```bash
uv venv --python 3.12 --seed --managed-python --system-certs
./uv-pip-install-for-lsp.sh
```

## 容器内构建 vllm-ascend

共享入口 `~/ai-infra-dev/code/dev` 要求 Python >=3.11，使用容器自带的 `pip`，不创建虚拟环境、不预先卸载。默认不编译算子；首次安装需加 `--compile-ops`。

`vllm install` 仍只面向 Ascend，安装 `code/vllm-ascend-workspace/vllm-ascend`，仅允许容器内执行；不提供 CUDA 安装或构建流程。`vllm serve` 可在容器、宿主机或已准备好的虚拟环境运行，按实际平台选择 NPU 或 NVIDIA GPU。实际执行的服务命令由配置中的 `serve_command` 指定，工具不会替你安装 vLLM 或运行时依赖。

源码目录不存在时交互询问仓库 URL 和分支，默认 `git@github.com:vllm-project/vllm-ascend.git`、`main`，以 `--depth=1000` 克隆后继续安装。已有目录不覆盖、不自动切分支；远程安装需要交互终端。

切换 Python 版本或镜像后，若共享的 `csrc/build` 缓存仍引用旧 Python/ninja 路径，可加 `--remove-csrc-build`。它默认关闭；启用后在 pip install 前删除当前项目的 `csrc/build`，保留 `csrc/build_cache`，会失去该目录的增量构建结果。共享源码的其他容器也会受此删除影响，勿与其他构建同时执行。

```bash
# 容器内执行（自动定位 workspace 下的 vllm-ascend）
python3 ~/ai-infra-dev/code/dev vllm install
python3 ~/ai-infra-dev/code/dev vllm install --compile-ops

# WSL 控制远程容器
python3 ~/ai-infra-dev/code/dev vllm install \
  --host suzhou-a3-130 --container jiangyinzuo-bindfs-dev1 --compile-ops
```

普通宿主机禁止执行。远程模式复用 `ssh/` 的连接和仓库 commit 检查，要求容器共享宿主机 HOME 下的仓库；先提交、推送并更新远程仓库。安装不拉取依赖、不使用 build isolation，失败会返回原退出码，不能保证保留原安装状态。

远程安装与 serve 均通过共享 SSH 接口固定登录 `jiangyinzuo`（不沿用 SSH config 的默认 User），再进入指定容器；commit 检查与 `git-repo update` 使用同一宿主机 HOME。

## 启动服务（容器、宿主机或虚拟环境）

配置文件位于 `~/ai-infra-dev/code/dev/serve_config/`，仅通过 Python import 加载，不支持 Bash。例如 `qwen3.5-0.8b.py`：

```python
from serve_config import ServeConfig

config = ServeConfig(
    device_count=1,
    serve_command=["vllm", "serve", "{model}", "--tensor-parallel-size", "1"],
)
```

```bash
python3 ~/ai-infra-dev/code/dev vllm serve --script qwen3.5-0.8b.py
python3 ~/ai-infra-dev/code/dev vllm serve --script qwen3.5-0.8b.py --model /mnt/share/weights/Qwen3.5-0.8B --devices 0
python3 ~/ai-infra-dev/code/dev vllm serve stop
# WSL 可追加 --host HOST --container NAME；--script 只传 code/dev/serve_config 下的文件名
```

dev 将 `serve_command` 参数中的 `{model}` 替换为所选模型路径或配置提供的默认模型 ID，按平台设置 `ASCEND_RT_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`，并检查所选设备数等于 `device_count`。命令直接执行，不经 shell，不额外添加 TP、端口或其他参数；工作目录为配置文件所在目录。启动前打印最终命令。配置导入会执行 Python 顶层代码，只加载可信文件；顶层只定义配置，不启动进程或占用设备。

显式 `--model` 仍只接受本地模型路径；未指定时先使用配置的 `model` 默认值，配置未提供默认值则复用 `models discovery` 搜索本机模型，使用 `~/plugged/fzf/bin/fzf` 实时模糊选择；缺少 fzf 时退回编号选择。NPU 延续 npu-top 空闲检查和全局逻辑 ID 选择。GPU 的空闲条件是 nvidia-smi 未报告 compute process 且 GPU utilization 为 0；非零显存占用不直接判忙，选择时显示显存用量，残留显示/驱动占用仍可能影响启动。`--devices` 接受 nvidia-smi 的编号或无歧义的 GPU UUID 前缀，最终用完整 UUID 设置 CUDA 可见性，不把 CUDA ordinal 当作 nvidia-smi 编号。继承的 `CUDA_VISIBLE_DEVICES` 会限制候选设备：UUID 可直接解析，数字掩码仅在已有 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 时按 PCI 顺序解析，否则拒绝猜测；工具不修改设备排序环境。空掩码或 `-1` 表示无可用 GPU。MIG、遥测不可用或同时发现 NVIDIA/Ascend 管理命令的平台歧义均拒绝启动，不悄悄切换另一平台。

服务后台运行，日志路径在启动后打印；后台进程启动不等于 API 已就绪。再次启动时询问是否停止旧服务并重启，`serve stop` 只停止本工具记录的实例，绝不按进程名批量杀进程。状态和日志按用户及容器 hostname 保存在 `/tmp/vllm-ascend-dev-*`，容器重建后不保证保留。空闲检查只涵盖容器可见信息，不提供跨用户独占分配，检查与启动之间仍有竞争窗口。

从 WSL 启动成功后，本地自动运行 `less -R +F` 跟随容器日志；需要本地安装 less 并使用交互终端。Ctrl-C 暂停实时跟随、进入浏览模式，`F` 恢复跟随；先 Ctrl-C 再按 `q` 退出查看。退出只断开日志流，不停止服务，并打印带实际 Host、容器及日志路径的重新查看命令。停止服务仍需执行 `serve stop`。不修改服务进程的输出缓冲行为，日志显示可能有延迟。

### 双卡 RTX PRO 2000 上学习 DeepSeek-V3.2 HiSparse

`deepseek-v3.2-hisparse.py` 是单卡学习配置：两张 SM120 16GiB 卡中只选择一张空闲卡，TP=1，不要求双卡通信。它使用 4 层、hidden size 256、8 个路由专家和 BF16 随机权重，保留 MLA 维度与 `index_topk=2048`，采用 `FLASHINFER_MLA_SPARSE_SM120`、`fp8_ds_mla` KV、64-token block、4096 行热缓冲和 1GiB 主机池。HiSparse 自动选择 Model Runner V2；起步使用 eager，不启用 speculative decoding，不设置服务的输出缓冲或其他运行时环境变量。

在 `code/vllm` 中使用已经准备好的 CUDA 虚拟环境（以下 `--no-sync --offline` 只防止 uv 同步依赖，不阻止 Hugging Face 联网）：

```bash
cd ~/ai-infra-dev/code/vllm
uv run --no-sync --offline python ../dev vllm serve --script deepseek-v3.2-hisparse.py
# 固定使用 nvidia-smi 编号 0；也可传完整 GPU UUID
uv run --no-sync --offline python ../dev vllm serve --script deepseek-v3.2-hisparse.py --devices 0
# 完全离线时，目录需已包含 config.json、tokenizer 等元数据
uv run --no-sync --offline python ../dev vllm serve --script deepseek-v3.2-hisparse.py --model /path/to/local/DeepSeek-V3.2 --devices 0
uv run --no-sync --offline python ../dev vllm serve stop
```

配置默认模型为 `deepseek-ai/DeepSeek-V3.2`。`--load-format dummy` 不下载完整模型权重，但仍需模型配置、tokenizer 等元数据，首次启动可能访问 Hugging Face；随机权重输出没有语言质量意义，仅适合观察调度、KV 分配/寻址、host mirror、spill 和 restore。

前提是当前解释器已安装可用于 SM120 的 CUDA vLLM 原生算子和带 sparse MLA decode API 的 FlashInfer，且未设置 `VLLM_USE_V2_MODEL_RUNNER=0`。FlashInfer 的 Python API 存在并不证明其 SM120 kernel 已构建可用；仍需兼容的 CUDA 工具链或预编译产物。已在本机单张 RTX PRO 2000 16GiB 上启动服务，并通过 2304-token 输入、2-token decode 请求；这不代表已验证 spill/restore 或真实权重精度。首次启动可能 JIT 编译数分钟；以实际日志和 API 就绪状态为准，若出现 CUDA 构建/依赖问题，不能用 Ascend 的 `install` 命令修复 CUDA 环境。

`max_model_len=8192` 只是上限，不会自动制造长输入或缓存压力。需要实际输入超过 2048 token 并继续 decode，才能观察非平凡 top-k 选择；上下文超过热缓冲也不等于一定触发页 spill。要观察 spill/restore，需要多个不同长前缀或足够缓存压力；在理解当前显存布局后再有意限制 GPU block 池或调整并发，并为每个请求的热缓冲保留容量。默认单请求与较宽裕显存设置优先让最小路径易于调试，不保证产生 spill。

## 开发工具与 LSP

```bash
# 容器内安装 pyrefly>=1.3.0、pylint（仅安装工具，不构建）
python3 ~/ai-infra-dev/code/dev devtools install

# WSL 驱动远程容器安装
python3 ~/ai-infra-dev/code/dev devtools install \
  --host suzhou-a3-130 --container jiangyinzuo-bindfs-dev1
```

依赖统一维护在 `../dev/requirements-dev.txt`（相对本 workspace）。Neovim 和 code agent 均启动 `pyrefly lsp`；项目配置通过 `../dev/lsp_python.py` 在 WSL 选择 workspace `.venv/bin/python`，在容器选择全局 Python，不创建虚拟环境。开发工具安装到容器的全局 Python 环境，与 vLLM 共享依赖环境。启动 LSP 时勿携带其他已激活的 venv/Conda 环境，以免覆盖项目解释器选择。

执行 `source ~/ai-infra-dev/completion.bash` 启用 Bash 子命令、选项及 SSH Host 补全。workspace 的 Pyrefly 分析目标为 Python 3.11，不改变 `ssh/` 工具的 Python 3.9 兼容要求。
