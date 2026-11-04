# Python 函数追踪

记录选定函数的参数和调用栈。需要 Python 3.12+，工具自身只用标准库；目标程序需使用已安装其依赖的解释器。只运行可信的 Python 配置。

## 运行 HiSparse 示例

从仓库根目录执行，先用 `nvidia-smi` 确认所选 GPU 空闲：

```bash
source code/vllm-workspace/vllm/.venv/bin/activate
CUDA_VISIBLE_DEVICES=<空闲GPU的完整UUID> \
python trace_python/call_stack_analysis \
  --config trace_python/call_stack_analysis/config/hisparse.py \
  --output-prefix /tmp/hisparse \
  2> /tmp/hisparse.log
```

示例运行 dummy 模型，输入 2304 token、最多输出 8 token。启动可能需要模型元数据和 CUDA 编译工具；请激活虚拟环境，确保 ninja 等工具在 PATH 中。工具不自动选卡。

HiSparse 配置从引擎创建前启动采集，包含后端实现初始化、预热和请求执行，命中计数不能只解释为请求调用。
另外采集稀疏 MLA 元数据构建、SM120 `forward_mqa` 与 `_run_mqa_kernel` 的 Python 调用入口；不追踪 CUDA kernel 内部。

也支持 `python -m trace_python.call_stack_analysis`。从其他目录调用时，使用工具目录和配置文件的绝对路径。

## 查看结果

输出目录需已存在。以下文件会覆盖写入，后缀直接追加到 `--output-prefix`：

| 文件 | 用途 |
|---|---|
| `hisparse.jsonl` | 程序分析的原始事件 |
| `hisparse.txt` | 逐事件查看参数和完整调用栈 |
| `hisparse.tree.json` | vat-viewer 折叠查看调用树 |
| `hisparse.tree.txt` | 紧凑文本调用树 |

文本树头部记录本次 Python 执行命令和启动目录；JSON 树的 `summary` 与 JSONL 的
`run`/`summary` 记录包含 `argv`、`command`、`cwd`。命令保留 Python 启动参数与 shell 引用，
不包含 shell 重定向或 `CUDA_VISIBLE_DEVICES=...` 等 shell 环境前缀。

`.txt` 摘要与 `.tree.txt` 头部用时分秒显示运行时长；JSON/JSONL 的
`summary.elapsed_seconds` 保留数值秒。计时覆盖配置场景执行（含初始化、预热与请求），
不包含 Python 启动、配置加载或最终文件汇总写入，不用于衡量未插桩程序性能。

```bash
vat /tmp/hisparse.tree.json
less -S /tmp/hisparse.tree.txt
rg -n 'allocate_slots|swap_in|free' /tmp/hisparse.tree.txt
```

`less` 中 `/swap_in` 搜索，`n` 下一个，`q` 退出。目标日志和异常在 `/tmp/hisparse.log`。

树中 `🎯[n]` 可对应事件序号；`caller:42` 指调用者行号；`samples` 是已保存栈数，不是调用总数，完整次数查看摘要。树不代表完整时间顺序。

## 编写配置

配置文件导出 `config`。例如，追踪同目录 `subject.py` 中的 `Service.handle`：

```python
from pathlib import Path
import runpy
from trace_python.call_stack_analysis import Target, TraceConfig

ROOT = Path(__file__).resolve().parent


def run(start):
    module = runpy.run_path(str(ROOT / "subject.py"))
    service = module["Service"]()
    start()  # 初始化完成后开始记录
    service.handle("hello")


config = TraceConfig(
    root=ROOT,
    run=run,
    targets=(Target("subject.py", "Service.handle", ("request",), limit=4),),
)
```

```bash
python trace_python/call_stack_analysis --config /path/to/config.py --output-prefix /tmp/my-trace
```

- `root`：绝对源码路径，也是运行工作目录。
- `run(start)`：执行场景；调用 `start()` 后开始采集。
- `Target(file, function, fields, limit)`：相对源码路径、完整函数限定名、局部变量或属性路径、最多保存的详细事件数。超过 limit 仍计数，0 表示仅计数。
- `env`：可选的本次运行环境变量。

## 多进程入口

使用公共接口 `TraceSession(root, targets, output_prefix)`，不要把已打开的 writer 或 `TraceConfig.run` 传给子进程：

- 当前进程：`with session.capture() as tracer:`；默认立即采集，传 `active=False` 后可调用 `tracer.start()` 开始。
- 子进程：`context.Process(target=session.run, args=(worker_main, ...))`；`worker_main` 必须能被所选启动方式加载，spawn 通常要求可导入的模块级函数或静态方法。
- 每一层子进程入口都需显式包装，嵌套启动时传递同一 `session`。兼容 fork、spawn、forkserver，不改变实际启动方式。
- 安装和退出必须在进程主线程；同一进程不能嵌套采集。不要与其他调试器或按线程单独安装的 trace hook 混用。

每个进程输出 `<output_prefix>.<PID>-<实例ID>` 加上述四个后缀，计数、limit、序号和调用树独立；所有记录带 `run_id` 和进程身份，事件另带线程身份及 `monotonic_ns`。输出目录须已存在。库接口不修改 cwd、环境或 stdout，也不会执行配置中的 `run()`。

CLI 的 `run(start)` 和原有四个输出文件名保持不变；它不会自动接管配置启动的子进程。父进程的 `start()` 也不会改变已有子进程的采集开关。这不是向已运行 PID 事后 attach 的接口。

### 运行离线多进程示例

不需要 vLLM、GPU 或额外依赖。从仓库根目录执行：

```bash
python trace_python/call_stack_analysis/tests/multiprocess_smoke.py --start-method fork --output-prefix /tmp/trace-fork
python trace_python/call_stack_analysis/tests/multiprocess_smoke.py --start-method spawn --output-prefix /tmp/trace-spawn
```

示例启动 API 主进程、EngineCore 和两个 Worker，通过 pipe 传递请求及结果，输出进程身份、计算结果与文件路径。省略 `--output-prefix` 时使用临时目录，验证后自动删除。

完整入口示例见 [multiprocess_smoke.py](tests/multiprocess_smoke.py)；每个进程的栈单独查看，不把 IPC 两端拼成一条 Python 调用栈。

## 直接 attach 已运行进程（CPython 3.14+）

无需目标预先导入采集器。控制端和目标端必须使用**相同主、次版本的 CPython，且至少为 3.14**；预发布解释器要求精确版本相同。工具目录、配置中的源码路径及输出目录须在双方可见的同一文件系统命名空间中。

从仓库根目录创建独立验证环境并运行：

```bash
uv venv --python 3.14 trace_python/call_stack_analysis/.venv-attach314
trace_python/call_stack_analysis/.venv-attach314/bin/python \
  -m trace_python.call_stack_analysis.attach \
  --pid 12345 12346 \
  --config /path/to/config.py \
  --output-prefix /tmp/attached \
  --duration 10 --timeout 10
```

- 复用现有 `TraceConfig`：只在控制端 import 配置，传入 `root` 和 `targets`；不调用 `run()`、不应用 `env`。配置顶层代码仍会执行，必须可信。
- 逐个安装并等待确认，全部确认后等待 `--duration` 秒，再请求卸载、输出摘要与调用树。先安装的进程采集时间会略长；不是跨进程同步采集窗口。
- 每个 PID 独立输出四份文件；stdout 的 JSON 给出 `run_id`、实际 `output_prefix` 与 `stopped` 状态。失败返回非零，Ctrl-C 会尝试卸载已安装的采集器。
- 只处理列出的 PID，不自动发现已有或新建子进程；只采集安装后的目标函数入口，不补录历史调用，也不暂停在交互式断点。

无需 vLLM 的真实 attach 示例：运行 `python3.14 trace_python/call_stack_analysis/tests/attach_smoke.py`（可使用上面虚拟环境中的解释器）。它先启动未插桩目标，再 attach；省略输出前缀时自动清理临时文件，详细验证命令见 [DESIGN.md](DESIGN.md#离线验证)。

目标主线程必须回到 Python 安全执行点才能安装或卸载；长时间停留在原生调用或 I/O 中可能超时。超时不是“没有安装”的证明。控制端会取消迟到的 start 请求；若仍有未确认请求，会报告并保留 `.call-stack-attach-*` 控制目录，**确认请求执行完毕前不要删除它**。卸载未确认时会明确报告采集器可能仍在运行。强杀控制端不能保证卸载。

Linux attach 受 Yama/ptrace、用户身份和容器权限限制；目标也不能禁用 Python remote debugging。工具不会修改系统安全设置。只对有权调试的进程使用；attach 会向目标加载 Python 模块、安装 hooks 并写文件，改变执行时序。卸载不移除已加载模块，不改变目标 cwd、环境或 stdout。

采集覆盖当前进程中已有和后续创建的 Python 线程，会影响时序，不用于测性能；不会进入 CUDA kernel。原有受控场景 CLI 将目标输出重定向至 stderr；attach 不改变目标输出去向。

采集机制、输出协议、调用树与计数规则见 [DESIGN.md](DESIGN.md)。
