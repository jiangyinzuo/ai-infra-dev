# Python 函数追踪

记录选定函数的参数和调用栈。需要 Python 3.11+，工具自身只用标准库；目标程序需使用已安装其依赖的解释器。只运行可信的 Python 配置。

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

HiSparse 配置还采集稀疏 MLA 元数据构建、SM120 `forward_mqa` 与
`_run_mqa_kernel` 的 Python 调用入口，便于查看 Attention 后端路径；不追踪 CUDA kernel 内部。

也支持 `python -m trace_python.call_stack_analysis`。从其他目录调用时，使用工具目录和配置文件的绝对路径。

## 查看结果

输出目录需已存在。以下文件会覆盖写入，后缀直接追加到 `--output-prefix`：

| 文件 | 用途 |
|---|---|
| `hisparse.jsonl` | 程序分析的原始事件 |
| `hisparse.txt` | 逐事件查看参数和完整调用栈 |
| `hisparse.tree.json` | vat-viewer 折叠查看调用树 |
| `hisparse.tree.txt` | 紧凑文本调用树 |

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

当前只追踪当前线程的 Python 调用，会影响时序，不用于测性能；不会进入 CUDA kernel。输出重定向至 stderr，目标失败返回非零并保留已采集结果。

采集机制、输出协议、调用树与计数规则见 [DESIGN.md](DESIGN.md)。
