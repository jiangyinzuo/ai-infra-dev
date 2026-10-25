# Python 对象流观察

使用 `bytecode` 改写内存中的代码对象，配合 `sys.monitoring` 记录真实对象身份、函数参数／返回、容器读写和浅复制。源文件与正常 pyc 缓存不改写。

## 运行

从仓库根目录执行：

```bash
uv venv --python 3.12 trace_python/object_flow_analysis/.venv
uv pip install --python trace_python/object_flow_analysis/.venv/bin/python -e './trace_python/object_flow_analysis[test]'
trace_python/object_flow_analysis/.venv/bin/python -m trace_python.object_flow_analysis \
  --config trace_python/object_flow_analysis/examples/config.py \
  --output /tmp/object-flow.jsonl
less -S /tmp/object-flow.jsonl.tree.txt
```

需要普通带 GIL 的 CPython 3.12–3.14。目标场景依赖必须安装在同一个解释器环境中。工具不选择 GPU、不修改模型参数。仅运行可信配置。

输出 JSONL 和 `.tree.txt` 均覆盖写入。目标输出（包括 fd 1）转到 stderr；场景抛异常仍输出已有事件，再以非零退出。没有捕获起点时退出码为 2。JSONL 最后一条为摘要，包含对象、捕获编号、资源停止原因和未覆盖指令位置。

容器观察默认开启；命令行追加 `--no-observe-containers` 可关闭，`--observe-containers` 显式开启。也可在 FlowConfig 设置 `observe_containers=False`，命令行优先。关闭后只保留起点、变量绑定和同步函数参数／返回，不插入容器与实例字段读写、构造、复制或内置方法代理探针，也不会发现历史载体和副本。之后在绑定或函数边界实际出现原目标时仍可识别。此模式仍有绑定探针和监控开销，不等于零插桩。

### 真实 vLLM HiSparse 示例

需要现有 vLLM 环境与空闲 GPU；先激活环境，使已安装的 ninja 也进入 PATH：

```bash
source code/vllm-workspace/vllm/.venv/bin/activate
uv pip install --python "$VIRTUAL_ENV/bin/python" 'bytecode==0.19.1'
CUDA_VISIBLE_DEVICES=<空闲GPU的UUID> python -m trace_python.object_flow_analysis \
  --config trace_python/object_flow_analysis/examples/vllm_hisparse.py \
  --output /tmp/vllm-object-flow.jsonl
less -S /tmp/vllm-object-flow.jsonl.tree.txt
```

该场景使用已有缩减随机权重配置，不下载完整模型权重；模型元数据和 CUDA 内核仍须可用。FlashInfer 缓存不可复用时可能触发编译。仅转换 scheduler、KV cache manager 与 scheduler output；真实请求已验证产生 8 个 token 和 91 个对象流事件，不代表全项目覆盖或 torch.compile 兼容。

跨模块 Request 示例：将 `--config` 改为 `trace_python/object_flow_analysis/examples/vllm_request.py`，输出可设 `/tmp/vllm-request-flow.jsonl`。从 EngineCore 创建 Request 后追踪同一身份，经请求队列、Scheduler、KV cache、HiSparse 到请求完成清理；已实际生成 8 个 token、873 条事件。该配置扩大了转换范围，初始化和观察开销也更高。

## 编写配置

```python
from pathlib import Path
from trace_python.object_flow_analysis import FlowConfig, Origin


def run():
    # 必须在这里首次导入项目模块；配置加载时不导入目标。
    from my_project.scenario import execute
    execute()


config = FlowConfig(
    root=Path('/absolute/project'),
    origin=Origin('my_project/scenario.py', 'Scheduler.schedule', 834, 'new_blocks'),
    run=run,
    include=None,  # None 转换整个 root；也可指定相对文件／目录元组。
    max_events=10000,
    max_objects=10000,
    observe_containers=True,
)
```

`line` 为原源码普通赋值语句的起始行，`function` 为完整限定名。起点每次执行独立捕获，相同对象共享对象节点。只跟踪赋值后的去向。当前入口执行 `run()`，不是自动启动或接管已有服务。源码须通过普通源码模块导入，不能用 `runpy` 绕过导入转换。

## 读取结果与限制

- 原对象、直接浅副本、历史载体、载体之间的关系分别记录。历史载体不保证当前仍持有目标，嵌套历史关系不能证明当前间接持有。
- 不展开项目外函数或建立其返回边；之后项目代码实际读到相同对象时可以重新记录。
- 单线程、同步 Python 路径；不跟踪 GPU kernel、子进程、生成器和协程。不是性能 profiler。
- 强引用改变 GC 时机；钩子增加调用帧和开销，栈反射可能看到工具帧。容器密集热循环可能显著变慢，建议先用 `include` 限定阅读范围。
- 已导入模块不会自动补插桩；自定义 import loader、动态 exec、框架编译路径不保证覆盖。不保证与 `torch.compile` 兼容。
- 精确内置 list/dict/set/tuple/frozenset 支持已实现的读写、构造与浅复制；子类、自定义 iterable、深复制不自动推广。普通实例字段排除 descriptor、slots 和自定义属性机制。
- 上限停止的是新增观察，不停止业务执行。事件目前有界保存在内存，运行结束输出；进程被强制终止时不保证保存。

实现边界与测量见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。用户需求原文保持在 [DEGISN.md](DEGISN.md)。
