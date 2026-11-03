# 追踪工具设计

## 要解决的问题

在受控场景或显式包装的进程入口中，记录选定函数的入口状态和调用路径，既能供程序分析，也能让人快速找到关键阶段。需要 Python 3.12+。它不是全量 profiler，不记录函数返回，也不追踪 CUDA kernel。

命令行只接收配置和统一输出前缀。每次固定覆盖生成四份文件，不维护格式选择参数或旧格式兼容：

| 后缀 | 内容 |
|---|---|
| `.jsonl` | 原始事件流 |
| `.txt` | 包含字段、调用栈和摘要的可读事件 |
| `.tree.json` | 用于 vat-viewer 的紧凑嵌套调用树 |
| `.tree.txt` | 每层两列的树线文本 |

后缀直接追加，不替换前缀已有扩展名。事件流负责表达执行顺序，树负责表达调用结构，不能互相替代。

## 配置与执行边界

Python 配置导出 `TraceConfig`。配置会被 import，属于可执行代码，只能加载可信文件。

- `root` 是绝对源码根目录，也是执行场景时的 cwd。配置可从自身的 `__file__` 计算路径，不依赖启动目录。
- `run(start)` 负责初始化和运行实际场景，在初始化结束后主动调用 `start()` 开始记录。不调用就没有采集事件。
- `targets` 指定文件、完整函数限定名、采集字段和详细事件上限。
- `env` 只在执行场景期间设置，退出时恢复。

`Target.file` 相对于 root；`function` 使用 `co_qualname`，例如 `KVCacheManager.allocate_slots` 或 `outer.<locals>.inner`。不能只凭裸方法名把不同类的方法混在一起。目标不存在、匹配不唯一或重复配置时直接报错，避免无声漏采。

`fields` 只接受局部变量及属性路径，不执行表达式或方法调用。字段不存在时记录 `unavailable`，不补假值。属性读取仍可能执行 property，配置者需要理解其副作用。

HiSparse 配置复用现有 dummy 模型参数，运行 2304-token 输入、8-token 输出的离线请求。它显式使用单进程、uni executor，不改变正常 `dev serve` 行为。设备选择和空闲检查由调用者负责。

## 采集机制

核心是 `sys.settrace()` 的 Python `call` 事件；用 Python 3.12 的 `threading.settrace_all_threads()` 覆盖当前进程已有和后续创建的 Python 线程，不使用 bdb 行断点。目标源码编译但不执行，通过 code object 的文件、首行和限定名定位函数，再匹配运行时调用。因此不依赖多行 `if` 等语句的可执行行位置。

函数内部循环不会反复触发入口采集；循环反复调用函数则会。生成器和协程恢复也可能产生 call 事件，命中数不能一概等同于独立调用或请求次数。

每进程、每目标独立计算 `limit` 和命中数。`limit` 限制详细事件数量，但不停止累计命中数；0 表示只计数。`sequence` 是该进程已保存详细事件的编号，`hit` 是该进程该目标的完整命中序号。进程内锁串行更新计数、写事件和聚合树；关闭时等待正在写入的事件结束，再写摘要。

调用栈沿 `frame.f_back` 从最外层 Python 帧收集到目标函数，包含工具入口、配置和依赖。root 内文件用相对路径，外部文件保留原路径。它只是当前线程的栈，不跨进程拼接因果链。

字段快照有意限制展开范围：标量保留值，list/tuple/dict 每层最多 16 项、最多 3 层；tensor 只读取 shape、dtype、device，其他对象只记类型，不调用 repr。不自动执行 `.cpu()`、`.item()` 或消费 transfer 队列。

## 日志隔离与资源恢复

仅 CLI 在加载配置、执行目标场景时，把 Python stdout 和底层 fd 1 临时指向 stderr，避免日志混入追踪结果。原有 stderr 保持不变；库接口不重定向输出。

fd 重定向作用于整个进程，其他线程的 stdout 也会受影响。它不设置强制刷新或改模型计算，但改变了输出去向。

`configuration.py` 保存 `Target`、`TraceConfig`；`tracing.py` 负责目标匹配、字段快照和栈采集；`output.py` 负责文件与树输出；`runtime.py` 的 `TraceSession` 负责进程内安装、关闭和错误摘要；`__main__.py` 仅负责 CLI、可信配置加载及 cwd/env/stdout 上下文。CLI 正常退出或异常时恢复 cwd 和配置覆盖的环境变量。

## 进程与线程生命周期

`TraceSession` 是可 pickle 的配置对象，只包含源码根目录、目标、绝对输出前缀、会话 ID 和可选配置路径。子进程不重新 import 用户配置，也不执行 `TraceConfig.run`。`session.run(callable, *args, **kwargs)` 在当前进程创建采集器、调用业务入口并保留其返回值或异常；`session.capture()` 提供同一生命周期的上下文接口。子进程默认立即采集，父进程的延迟 start 开关不跨进程传播。

每个进程必须显式接入入口，包括孙进程。fork 后、进入包装入口之前，继承的 hook 按 PID 拒绝采集，且在触碰锁、计数和 writer 之前返回。进入包装入口后创建新的锁、writer、计数和树；spawn/forkserver 在新进程中走同一初始化路径。没有 multiprocessing 全局补丁、at-fork 自动安装或原生代码注入，也不为了采集改变业务的启动方式。

库接口按 `<prefix>.<PID>-<随机实例ID>` 创建四份输出，避免共享 writer 与 PID 复用冲突。CLI 显式覆盖前缀，保留原有四个主进程文件名。各文件只含一个进程的事件；相同 `run_id` 表示共享会话配置，不表示跨进程连续调用栈。forkserver 下 OS 父进程可能是辅助服务，不能用 PPID 冒充业务上的请求来源。

安装和退出限定在主线程，同一进程不能嵌套采集。退出时停止采集，恢复进入时的当前线程 trace hook 和 threading 默认 hook；继承自其他 PID 的本工具 hook 不会恢复。Python 接口不能取得并重建其他线程各自单独设置的不同 hook，因此不支持与这类 hook 或其他调试器同时使用。现有 profile hook 不改动。

未包装的子进程不会生成采集文件。强杀、`os._exit()` 或写入故障可能没有最终摘要和树，已刷新的 JSONL 事件仍可读取。跨进程同步启停、向任意现存 PID attach、IPC 因果链和跨机器采集不在此接口范围。


## 事件协议与失败行为

事件 JSONL 只维护当前格式，按以下顺序输出：

1. `run`：配置路径（库接口可为空）、root、解释器、启动方式、实际输出前缀和采集范围。
2. `event`：`sequence`、`target`、`hit`、字段快照 `values`、外层到内层的调用栈 `stack`、`thread_id`、`native_thread_id`、`monotonic_ns`。
3. `summary`：`status`、`error`、已保存事件数、各目标完整命中数和总耗时。

所有记录包含 `run_id`、`process_instance_id`、`pid`、`ppid`、`process_name`。事件时间戳取自当前主机的单调时钟，不是墙钟；多线程时间戳在等待写锁前采集，因此文件序号表示串行写入顺序，不保证时间戳严格递增。时间相邻不证明 IPC 因果关系。

`run` 和 `summary` 的 `started_at` 为运行开始时的本地 ISO 8601 日期时间，包含时区偏移，精确到秒。JSON 树通过 `summary.started_at` 保留该时间，文本树在头部显示 `trace_started_at`；不是生成树文件的时间。

耗时包含采集上下文内的初始化，不是函数耗时。业务异常会保留已写事件、写失败摘要及部分调用树，再原样抛出；CLI 转为非零退出码并将 traceback 写 stderr，子进程由 multiprocessing 处理未捕获异常。配置加载、目标解析、文件创建失败发生在运行前，不保证存在摘要。输出流故障也无法保证最终摘要。

## 调用树如何合并

调用树是通往已保存采集点的路径并集，不是最小生成树，也不是程序的完整执行树。没通向采集点的分支不会出现；超过 limit 后未保存的栈也不会进入树。

逐条将调用栈插入树，共同前缀合并。节点在其父路径下用文件、函数限定名和调用者行号区分。同一函数从不同路径或同一父函数的不同源码行进入，保留为不同分支。不能使用节点当前暂停行作为身份：同一函数在不同后代采集事件中会停在不同位置。

循环中的重复路径折叠，递归按实际栈深度保留。JSON 兄弟分支按子树最早采集事件排序；不移动节点来强行保证整棵树的深度优先顺序按时间递增。

标签语义：

- `函数名 🎯[n] (文件位置)`：该路径首次作为采集终点的事件序号。它可能先作为中间栈帧出现，因此 n 不一定等于该节点最早样本序号。同一目标在不同路径下可有不同 n。标记放在函数名后，避免影响同层函数名起始对齐。
- `caller:42`：调用者文件的第 42 行，不是当前节点文件的位置。
- `samples`：经过该节点的已保存栈数，不是函数总调用次数。
- `events=首序号..末序号`：范围，不保证中间每条事件都经过该节点。
- `summary.counts`：目标完整命中次数，包括超过 limit 的命中。

JSON 的 `tree` 用函数标签作键、子调用作值，叶子为 `{}`。这样 vat-viewer 展开后直接看到函数，避免每层重复 function/file/children 等字段。完整摘要放在单独的 `summary` 中。文本树保留相同语义，每层两列，减少深调用链的横向占用。

路径节点在内存聚合，运行结束时输出树；事件文件则逐条写出。树为空时仍保留摘要，不能把空树解释为程序没有执行。

## 能证明什么

追踪覆盖显式接入进程中的 Python 线程；不自动接管任意子进程、现存 PID、远端或 CUDA kernel。Python tracing 和文件写入会改变时序，不用于性能结论。

CPU 侧函数进入顺序不等于 GPU 工作完成顺序。命中 `swap_in` 不证明实际发生非空 H2D；进入 DMA 提交函数也不等于测得传输字节或带宽。需要进一步采集状态或使用专门的 GPU 分析工具才能回答这些问题。

## 离线验证

`tests/multiprocess_smoke.py` 用真实 pipe 和 `multiprocessing.get_context(method).Process` 模拟 API → EngineCore → 两个 Worker；业务入口采用与 vLLM 类似的静态进程入口，所有入口显式经 `session.run` 包装。不加载 vLLM，不使用 GPU。

验证业务结果、PID/PPID 与进程名、每进程计数及 limit、原始字段及真实栈、独立输出和树摘要；包括安装前已有的线程、安装后创建的线程，以及 fork 后包装入口前的目标调用。pytest 另覆盖 CLI 入口、cwd/env 恢复、异常退出、hook 恢复和子进程失败隔离。

从仓库根目录执行：

```bash
uv run --no-project --with pytest --python 3.12 python -m pytest trace_python/call_stack_analysis/tests -q
```

