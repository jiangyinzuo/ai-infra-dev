# 实现、支持边界与验证

## 技术选择

依赖 `bytecode==0.19.1` 系列符号指令 API，处理跳转布局、栈大小和异常表组装。不手写 co_code，不做 AST 改写。AST 仅用于 CLI 起点语法验证。`sys.monitoring` 提供同步 Python 参数、返回、动态调用编号和外部 C 调用边界。

- `protocol.py`：插桩与运行时共享的轻量位置协议，不在探针热路径创建位置对象。
- `backend.py`：CodeTransformer 按绑定、属性、下标、容器、调用分组转换；保留 NULL 和异常区间规则。
- `runtime.py`：有界身份注册、强引用、捕获、历史关系和事件；协调观察组件。
- `monitoring.py`：线程与项目边界、真实参数／返回、动态调用身份及 monitoring 资源恢复。
- `operations.py`：原操作仅执行一次，成功后解释内置容器和普通字段证据；方法白名单集中管理。
- `loading.py`：普通 SourceFileLoader 模块首次导入时转换，不读写转换后的 pyc。
- `tree.py`：独立的源码展示、聚合与调用树渲染；原始事件保留动态 call_id，合并树不证明连续执行或跨调用因果。
- `cli.py`：配置校验、运行环境与文件输出；`__main__.py` 只调用入口。

文本树中的参数事件使用事件栈中调用者的源码行，而不是被调函数入口的 def 行；原始 JSONL 仍保留采集位置。每个合并调用路径内按源码文件和行号聚合叶子，跨对象、事件类别和重复执行累计，显示各类别数量、对象编号并集及首末事件编号。不同调用路径、不同调用位置不合并；事件编号范围不表示连续命中，类别数量不是调用次数。缺少调用者帧时保留原采集位置。

多行赋值、表达式与 return 通过 AST 只定位语句范围，再截取原文件源码，保留换行、注释、引号和语句内部相对缩进；去掉语句首行的外层缩进，由树前缀定位。事件编号、位置与统计放在语句首行，后续源码行保留树线。源码只在渲染时按文件解析，不增加采集开销；位置字段置于事件类型与 objects 之前。

普通 CALL 保留原指令；运行时只对精确内置容器构造和已知方法选择代理。这样别名调用 locals/eval、零参数 super 不被统一调用包装破坏。方法加载仍会通过属性 helper，栈反射可见额外帧，不承诺任意反射程序透明。

Python 3.13/3.14 的调用布局由运行解释器编译的微型模板取得；融合局部指令拆为等价单指令。3.14 的 CALL_FUNCTION_EX 空 kwargs 使用 NULL，进入 Python 参数元组前必须转成空 dict。3.12+ 内联推导式恢复局部槽位也可能携带 NULL，因此对应局部绑定不插观察，保留原 STORE。

## 实际覆盖

- 普通局部／闭包／全局／名字存储观察，指定普通赋值起点捕获。
- 下标读取、写入、完整切片；切片赋值不把 RHS iterable 本身当作写入值。
- 普通项目实例字段；排除自定义属性机制、descriptor、slots。
- list/tuple/set/dict 字面量，以及 LIST_APPEND/SET_ADD/MAP_ADD 推导式操作。
- 精确内置容器的支持构造、copy()、完整切片；目标副本与载体副本分开。
- list append/insert/extend/pop、set add/update/pop、dict update/get/pop/setdefault 的已知操作。
- 参数包括 self、关键字和展开参数，函数返回；资源上限不停止业务执行。

dict get/pop/setdefault 的实际返回身份记录为 builtin_result(operation=方法名)，包括默认值返回；不将其自动认定为字典取出来源。setdefault 仅在 key 或 default 与已追踪对象有关时保留调用前直接 key 引用，成功后以 key 身份确认新增项并记录持有关系，不额外调用 key 的 hash/eq。字典子类不套用这些适配器。失败调用不提交成功事件；历史载体返回仍标明历史角色。

精确 list.pop（包括指定索引）与 set.pop 同样按实际返回身份记录 builtin_result，支持方法别名和参数展开；结果被丢弃也会观察。空集合、空列表、越界等异常保留原语义，不提交成功事件；不新增持有关系或推广子类支持。

`LOAD_SUPER_ATTR`、`LIST_EXTEND`、`SET_UPDATE`、`DICT_UPDATE`、`DICT_MERGE` 的直接 opcode 持有关系未采集，保留原执行并在摘要报告位置。普通赋值观察、方法 append 等不依赖这些操作的来源证据。隐式或无法证明的调用协议保留原调用，相关复制可能漏报。不支持生成器／协程、跨线程／进程、原生扩展内部、动态 exec 的转换。

载体边表示实际观察过的直接写入，按对象与边去重；不沿历史边做传递闭包。载体的载体保留 `carrier_relation`，不宣称已证实目标间接持有；候选载体复制记录 `carrier_copy`。这里遵守需求中的历史关系约束，不把历史可达性当作当前持有证据。

事件种类与对象 role 都保留在 JSONL。`shallow_copy` 只表示目标对象或其直接复制链；`carrier_copy` 不自动推广目标关系。到事件／对象上限后设 incomplete，业务原操作继续执行。所有新增对象都有强引用；对象身份不使用用户比较或 repr。

## 验证

2026-10-25，实际执行同一组 28 项行为回归：

| 解释器 | 结果 |
|---|---|
| CPython 3.12.15 | 28 passed |
| CPython 3.13.16 | 28 passed |
| CPython 3.14.8 | 28 passed；测试的 finally-return 语法产生解释器 SyntaxWarning |

覆盖身份别名、参数展开、self、成功／失败写入、求值顺序、循环推导式、异常 finally、普通字段、descriptor 排除、直接复制／载体复制、历史载体、循环载体、外部返回后重新识别、外部 C 回调隔离、帧敏感函数及其别名、上限、CLI 与失败输出。

实际命令运行 examples/config.py，生成 `/tmp/object-flow-example.jsonl` 及文本树，捕获起点并记录读写、参数和副本。不保证任意 Python 程序语义保持。

### 真实 vLLM GPU 场景

2026-10-25 使用 vLLM `0.32.1rc1.dev281+g49dbf7c44`、CPython 3.12.15、RTX PRO 2000 Blackwell，运行 `examples/vllm_hisparse.py`。复用已有 DeepSeek-V3.2 缩减随机权重配置，2304 输入 token、最多 8 输出 token、uni executor、eager 与 HiSparse。仅转换 scheduler.py、kv_cache_manager.py 和 sched/output.py，不是全项目插桩验证。

真实请求输出 `REQUEST_RESULT object-flow-hisparse generated_tokens 8 finish_reason length`，退出码 0；保存 91 个事件、11 次起点捕获、6 个不同目标身份、11 条写入关系，11995 次语义探针，`incomplete=False` 仅表示未触发资源上限，不表示没有漏报。实际路径包括 `Scheduler.schedule:834` 捕获、`:910` 写入、`_make_cached_request_data` 载体参数、`:1822` 读取及 `KVCacheBlocks.get_block_ids` 的 self 参数。

最初配置 run 位于项目 root 外，函数边界过滤使采集为空；现在显式允许配置入口作为项目调用的父帧，并以 root 外配置回归覆盖。后续起点可能重复捕获同一个 empty_kv_cache_blocks；它在未来 allocate_slots 调用中出现属于对象后续观察，不是回溯最初分配来源。

首次模型初始化触发 FlashInfer 缓存重建，旧 Ninja 依赖仍指向迁移前 code/vllm/.venv。缓存重建后的成功运行总耗时 123.79 秒，engine 初始化日志 24.09 秒；未做不插桩对照，不能据此计算 vLLM 追踪开销。未验证 torch.compile。运行后释放 GPU 和 HiSparse host pool。

### 跨模块 Request

`examples/vllm_request.py` 复用同一个真实 GPU 场景，以 `EngineCore.preprocess_add_request` 中 `req = Request.from_engine_core_request(...)`（core.py:1128）为起点，扩大转换范围至 engine core/client、Request、scheduler、KV cache 与 HiSparse。真实请求生成 8 个 token，退出码 0，873 条事件、1 次捕获（Request O1）、18 条关系、392161 次探针，未触发资源上限。直接目标事件覆盖 18 个源码文件，其中范围外文件的参数／返回由 monitoring 观察，不能等同为这些文件内部已全面插桩。

已确认路径包括 EngineCore → InprocClient → Scheduler → FCFSRequestQueue，进入 KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager/BlockPool 及 HiSparseCoordinator；完成后进入 Scheduler._free_request、HiSparseConnector、EncoderCacheManager.free 与 KVCacheManager.free。新增 dict.get 在 Scheduler.update_from_output:2172 实际记录 12 次返回目标，dict.pop 在 HiSparseConnector.request_finished_all_groups:407 实际记录 1 次目标返回。

FCFSRequestQueue 是 deque 子类，按容器支持边界不记录内部持有，但其 Python add/peek/pop 方法的实际参数与返回可观察，后续身份匹配恢复。删除 self.requests 的操作不是对象销毁终点；强引用持续保留。总运行 236.06 秒，包含更大范围导入转换和模型初始化，无基线对照，不作为性能结论。GPU 进程和 HiSparse host pool 均在结束后释放。

职责拆分后重新执行同一真实 Request 场景，生成 8 个 token、873 条事件，退出码 0，无资源截断。与拆分前 JSONL 按事件逐条比较，除运行相关的动态 call_id 外所有事件字段完全一致；输出为 `/tmp/vllm-request-refactored.jsonl` 及 `.tree.txt`。这验证此次结构重构没有改变该真实路径的采集事实，不作为任意程序等价证明。

## 性能测量与结论

CPython 3.12.15，目标函数先捕获 object()，随后执行 50,000 次无关 `mapping[i] = i` 并返回长度。预热后分别独立执行 7 次，取中位数；不计转换时间，不计最后文件输出。只保存 1 个起点事件，完整观察执行 100,004 次语义探针（循环变量绑定和写入为主）。

| 模式 | 中位数 | 相对基线 |
|---|---:|---:|
| 原程序 | 2.17 ms | 1.00× |
| 只启用 monitoring | 2.42 ms | 1.12× |
| 字节码完整观察 | 137.76 ms | 63.58× |

这是纯 Python 热循环的探针成本，不代表模型推理端到端减速比例。即使不命中目标，探针调用、操作数打包和 monitoring 仍有成本。

结论：这份可运行实现证明了对象流采集和跨版本规则，但没有达到“全项目密集插桩低开销”的目标。应先按 include 限制研究范围，不应该据此宣称全项目 vLLM 运行开销很小。要决定是否继续采用该后端，需要真实场景测量，而不是仅凭成熟依赖库的存在推断性能。
