# Python 动态对象流分析（dynamic object-flow analysis）+ 复制来源追踪（copy-aware provenance tracking）

## 用户需求

用户想借助对象流分析技术理解阅读代码，特别是理解代码中一个对象在层层函数调用下的传播。

**NOTE：可以漏报，不能误报！**

用户给定大型python项目（如vLLM）中需要分析的对象，例如用户要求分析下面代码中的new_blocks对象：

vllm/vllm:vllm/v1/core/sched/scheduler.py:834-838
```python
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
```

程序需要给出new_blocks的对象在整个项目实际执行路径上的传播路径，格式为：

```
              ├─InprocClient.get_output (vllm/v1/engine/core_client.py) [caller:327] samples=60 events=3..64
              │ └─EngineCore.step_with_batch_queue (vllm/v1/engine/core.py) [caller:397] samples=60 events=3..64
              │   ├─Scheduler.schedule 🎯[3] (vllm/v1/core/sched/scheduler.py) [caller:708] samples=34 events=3..59
              │   │ ├─new_blocks = self.kv_cache_manager.allocate_slots(...) 🚥[1]:834
              │   │ ├─req_to_new_blocks[request_id] = new_blocks 🚩[2]:910
              │   │ ├─req_to_new_blocks[req.request_id].get_block_ids() 🚩[3]:1511
              │   │ ├─req_to_new_blocks[req.request_id].get_block_ids() 🚩[4]:1521
              │   │ ├─cached_reqs_data = self._make_cached_request_data(...req_to_new_blocks...) 🚩[5]:1528
              │   │ │ └─Scheduler._make_cached_request_data
              │   │ │   └─req_to_new_blocks[req_id].get_block_ids(allow_none=True) 🚩[6]:1822
              │   │ ├─Scheduler._get_local_prefix_cache_hit (vllm/v1/core/sched/scheduler.py) [caller:1022] samples=1 events=4..4
              │   │ │ └─KVCacheManager.get_computed_blocks 🎯[4] (vllm/v1/core/kv_cache_manager.py) [caller:596] samples=1 events=4..4
              │   │ ├─KVCacheManager.allocate_slots 🎯[5] (vllm/v1/core/kv_cache_manager.py) [caller:1313] samples=3 events=5..7
              │   │ │ └─KVCacheCoordinator.allocate_new_blocks (vllm/v1/core/kv_cache_coordinator.py) [caller:595] samples=2 events=6..7
```

其中，整体是函数调用的call tree，可以模仿现有的 ~/ai-infra-dev/trace_python/call_stack_analysis/ 中的实现。

- 🚥是追溯对象的起点。
    - 每次执行都生成独立的追踪实例，用 object_id 区分。
    - 多次起点捕获同一对象时保留多个起点记录，但共享对象节点。
    - 起点仅分析对象后续去向，不分析`allocate_slots`内部，new_blocks的来源。
- `🚩[n]` 是追溯对象的对象流事件。`[n]`代表该显示节点首次发生时的对象流事件编号，与 `🎯` 的调用采样事件编号分开。
- 示例中的 `1511`、`1521` 代表不同执行路径，不表示一次运行全部发生。

### 对象流分析，而非针对变量DEF-USE的分析

例如下面的例子中，
```python
new_blocks = allocate_slots(...)
consume(new_blocks)
new_blocks = None
```

用户真正关心的是allocate_slots产生的对象O1，而非`new_blocks`。

### 载体传播（carrier propagation）

```python
req_to_new_blocks[request_id] = new_blocks
use(req_to_new_blocks)
```
用户希望在动态分析中进一步追溯`req_to_new_blocks`，这样才能更好地理解代码。

#### 载体的语义：曾经可能持有目标，不保证当前依然持有目标

例如
```python
mapping[key] = target
mapping[key] = other
use(mapping)
```

这时 `mapping` 仍可作为历史载体展示，但不能暗示它当前仍持有目标。
这样设计的目的是减少程序维护开销。

NOTE：历史载体被写入另一载体时，只建立“载体的载体”关系，不自动把新载体标记为目标的历史间接载体。只有后续实际读取结果与已追踪对象身份匹配，才恢复目标对象流。

#### 支持的载体类型

##### 部分Python标准容器

- list
- dict
- set
- tuple
- frozenset

不追踪其它容器，保证接受漏报，但不接受误报。

##### 本项目定义的class中明确的普通实例字段

仅当运行时确认目标对象实际写入项目内实例的 `__dict__` 时，建立字段持有关系。property、自定义 `__setattr__`、descriptor、
`__slots__` 及原生扩展属性不建立关系，允许漏报。

### 赋值与拷贝（读写事件）

#### 变量
- 普通变量赋值／别名：`alias = new_blocks`

#### 容器
- 容器／属性写入：`mapping[key] = new_blocks`
- 容器／属性读取：`value = mapping[key]`
- 内置容器修改：`items.append(new_blocks)`
- 容器自身复制：`items.copy()`、`mapping.copy()`
- 构造：`list(items)`、`dict(mapping)`、`set(items)`、`tuple(items)`、`frozenset(items)`
- 完整切片：`items[:]`

约束： 构造、完整切片及 copy() 仅对支持的精确内置容器生效；容器子类和自定义 iterable 不自动处理。

#### 函数
- 函数参数 / 函数返回，包括绑定方法的实际接收者 `self`。

需要确认真实参数绑定，包括：
```python
consume(value=new_blocks)
consume(*args, **kwargs)
```

暂不支持
- 深拷贝
- 标准库copy.copy()函数。因为`copy.copy()`可能会调用用户自定义的`__copy__`函数。

#### 支持嵌套载体传播

例如
```python
inner = [target]
outer = {"inner": inner}
use(outer)
```

需要确认outer也是target的间接载体。实现方式为，只根据上面的写入事件来维护嵌套载体，不直接递归扫描整个对象。

##### 特殊场景：循环载体

```python
a.append(b)
b.append(a)
```

实现时用对象身份和边去重，避免无限扩散。

##### 特殊场景：outer 持有一个“曾经持有 target 的载体”

 ```python
inner = [target]       # inner 曾经持有 target
inner.clear()
outer = {"inner": inner}
 ```

##### 载体复制与目标对象复制

目标对象复制与载体复制分开处理。
- 直接复制目标对象，且该操作属于上述受支持的精确内置容器复制时，根据实际返回对象建立浅复制来源关系，并继续追踪副本。
- 复制候选载体时，仅记录载体之间的复制来源，不自动将副本标记为目标的候选载体；只有后续实际读取结果与已追踪对象身份匹配，才恢复目标对象流。
- 复制操作返回原对象时记录为同一对象；返回新对象时建立独立的副本节点。

### 终点定义

#### 对象进入python标准库或者项目之外的三方库后，不再追踪

例如：
```python
payload = json.dumps(target)
```

后续不再追踪`json.dumps`内部调用。
对于“赋值与拷贝（读写事件）”章节规定的容器，我们不展开其内部调用，但是更新载体维护信息。

约束： 除明确支持的容器操作外，进入标准库或第三方库即终止该路径，即使返回同一对象也不再继续；允许漏报。

#### 达到事件或资源上限后停止新增观察，保留已有结果并标记“不完整”

#### 副作用：为避免 id() 重用导致误报，通常需要保留对象强引用，这会改变 GC 时机

不要把“对象销毁”当作可精确观察的终点。

### Non-Goal

- 仅支持单线程追溯，允许漏报跨线程场景。
    - 因此，载体的语义是，历史上观测到写入，可能还带有target对象。
- 仅保证普通同步 Python 函数的参数和返回；生成器及异步执行路径允许漏报。
- 不追溯派生对象：`new_block_ids = new_blocks.get_block_ids()` 后，不继续追 `new_block_ids`。

## 技术选型
