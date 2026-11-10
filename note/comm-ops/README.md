# 安装环境


本文由 [README.qmd](README.qmd) 执行 Python
脚本后生成，请修改源文件后重新渲染，不要手改 `README.md`。

``` bash
repo_root="$(git rev-parse --show-toplevel)" &&
uv pip install --python "$repo_root/note/.venv/bin/python" \
  torch --torch-backend=auto
```

## 运行 CPU 集合通信示例

复用 `code/vllm-workspace/vllm/.venv`，不再为笔记单独创建环境或安装
PyTorch。以下命令从本仓库任意子目录执行，独立嵌套的 Git 仓库除外。

先在当前终端激活虚拟环境：

``` bash
repo_root="$(git rev-parse --show-toplevel)" &&
source "$repo_root/code/vllm-workspace/vllm/.venv/bin/activate"
```

然后在同一终端运行。单机 4 个 CPU 进程（rank 0–3），依次运行全部算子：

``` bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  "$repo_root/note/comm-ops/cpu_collectives.py"
```

实际运行输出（渲染本文时执行）：

    *****************************************
    Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
    *****************************************
    [broadcast] rank=0: input=[1, 10], output=[1, 10]
    [broadcast] rank=1: input=[2, 20], output=[1, 10]
    [broadcast] rank=2: input=[3, 30], output=[1, 10]
    [broadcast] rank=3: input=[4, 40], output=[1, 10]
    [reduce] rank=0: input=[1, 10, 2, 20, 3, 30, 4, 40], output=[10, 100, 20, 200, 30, 300, 40, 400]
    [reduce] rank=1: input=[2, 20, 4, 40, 6, 60, 8, 80], output=仅 rank=0 接收结果
    [reduce] rank=2: input=[3, 30, 6, 60, 9, 90, 12, 120], output=仅 rank=0 接收结果
    [reduce] rank=3: input=[4, 40, 8, 80, 12, 120, 16, 160], output=仅 rank=0 接收结果
    [all_reduce] rank=0: input=[1, 10, 2, 20, 3, 30, 4, 40], output=[10, 100, 20, 200, 30, 300, 40, 400]
    [all_reduce] rank=1: input=[2, 20, 4, 40, 6, 60, 8, 80], output=[10, 100, 20, 200, 30, 300, 40, 400]
    [all_reduce] rank=2: input=[3, 30, 6, 60, 9, 90, 12, 120], output=[10, 100, 20, 200, 30, 300, 40, 400]
    [all_reduce] rank=3: input=[4, 40, 8, 80, 12, 120, 16, 160], output=[10, 100, 20, 200, 30, 300, 40, 400]
    [reduce_scatter] rank=0: input=[1, 10, 2, 20, 3, 30, 4, 40], output=[10, 100]
    [reduce_scatter] rank=1: input=[2, 20, 4, 40, 6, 60, 8, 80], output=[20, 200]
    [reduce_scatter] rank=2: input=[3, 30, 6, 60, 9, 90, 12, 120], output=[30, 300]
    [reduce_scatter] rank=3: input=[4, 40, 8, 80, 12, 120, 16, 160], output=[40, 400]
    [gather] rank=0: input=[1, 10], output=[[1, 10], [2, 20], [3, 30], [4, 40]]
    [gather] rank=1: input=[2, 20], output=仅 rank=0 接收结果
    [gather] rank=2: input=[3, 30], output=仅 rank=0 接收结果
    [gather] rank=3: input=[4, 40], output=仅 rank=0 接收结果
    [all_gather] rank=0: input=[1, 10], output=[[1, 10], [2, 20], [3, 30], [4, 40]]
    [all_gather] rank=1: input=[2, 20], output=[[1, 10], [2, 20], [3, 30], [4, 40]]
    [all_gather] rank=2: input=[3, 30], output=[[1, 10], [2, 20], [3, 30], [4, 40]]
    [all_gather] rank=3: input=[4, 40], output=[[1, 10], [2, 20], [3, 30], [4, 40]]
    [scatter] rank=0: chunks=[[1, 10], [2, 20], [3, 30], [4, 40]]
    [scatter] rank=0: output=[1, 10]
    [scatter] rank=1: output=[2, 20]
    [scatter] rank=2: output=[3, 30]
    [scatter] rank=3: output=[4, 40]
    [all_to_all] rank=0: input=[0, 1, 2, 3], output=[0, 4, 8, 12]
    [all_to_all] rank=1: input=[4, 5, 6, 7], output=[1, 5, 9, 13]
    [all_to_all] rank=2: input=[8, 9, 10, 11], output=[2, 6, 10, 14]
    [all_to_all] rank=3: input=[12, 13, 14, 15], output=[3, 7, 11, 15]

输出按算子分组，组内按 rank 从小到大排序。每个算子结束后额外汇总日志，由
rank 0
统一打印；单独运行一个算子也会汇总。这段演示流程不用于性能测量，显示顺序不代表实际完成顺序。

`reduce`、`all_reduce`、`reduce_scatter` 使用相同输入，每个 rank
的向量包含 4 块、每块 2 个元素。`reduce` 仅 rank 0 得到完整结果
`[10, 100, 20, 200, 30, 300, 40, 400]`，`all_reduce` 的所有 rank
都得到该完整结果，`reduce_scatter` 的每个 rank 只得到其中对应的 2
元素块。

只运行一个算子（例如 `all_to_all`）：

``` bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  "$repo_root/note/comm-ops/cpu_collectives.py" --op all_to_all
```

只运行 `reduce_scatter`：

``` bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  "$repo_root/note/comm-ops/cpu_collectives.py" --op reduce_scatter
```

该环境的 PyTorch 即使带 CUDA，也不会让本脚本使用 GPU。换环境时需确保
PyTorch 支持 `torch.distributed` 和 Gloo。

## 验证 AllReduce = ReduceScatter + AllGather

先按上文在当前终端设置 `repo_root` 并激活虚拟环境，再运行：

``` bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  "$repo_root/note/comm-ops/verify_all_reduce_decomposition.py"
```

脚本固定使用 4 个 CPU rank、Gloo 和 int64。输入按 rank 堆叠为 4×8
矩阵，每个 rank 只持有一行（长度为 8 的张量），行值依次为 `[1, …, 8]` 的
1、10、100、1000 倍。逐步按 rank 打印实际输入、分块与输出：AllReduce
得到完整的 8 元素按列求和结果；ReduceScatter 每个 rank 得到 2
个元素；AllGather 再拼成完整结果。最后逐 rank
检查两条路径相等，并校验结果为
`[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]`；不相等时以异常退出。

黄色文字是原理解释，算子日志是实际运行数据。日志复用
`cpu_collectives.py`
的汇总打印函数，额外日志通信不用于性能测量；整数验证不代表浮点规约一定逐位相等。

实际运行输出（渲染本文时执行）：

    *****************************************
    Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
    *****************************************
    输入按 rank 堆叠为 4×8 矩阵，每个 rank 只持有自己的 1 行（8 个元素）。规约将各行的同一列相加，不是对每行内部的元素求和。
    [input] rank=0: matrix_row=[1, 2, 3, 4, 5, 6, 7, 8]
    [input] rank=1: matrix_row=[10, 20, 30, 40, 50, 60, 70, 80]
    [input] rank=2: matrix_row=[100, 200, 300, 400, 500, 600, 700, 800]
    [input] rank=3: matrix_row=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    1. AllReduce(sum)：将 4 行按列求和，每个 rank 接收完整的 8 元素结果。
    [all_reduce] rank=0: input=[1, 2, 3, 4, 5, 6, 7, 8], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [all_reduce] rank=1: input=[10, 20, 30, 40, 50, 60, 70, 80], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [all_reduce] rank=2: input=[100, 200, 300, 400, 500, 600, 700, 800], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [all_reduce] rank=3: input=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    2. ReduceScatter(sum)：每行沿第 0 维分为 4 块，每块 2 个元素；逐元素求和后，rank i 只接收第 i 块。
    [reduce_scatter] rank=0: input=[1, 2, 3, 4, 5, 6, 7, 8], input_chunks=[[1, 2], [3, 4], [5, 6], [7, 8]], output=[1111, 2222]
    [reduce_scatter] rank=1: input=[10, 20, 30, 40, 50, 60, 70, 80], input_chunks=[[10, 20], [30, 40], [50, 60], [70, 80]], output=[3333, 4444]
    [reduce_scatter] rank=2: input=[100, 200, 300, 400, 500, 600, 700, 800], input_chunks=[[100, 200], [300, 400], [500, 600], [700, 800]], output=[5555, 6666]
    [reduce_scatter] rank=3: input=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], input_chunks=[[1000, 2000], [3000, 4000], [5000, 6000], [7000, 8000]], output=[7777, 8888]
    3. AllGather：输入是上一步实际得到的 shard；按来源 rank 收集并拼接，不再求和。
    [all_gather] rank=0: input=[1111, 2222], received_chunks=[[1111, 2222], [3333, 4444], [5555, 6666], [7777, 8888]], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [all_gather] rank=1: input=[3333, 4444], received_chunks=[[1111, 2222], [3333, 4444], [5555, 6666], [7777, 8888]], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [all_gather] rank=2: input=[5555, 6666], received_chunks=[[1111, 2222], [3333, 4444], [5555, 6666], [7777, 8888]], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [all_gather] rank=3: input=[7777, 8888], received_chunks=[[1111, 2222], [3333, 4444], [5555, 6666], [7777, 8888]], output=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    [verify] rank=0: all_reduce=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], reduce_scatter_then_all_gather=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], equal=True
    [verify] rank=1: all_reduce=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], reduce_scatter_then_all_gather=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], equal=True
    [verify] rank=2: all_reduce=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], reduce_scatter_then_all_gather=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], equal=True
    [verify] rank=3: all_reduce=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], reduce_scatter_then_all_gather=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888], equal=True
    验证通过：4 个 rank 的两条路径结果均为 [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]。本例使用 int64 精确比较；浮点规约可能因运算顺序不同产生舍入差异。

## 重新生成本文

在已设置 `repo_root` 的终端执行。Quarto 使用现有 PyTorch
虚拟环境，每次重新运行两个脚本，将结果写入同目录下的
`README.md`；任一脚本失败则渲染报错。

``` bash
QUARTO_PYTHON="$repo_root/code/vllm-workspace/vllm/.venv/bin/python" \
  quarto render "$repo_root/note/comm-ops/README.qmd"
```
