# Vector add AIV kernels

Copied from `asc-devkit-9.1.0-examples/examples/01_simd_cpp_api/00_introduction/01_add/` for profiling two Ascend C vector-add implementations.

- `add_static.asc`: static `LocalMemAllocator` buffers; reads the shared binary inputs.
- `add_tpipe_tque.asc`: `TPipe`/`TQue` buffers; reads the same shared binary inputs.
- `gen_data.py`: creates the shared random float32 inputs and golden output.
- `data_utils.h`: binary input/output helpers used by the TPipe/TQue example.


## 编译

先配置 CANN 环境，再使用 Bisheng 直接编译，默认架构为 `dav-2201`。编译阶段使用 `-O2` 优化选项：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./build.sh
```

脚本分别编译并链接两个 `.asc` 文件，不使用 CMake。可通过环境变量指定 Bisheng 路径或架构：
```sh
BISHENG=/path/to/bisheng CMAKE_ASC_ARCHITECTURES=dav-3510 ./build.sh
```

生成的可执行文件位于 `build/add_static/add_static` 和 `build/add_tpipe_tque/add_tpipe_tque`。
Both examples define their own `main` and `add_custom`; build and profile them separately. The TPipe/TQue executable expects to run from this directory after generating input data:

```sh
python3 gen_data.py
```

This creates `input/input_x.bin`, `input/input_y.bin`, and `output/golden.bin`. Running `add_tpipe_tque` writes `output/output.bin`; compare that file against `output/golden.bin` to validate results.


## 重复 profiling

`profile.sh` 会重新生成固定 seed 的输入、用当前 `build.sh`（`-O2`）重新编译、先各运行一次验证输出，然后交错采集多个 profile。奇数轮先测 static，偶数轮先测 TPipe/TQue，降低固定执行顺序带来的偏差：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
REPS=5 ./profile.sh
```

每轮的 `Task Duration(us)` 位于生成目录的 `durations.tsv`，均值、最小值和最大值位于 `summary.tsv`。每轮完整的 `msprof` 数据也保留在同一目录。可用 `REPS=10` 增加重复次数。脚本每轮都会比较两个程序的输出与 `output/golden.bin`，结果不一致会立即退出。

这个流程比较的是单次 kernel task duration，不是包含 ACL 初始化、内存申请和 Host/Device 拷贝的进程总时间。两种实现使用相同输入、数据规模、架构、`-O2` 编译选项和 8 个 block；但每次独立进程仍会重新初始化设备，因此结果应以多轮分布而不是单次最小值判断。

本机执行 `REPS=5 PROFILE_ROOT=profiles/measurement_20261028 ./profile.sh` 的实测结果如下；频率为 1800 MHz、Block Dim 为 8：

| kernel | 次数 | 平均 Task Duration | 最小 | 最大 |
|---|---:|---:|---:|---:|
| static | 5 | 1.772035 us | 1.680034 us | 1.860037 us |
| tpipe_tque | 5 | 1.944039 us | 1.840037 us | 2.040041 us |

在这组固定输入和 `-O2` 编译条件下，TPipe/TQue 平均比 static 慢约 9.7%。这是单次 kernel task duration 的结果，不代表包含 Host/Device 拷贝的端到端耗时；原因分析仍应结合对应 profile 目录中的 `PipeUtilization.csv` 等指标。

## 分块流水线对照实验

原始 `add_static.asc` 和 `add_tpipe_tque.asc` 原样保留，仍可用 `./profile.sh` 比较。新的一对实现使用相同的 8 个 AIV、每核 16 × 1024 个 float32、相同的 `-O2` 和 `dav-2201` 编译参数，对每块计算 `z = (x + y) * 1.25f`：

- `add_pipeline_static.asc`：单组 UB 缓冲，逐块 `CopyIn → PipeBarrier<PIPE_ALL> → Add/Muls → PipeBarrier<PIPE_ALL> → CopyOut → PipeBarrier<PIPE_ALL>`；这是**刻意串行的朴素基线**，不是最优化的静态缓冲实现。
- `add_pipeline_tpipe_tque.asc`：输入/输出队列 `TQue<..., 1>`，但 `InitBuffer(..., 2, ...)` 分配两组片上缓存；每块依次 `EnQue → DeQue → FreeTensor`，以队列同步替代每块的全局 `PIPE_ALL` 屏障。这不是显式预取两块的实现，不应把性能收益全部归因于双缓冲。

输入由 `gen_pipeline_data.py` 使用固定随机种子生成，`pipeline_output/golden.bin` 为共同的真值。运行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
REPS=5 ./profile_pipeline.sh
```

脚本重编译全部四个实现、检查两个新程序的输出逐字节等于 golden，奇偶轮交替采集 `msprof op`，逐次校验输出，保存 `durations.tsv`、`summary.tsv` 及原始 profile。用 `PROFILE_ROOT=profiles/my_run` 可以指定新结果目录；目录已存在时脚本拒绝覆盖。单次执行默认 120 秒超时（`PROFILE_TIMEOUT` 可调）。此前尝试连续两次 `EnQue` 的预取版本在用户设备上出现卡住，已撤销，不能使用那组版本的 profile 作为结论。

**解释范围：**若 TPipe/TQue 在这里胜出，证明的是“此流水线实现优于这个串行基线”，不是“TPipe/TQue 永远优于静态缓冲”。静态实现也可以用双缓冲、精细的事件同步和流水调度达到类似重叠效果；`num=2` 本身也不保证自动获得最大吞吐。分析重叠和开销时看各次 `PipeUtilization.csv` 的 MTE2/MTE3/Vector/Scalar 时间，单凭 Task Duration 无法证明具体是哪条指令导致加速。

安全队列版本实测：`REPS=3 PROFILE_ROOT=profiles/pipeline_safe_20261028 ./profile_pipeline.sh`；两边与 golden 逐字节相同，报告的频率为 1800 MHz，Block Dim 为 8。

| 实现 | 轮数 | 平均 Task Duration (us) | 最小 (us) | 最大 (us) |
|---|---:|---:|---:|---:|
| 串行 static | 3 | 10.633546 | 10.420208 | 10.900218 |
| TPipe/TQue | 3 | 5.193437 | 5.080101 | 5.380107 |

这组单次 kernel task 耗时中，队列实现相对串行基线缩短约 51.2%；原始 profile 位于 `profiles/pipeline_safe_20261028/`。不能推断差值完全来自双缓冲：串行版本每块包含三个全流水线屏障，队列版本没有。
