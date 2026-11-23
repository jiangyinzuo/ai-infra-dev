## 昇腾NPU不建议Block数量远超AICore数量

以向量算子为例，

昇腾950 NPU的编程模型中，逻辑上的Block数量(1-65535)可以超过物理Vector Core(56/64个)的数量。

但是，当逻辑Block数量多于物理Vector Core时，超出物理核数会进入等待队列，由硬件调度器进行分时复用：
- 最大并发数受限于物理核数：任一时刻，正在并行执行的Block数量，不会超过物理Vector Core的总数。例如，在56核的950PR上，最多同时运行56个Block。
- 超出部分排队等待：第57个及之后的Block必须等待，直到有物理核完成当前任务并空闲出来，才能被调度执行。
- 调度开销近似线性增加：随着Block数量超出物理核数的程度加剧，调度和切换的开销会近似线性增长，这可能会成为性能瓶颈

### Persistent Kernel

建议Block数量（blockDim）不应超过物理核数，以避免超发带来的调度开销。一种更高效的替代模式是持久化执行（Persistent Execution）：启动固定数量（如32个）的Block，让每个Block通过内部循环处理多个工作项（tile），从而充分利用每个物理核，避免频繁的核间调度。

### 英伟达GPU鼓励Block数量超过SM数量

GPU的SIMT模型擅长通过大量线程的切换来隐藏延迟。
SM（流式多处理器）内部有大量的执行单元。当一个Block中的线程（Warp）因等待内存数据而停顿时，SM可以迅速切换到另一个已驻留的Block的Warp来执行，从而让计算单元保持忙碌。因此，为了让SM总有“备选”任务可切换，通常需要在其上驻留多个Block。

这种驻留能力受限于SM的资源（寄存器、共享内存等），而非Block总数。只要资源允许，一个SM上可以同时驻留多个Block。例如，一个SM可能支持最多32个并发Block，只要资源足够，这些Block就能“填满”SM的延迟隐藏能力。

调度开销低：对于常规规模的超发（例如数千到数万个Block），GPU的硬件调度器可以高效地管理，额外的排队开销并不显著。

### 核心官方来源与证据

以下是几条最具代表性的官方文档，直接支持“应避免Block数量大幅超过物理核数”这一优化原则：

**1. 原子操作指令优化（适用于Ascend 950PR & 950DT）**
这是最直接相关的文档。它明确指出，当线程块数量超过物理核数时，**调度开销会随超出的线程块数量近似线性增加**，并给出了具体的优化建议。
> “当线程块数量不超过物理核数时，所有线程块可以一次占满物理核并行执行；当线程块数量超过物理核数时，超出的线程块必须等待物理核空闲后才能被调度，**调度开销随超出的线程块数量近似线性增加**。因此，**线程块数量应结合物理核数与数据规模设置**。”
>
> **URL**: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/asc-devkit/docs/zh/guide/operator_practice/simt_operator_optimization/instruction_optimization/atomic_instruction_optimization.md`

**2. Matmul算子优化Tiling策略**
该文档从矩阵乘法优化的角度，阐述了完全相同的原理：超出物理核数的Block会引入显著的调度固定开销。
> “当线程块数量 > 物理核数时，超出物理核数的线程块需等待前序线程块执行完成后再调度……**多线程块启动和调度的固定开销开始明显累积，调度开销随线程块数量增大而上升**。”
>
> **URL**: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/matmul_atomic_add.md`

**3. 创建算子工程（CANN编程指南）**
在更基础的算子工程指南中，对Block（`numBlocks`）的设置给出了方向性建议，并特别指出在CV融合场景下，逻辑核数量**不能超过**物理核数量。
> “为了充分利用硬件资源，**一般设置为物理核的核数或其倍数**。”
> “该场景下，设置的`numBlocks`逻辑核的核数**不能超过物理核**（2个Vector核和1个Cube核组合为1个物理核）的核数。”
>
> **URL**: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/programug/Ascendcopdevg/docs/guide/编程指南/高级编程/Aclnn算子工程化开发/创建算子工程.md`

### 来自Triton-Ascend的实践指南

除了CANN官方文档，昇腾平台上的Triton算子开发指南（托管于GitHub的`triton-lang/triton-ascend`仓库）也从实践角度给出了强烈建议，并解释了与GPU的差异。
> “虽然运行时接口允许下发并发任务数最大为65535，但**超过物理核数的部分是通过新一轮的下发来完成的**。如果直接将GPU上的Triton算子拿到昇腾平台上运行，这些大量的任务会引入**可观的核启动和核初始化时的额外开销**，影响到算子性能表现。因此，**最推荐的做法是将分核的数量直接固定为硬件的物理核数**。”

