https://gitcode.com/cann/asc-devkit

https://gitcode.com/cann/asc-devkit/tree/master/examples

## Block

## Tiling编程

Tiling编程: TilingContext TilingData

以向量Reduce为例，
数据划分的层次结构：**Core → Block → Tile**

适配其**SPMD（单程序多数据）多核并行架构**，
- 有限的AI Core数量
- 每个Core内有限的片上内存（如UB）容量。

1.  **第一层：Core级切分（核间切分 / Inter-Core）**
    这是最粗粒度的切分。系统会将全局大张量（Global Memory中的Tensor）均匀地划分成多个数据块，每个块分配给一个**AI Core**独立处理，以实现多核并行和负载均衡。在Reduce操作的语义中，这通常意味着**每个Core先对其负责的数据分片进行一次局部的Reduce计算**，以减少后续的通信开销。
2.  **第二层：Block级切分（核内切分 / Intra-Core）**
    分配到单个Core上的数据量仍然可能远超其片上缓存（如Unified Buffer, UB）的容量。因此，在Core内部，需要将数据进一步切分为多个**Block**。这种切分通常是将数据分割成适合UB容量大小的块，以便“迭代分块搬运、分批计算”。在Tiling术语中，**Block切分特指将数据分配到多个核上并行执行**，但在核内的语境下，它也指代了将任务进一步分解为适合硬件处理的逻辑块。
3.  **第三层：Tile级切分**
    Tile是更细粒度的切分单元，通常发生在Block内部。一个Block的数据会被进一步拆解成多个**Tile**，这些Tile是实际进行向量计算（Vector Function, VF）的基本单位。Tile的大小需要精细匹配硬件向量寄存器的宽度或UB的子分区，以最大化计算效率。例如，在向量化执行时，一个Tile可能正好对应一条SIMD指令所能处理的数据量。

### 总结与关键点

*   **层次关系**：数据流清晰地遵循 **`Global Memory -> Core (核间切分) -> Block (核内逻辑切分) -> Tile (计算单元切分) -> UB/Register`** 的路径。
*   **Reduce的特殊性**：对于Reduce操作，**Core级**切分往往对应**局部归约**，而最终的**全局归约**可能在单个Core或Host侧完成。同时，归约轴上的数据通常不会跨Core切分，而是在Core内的Block/Tile层次通过循环来逐步累加。

## 动态shape下的tiling切分

UB地址必须32字节对齐，以匹配硬件总线粒度和SIMD计算单元并行度，非对齐访问将导致运行时错误。
- [UB上最小访问粒度：32字节](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/cpp_tensor_programming_overview.md)
- [避免bank冲突（NPU架构版本3510）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md)
