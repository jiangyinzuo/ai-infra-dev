mkn轴分别是什么: MxK @ KxN -> MxN

CUBE VECTOR和通信的融合
- AIC（CUBE）只能矩阵乘
- AIV（VECTOR）有规约能力、elementwise计算

算子融合的优势：
1. 减少数据来回搬运开销
2. 减少kernel launch开销
3. 减少拷贝，直接写回KVCache
4. C/V融合算子
    - Cube矩阵计算、Vector后处理。计算与后处理分块流水，流水线重叠。
5. 通算融合算子
    - 计算与通信重叠

访存连续的优势：
1. 放在同一个cacheline，L2缓存命中率高
2. HBM的访存速率
    - 2TB
    - 1.8TB @ 4K（举例）
    - 1.5TB @ 2K

有的stream专门负责通信
