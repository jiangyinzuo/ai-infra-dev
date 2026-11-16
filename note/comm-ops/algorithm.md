Ring算法实现AllReduce：N张卡，通信步骤N-1。但在卡数量多的时候，步骤太多了。
- Ring可以搞多个环

A3: 两die间有个SIO DoubleRing环

Mesh算法：适用于Full-Mesh拓扑。只需要一步通信。

Halving Doubling算法

A5: server内带宽和server间带宽相差不大（看端口连接数量）。一维做server内、一维做server间，两维并行算法。

单次同步：一个rtt时间。

server间带宽低，所以先在server内reduce，再在server间reduce。

随路计算：DMA一边般一边算


