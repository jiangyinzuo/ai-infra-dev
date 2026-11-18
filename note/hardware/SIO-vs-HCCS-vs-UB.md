SIO（Serial IO）：片内跨Die通信。
HCCS
    - A1 ~ A3代际片间互联（Host <-> Device，Device <-> Device）
    - 内存语义协议
    - SDMA（System Direct Memory Access）
URMA（Unified Remote Memory Access）
    - UB互联域内DMA操作
    - A5代际
    - 同时承担节点内部多卡间（Intra-node）和跨节点（Inter-node）的通信
RDMA（RoCE）
    - 跨节点片间拷贝
    - Scale-out参数面
PCIe 
