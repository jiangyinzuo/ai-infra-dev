[CANN bot skills - NPU 硬件架构总览](https://atomgit.com/cann/cannbot-skills/blob/229ae052ccbb6377bd2599a355797ce0eb8ec4b6/ops/triton-latency-optimizer/references/docs_triton_IR/docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md#ascend910_95-%E7%B3%BB%E5%88%97)

## SIO vs. HCCS vs. UB
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

## A2

[A300I A2标卡](https://support.huawei.com/enterprise/zh/edge-computing-and-server/a300i-a2-pid-260323393)

- 节点内：
    - CPU <-> NPU：PCIe 4.0x16（双向64GB/s）。
    - 8卡NPU间： HCCS fullmesh互联
- 节点间：RoCE互联

Atlas 800I A2等4U服务器中，通常配置4颗CPU与8颗NPU，每个NPU独享一条PCIe 4.0 x16链路连接到CPU主板。这种设计下，4颗CPU与8颗NPU之间的双向互联理论总带宽可达 512GB/s。

## A3

A3似乎没有卖标卡，不能直接插PCIe？

### A3 CloudMatrix384超节点 (aka. Atlas 900 A3 SuperPoD 超节点)

[白皮书](https://e.huawei.com/marketingcloud/pep/asset/20000001/Material/c6e81925d3cf4673a44a687d03a0d6d7/M3T1A669N1272244469801513156/Atlas%20900%20A3%20SuperPoD%20%E8%B6%85%E8%8A%82%E7%82%B9%20%E6%8A%80%E6%9C%AF%E7%99%BD%E7%9A%AE%E4%B9%A6%2007.pdf#1#12#1)
[Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/abs/2506.12708)


其它论文
- xDeepServe: Model-as-a-Service on Huawei CloudMatrix384
- StrataCL: Fabric-Native Communication Library for Production Supernodes
- Huawei Cloud Model-as-a-Service on the CloudMatrix384 SuperPod

Atlas 900 A3 SuperPoD 超节点支持多种配置组网（如64超节点、96超节点、192超节
点和384超节点等）

### 计算柜
单计算柜4个节点，单节点(8NPUs+4CPUs)，单NPU双Die。
总计12柜，`12*4*8=384`个NPU

### 总线设备柜（灵衢柜）

4个灵衢柜，单柜14台灵衢总线设备，总共56台灵衢总线设备。

单灵衢总线设备2个交换芯片。
- 两个交换芯片分别出一个200Gb/s到每个端口组成一个400Gb/s QSFP-DD端口。
- 总共48个400Gb/s 端口

### L1互联（单节点8卡间互联）
NPU模组（HiAM模组）

单机8卡间Clos互联，通过1520交换芯片互联。

每个NPU模组2个Die。一个Die7个灵衢1.0（HCCS）总线端口。
一个端口x4Lane，Serdes速率56Gb/s，双向。单端口单向通信带宽`4*56Gb/s=224Gb/s`
每个NPU模组出单向`2（Die）*7（个端口）*200Gb/s（224Gb/s约等于200）`，通过线缆连接至L1灵衢总线板的灵衢总线网络。

物理上每个NPU出2*7个链路。逻辑上分7个交换平面

### L2互联

L1: 48个计算节点。
L2: 7个Sub-plane，每个sub-plane8台灵衢总线设备。单个灵衢总线设备2个交换芯片。
LQC可能也叫HCCS

单交换设备48个端口，连接48个计算设备。

## A5

### Atlas 350标卡

1 * 昇腾950PR
