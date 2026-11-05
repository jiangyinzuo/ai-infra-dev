模型结构都是为了解决当时的问题作出的取舍。
Transformer取代LSTM不代表Transformer结构永远是最好的。
模型结构的演进是有迹可循的。

找到当前的瓶颈，做出针对性优化
- 降维、拆卡、不算、复用...

模型结构受到物理世界的约束。

机器学习 -> 深度学习：手动构造特征 -> 自动学习特征
ResNet：解决深层网络难以学习恒等映射的问题，扩展网络层数
多头注意力：多头方便拆到多卡
GPT-2: 用Decoder-Only续写文本, Pre-Norm
LLaMA2-7B: 现代Dense Decoder
针对KVCache的优化：
- GQA/MQA
    - LLaMA2: GQA
- DeepSeek V2 / V3: MLA
    - 矩阵吸收
DeepSeek V2 / V3: MoE
    - 小专家: 可选组合数变大
    - 共享专家

长上下文，KVCache暴涨
1. 能否不存全量KV（线性注意力）
2. 能否不算全量KV（稀疏注意力）
3. 能否复用KVCache（DeepSeek V4.1）

稀疏注意力 vs. 线性注意力, KVCache压缩

线性注意力：把历史K/V汇总到一个State（S和z）
和LSTM有点像？

稀疏注意力
- KVCache分了Main Cache / Index Cache
- 访存很稀疏，不够昇腾亲和

SeqToSeq -> Mamba -> LinearAttention
MHA -> MQA -> GQA -> MLA

DeepSeek V4
- SWA: 只看前128个token的KV
- CSA: Compressed Sparse Attention, 把多个Token存成一个KVCache
- HCA
- mHC

Kimi K3

对残差的优化
- mHC
- AttnRes

DeepSeek V4.1
- Engram
- CSA
- CED


未来模型可能的改进方向：
- KVCache成为瓶颈
- MoE，怎么放更多专家

分类器 <-> lmhead
