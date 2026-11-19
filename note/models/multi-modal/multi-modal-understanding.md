## 多模态模型范式

Cross-Attention、统一嵌入Decoder-Only、原生多模态三种范式

### 组合式多模态

#### 跨模态交叉注意力架构（Cross-Attention）：中期融合范式
独立的视觉编码器（如ResNet或ViT）和文本编码器先各自处理输入，
通过交叉注意力层让文本特征去“查询”并融合视觉特征，实现深度交互。

ViT: Vision Transformer。Encoder-only，理解图像。

代表模型：ViLBERT、LXMERT、UNITER等。
能实现模态间的动态交互，但计算开销较大，且本质上仍是两个独立模块的后期结合。

$$
Q = H_t * W_Q
K = H_v * W_K
V = H_v * W_V
$$
#### 统一嵌入Decoder-Only架构：早期融合范式 / 统一解码器范式

将视觉特征通过一个投影层（Connector/Projector）映射为一系列“虚拟视觉Token”，与文本Token拼接成一个单一的序列，然后送入一个统一的Decoder-only Transformer（如GPT系列）中进行因果自注意力计算。

代表模型：LLaVA、Flamingo、GPT-4V、Gemini。
充分利用了LLM的强大推理能力，但视觉编码器与LLM主干往往仍是分离的，融合发生在LLM的输入端，而非其内部表征中。

```
X_fused = Concat(Proj(X_v), X_t)
```

#### CLIP：Dual-Encoder预训练Encoder

图像和文本各有一个独立的编码器。这两个编码器完全独立，图像编码器不知道文本编码器在做什么，反之亦然。它们之间唯一的交互，发生在各自生成最终的特征向量之后——通过计算余弦相似度来对齐到同一个共享空间。

既可以作为“零件”被装配到统一嵌入Decoder-Only架构中，也可以为交叉注意力机制提供特征输入。

### 原生多模态架构

不依赖后期融合或浅层的投影连接。
端到端联合训练。
统一的表征空间。

原生架构：
原始图块嵌入（Native Patch Embedding），原生三维旋转位置编码（Native-RoPE）
