// Rich symbol previews: edit preview: [...] using Typst text, formulas, images or tables.
// Existing LaTeX formulas use MiTeX raw strings; new content can use native $...$ math.
#import "@preview/mitex:0.2.7": mi

#let symbols = (
  defaultContext: "mla",
  sections: (
    (
      heading: "DeepSeek MoE",
      "context": "moe",
    ),
    (
      heading: "SwiGLU FFN",
      "context": "ffn",
    ),
    (
      heading: "Dense FFN和MoE",
      "context": "ffn",
    ),
    (
      heading: "DeepSeek Sparse Attention",
      "context": "dsa",
    ),
    (
      heading: "Lightning Indexer 与 Full Attention",
      "context": "complexity",
    ),
  ),
  fallbacks: (
    mla: ("mla", "any",),
    moe: ("moe", "any", "mla",),
    ffn: ("ffn", "moe", "mla", "any",),
    dsa: ("dsa", "mla", "any",),
    complexity: ("complexity", "dsa", "mla", "any",),
    any: ("any", "mla",),
  ),
  definitions: (
    (
      key: "T",
      "context": "any",
      label: $T$,
      preview: [
        上标 $T$ 表示转置。对列向量 $x in RR^d$，$x^T$ 是行向量；QK 分数中的 $q^T k$ 是标量内积。
      ],
    ),
    (
      key: "MatMul",
      "context": "any",
      label: [MatMul],
      preview: [
        矩阵乘法。对输入向量 $x in RR^n$ 和投影权重 $W in RR^(m times n)$，结果为 $W x in RR^m$。
      ],
    ),
    (
      key: "Attention",
      "context": "mla",
      label: [Attention],
      preview: [
        将 QK 点积、缩放、Softmax 和 Value 加权聚合画在同一个计算框中；三个数据输入是 Query、Key、Value，输出为当前头的 #mi(`\mathbf o_{t,i}`)。

        #mi(`s_j=\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}`)
        \ #mi(`A_j=\frac{\exp(s_j)}{\sum_{\ell\in\mathcal J_t}\exp(s_\ell)}`)
        \ #mi(`\operatorname{Attention}=\sum_{j\in\mathcal J_t}A_j\mathbf v_{j,i}^C`)

        MLA 的候选集合为 #mi(`\mathcal J_t=\{1,\ldots,t\}`)；DSA 为 Indexer 选中的 #mi(`\mathcal J_t=\mathcal S_t`)，仅在所选位置上打分、归一化并聚合。这里的未吸收展开使用 #mi(`d_h+d_h^R=128+64=192`)，不是 Indexer 的 128 维，也不是吸收后的 576 维。
      ],
    ),
    (
      key: "AttentionQKAbsorbed",
      "context": "mla",
      label: [Attention],
      preview: [
        QK 内容矩阵吸收后的注意力：#mi(`W_i^{QK}=(W_i^{UQ})^TW_i^{UK}`)，不再显式构造历史内容 Key。

        #mi(`\widehat{\mathbf q}_{t,i}^C=(W_i^{QK})^T\mathbf c_t^Q`)
        \ #mi(`\widehat{\mathbf q}_{t,i}=[\widehat{\mathbf q}_{t,i}^C;\mathbf q_{t,i}^R]`)
        \ #mi(`\widehat{\mathbf k}_j=[\mathbf c_j^{KV};\mathbf k_j^R]`)
        \ #mi(`s_j=\frac{(\widehat{\mathbf q}_{t,i}^C)^T\mathbf c_j^{KV}+(\mathbf q_{t,i}^R)^T\mathbf k_j^R}{\sqrt{d_h+d_h^R}}`)
        \ #mi(`\mathbf o_{t,i}=\sum_{j\le t}\operatorname{Softmax}_{j\le t}(s_j)\mathbf v_{j,i}^C`)

        内容维度由 128 变为 512，拼接 RoPE 后为 576；缩放仍为原模型的 #mi(`1/\sqrt{192}`)，不是 #mi(`1/\sqrt{576}`)。RoPE 与 Value 分支不变。
      ],
    ),
    (
      key: "AttentionValueAbsorbed",
      "context": "mla",
      label: [Attention],
      preview: [
        Value / 输出矩阵吸收后的注意力：QK 与 RoPE 不变，聚合输入换成 512 维 KV latent。

        #mi(`s_j=\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}`)
        \ #mi(`\widetilde{\mathbf o}_{t,i}=\sum_{j\le t}\operatorname{Softmax}_{j\le t}(s_j)\mathbf c_j^{KV}`)
        \ #mi(`\mathbf u_t=W^{O,UV}[\widetilde{\mathbf o}_{t,1};\ldots;\widetilde{\mathbf o}_{t,n_h}]`)

        #mi(`W^{O,UV}=W^O\operatorname{diag}(W_1^{UV},\ldots,W_{n_h}^{UV})`)；不再为每个历史位置显式构造 128 维 Value。
      ],
    ),
    (
      key: "SparseAttention",
      "context": "dsa",
      label: [Sparse Attention],
      preview: [
        输入：Query、Key、Value、选中位置索引 #mi(`\mathcal S_t`)；输出：每头结果 #mi(`\mathbf o_{t,i}`)。

        #mi(`s_j=\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}`)
        \ #mi(`\operatorname{SparseAttention}(\mathbf q,K,V,\mathcal S_t)`)
        \ #mi(`=\sum_{j\in\mathcal S_t}\frac{\exp(s_j)}{\sum_{\ell\in\mathcal S_t}\exp(s_\ell)}\mathbf v_{j,i}^C`)

        仅对 #mi(`\mathcal S_t`) 中的位置计算注意力。

        #link("https://docs.flashinfer.ai/api/sparse.html#flashinfer.sparse.BlockSparseAttentionWrapper")[FlashInfer `BlockSparseAttentionWrapper`] 支持索引驱动的稀疏注意力：`plan(indptr, indices, ...)` 指定稀疏块，`run(q, k, v)` 执行计算。
      ],
    ),
    (
      key: "SparseAttentionAbsorbed",
      "context": "dsa",
      label: [Sparse Attention],
      preview: [
        同时完成 MLA 的 QK 内容吸收与 Value / 输出吸收。输入为 #mi(`\widehat{\mathbf q}_{t,i}=[\widehat{\mathbf q}_{t,i}^C;\mathbf q_{t,i}^R]`)、#mi(`[\mathbf c_j^{KV};\mathbf k_j^R]`)、聚合用 KV latent 和选中位置 #mi(`\mathcal S_t`)。

        #mi(`s_j=\frac{(\widehat{\mathbf q}_{t,i}^C)^T\mathbf c_j^{KV}+(\mathbf q_{t,i}^R)^T\mathbf k_j^R}{\sqrt{192}}`)
        \ #mi(`\widetilde{\mathbf o}_{t,i}=\sum_{j\in\mathcal S_t}\operatorname{Softmax}_{j\in\mathcal S_t}(s_j)\mathbf c_j^{KV}`)
        \ #mi(`\mathbf u_t=W^{O,UV}[\widetilde{\mathbf o}_{t,1};\ldots;\widetilde{\mathbf o}_{t,n_h}]`)

        Query / Key 点积维度为 576，Value 聚合维度为 512；缩放仍为原模型的 #mi(`1/\sqrt{192}`)。
      ],
    ),
    (
      key: "Wbar^IW",
      "context": "dsa",
      label: [#mi(`\bar{W}^{IW}`)],
      preview: [
        固定标量缩放后的 Indexer 投影权重（等价重参数化）：

        #mi(`\bar{W}^{IW}=W^{IW}/\sqrt{n_h^I d_h^I}`)
        \ #mi(`\mathbf w_t^I=\bar{W}^{IW}\mathbf h_t`)

        尺寸仍为 #mi(`64\times7168`)，只是把固定标量 #mi(`1/\sqrt{64\times128}`) 乘入 #mi(`W^{IW}`) 的每个元素；也可在投影激活上逐元素乘该常量。它不是相邻线性层的矩阵乘法合并，不改变 GEMM 形状或复杂度。Indexer 的 LayerNorm、部分 RoPE 和 ReLU 均保留，不能将 Query / Key 投影直接合成静态矩阵。等式忽略浮点舍入与量化误差。
      ],
    ),
    (
      key: "IndexerWeightScaleFold",
      "context": "dsa",
      label: [#mi(`w_{t,i}^{I,\mathrm{orig}}\to w_{t,i}^{I,\mathrm{fold}}`)],
      preview: [
        #strong[MatMul / #mi(`\sqrt{n_h^I d_h^I}`)：折叠前与折叠后]

        #mi(`w_{t,i}^{I,\mathrm{orig}}=\frac{[W^{IW}\mathbf h_t]_i}{\sqrt{n_h^I d_h^I}}=\left[\frac{W^{IW}}{\sqrt{n_h^I d_h^I}}\mathbf h_t\right]_i=w_{t,i}^{I,\mathrm{fold}}`)

        #mi(`\bar W^{IW}=W^{IW}/\sqrt{n_h^I d_h^I}`)；仅将固定标量并入权重，GEMM 形状和计算量不变。
      ],
    ),
    (
      key: "MatMulRMSNorm",
      "context": "mla",
      label: [MatMulRMSNorm],
      preview: [
        #strong[融合算子的数学展开（本图为 MatMul → RMSNorm）]

        #mi(`\operatorname{MatMulRMSNorm}(\mathbf x,W,\mathbf g)`)
        \ #mi(`=\operatorname{RMSNorm}_{\mathbf g}(W\mathbf x)`)
        \ #mi(`=\frac{(W\mathbf x)\odot\mathbf g}{\operatorname{RMS}_m(W\mathbf x)}`)

        将归约与逐元素后处理拆开：

        #mi(`\mathbf z=W\mathbf x,\quad s=\sum_{a=1}^{m}z_a^2`)
        \ #mi(`\rho=\operatorname{rsqrt}(s/m+\epsilon)`)
        \ #mi(`\operatorname{MatMulRMSNorm}(\mathbf x,W,\mathbf g)=\mathbf z\odot(\mathbf g\rho)`)

        #strong[Prologue：准备 MatMul 输入]

        读取 hidden state #mi(`\mathbf x`) 和投影权重 #mi(`W`)，准备矩阵乘法所需的数据块；读取缩放参数 #mi(`\mathbf g`) 供 Epilogue 使用。随后 MatMul 主循环累加得到 #mi(`\mathbf z=W\mathbf x`)。

        #strong[Epilogue：归约、归一化、写出最终结果]

        ① 对完整输出行求平方和 #mi(`s=\sum_a z_a^2`)；② 计算每行共享的倒数 RMS #mi(`\rho=\operatorname{rsqrt}(s/m+\epsilon)`)；③ 逐元素计算 #mi(`c_a=z_a g_a\rho`)，直接写出归一化后的 #mi(`\mathbf c`)。这样不必把未归一化的 #mi(`\mathbf z`) 单独写回再读入 RMSNorm；跨 tile 的行归约与同步仍需内核保证。

        #strong[分母取 MatMul 输出的 RMS，不是输入 hidden state 的 RMS。] Query 为 #mi(`W^{DQ},\boldsymbol\gamma^Q,m=1536`)，KV 为 #mi(`W^{DKV},\boldsymbol\gamma^{KV},m=512`)；输出沿用图中的 #mi(`\mathbf c_t^Q`) / #mi(`\mathbf c_j^{KV}`)。公式忽略浮点舍入；逻辑融合不等于声明官方已实现单个融合 kernel。
      ],
    ),
    (
      key: "calO",
      "context": "any",
      preview: [
        大 #mi(`O`) 记号，表示计算复杂度的渐近上界，不是注意力输出 #mi(`O`)。
      ],
    ),
    (
      key: "n",
      "context": "any",
      preview: [
        序列长度 / 上下文token数
      ],
    ),
    (
      key: "d",
      "context": "any",
      preview: [
        模型hidden state维度；Llama 2-7B 为 4096，DeepSeek-V3 为 7168；不要求 #mi(`d=n_{h} d_{h}`)。
      ],
    ),
    (
      key: "t",
      "context": "any",
      preview: [
        当前 Query 或当前生成 token 的位置；因果范围 #mi(`1\le t\le n`)。
      ],
    ),
    (
      key: "j",
      "context": "any",
      preview: [
        历史 Key/Value 的位置；固定 Query #mi(`t`) 时仅访问 #mi(`1\le j\le t`)；MoE 求和中另作专家索引。
      ],
    ),
    (
      key: "i",
      "context": "any",
      preview: [
        当前头或专家的索引；范围由所在段落的 #mi(`n_{h}`)、#mi(`n_{h}^{I}`)、#mi(`N_{s}`) 或 #mi(`N_{r}`) 决定。
      ],
    ),
    (
      key: "ell",
      "context": "any",
      preview: [
        层编号；MoE 推导省略该下标，参数属于当前层。
      ],
    ),
    (
      key: "h_t",
      "context": "any",
      preview: [
        第 #mi(`t`) 个 token 的输入hidden state，维度 #mi(`d`)；DSA 中已过 attention 前置 RMSNorm。
      ],
    ),
    (
      key: "h_j",
      "context": "any",
      preview: [
        第 #mi(`j`) 个历史 token 的hidden state，维度 #mi(`d`)。
      ],
    ),
    (
      key: "Softmax",
      "context": "any",
      preview: [
        沿历史 Key 位置 #mi(`j`) 归一化；矩阵形式按行沿列归一化，比较表中已含因果 mask。
      ],
    ),
    (
      key: "Softmax_j",
      "context": "any",
      preview: [
        固定 Query 位置和头，对可见历史位置 #mi(`j`) 的分数进行 Softmax。
      ],
    ),
    (
      key: "RoPE",
      "context": "any",
      preview: [
        旋转位置编码：对向量施加随 token 位置变化的旋转；Value 不施加 RoPE。
      ],
    ),
    (
      key: "Sigmoid",
      "context": "any",
      preview: [
        逐元素 sigmoid 函数；MoE 将 token 与专家中心的内积转为亲和分数。
      ],
    ),
    (
      key: "diag",
      "context": "any",
      preview: [
        由给定矩阵构造分块对角矩阵；输入标量或向量时构造普通对角矩阵。
      ],
    ),
    (
      key: "odot",
      "context": "any",
      preview: [
        逐元素乘法；FFN 门控和归一化的可学习缩放均使用此运算。
      ],
    ),
    (
      key: "concat",
      "context": "any",
      label: [Concat],
      preview: [
        向量 $[x;y]$ 或矩阵 $[X,Y]$ 表示拼接；矩阵比较表约定沿最后一维拼接。
      ],
    ),
    (
      key: "sqrt",
      label: $sqrt(dot.op)$,
      "context": "any",
      preview: [
        平方根；用于注意力分数缩放和归一化分母。
      ],
    ),
    (
      key: "n_h",
      "context": "mla",
      preview: [
        MLA/MHA 注意力头数；Llama 2-7B 为 32，DeepSeek-V3 为 128；不是 indexer 头数。
      ],
    ),
    (
      key: "d_h",
      "context": "mla",
      preview: [
        每个注意力头的内容 Key/Query 和 Value 维度；两种示例模型均为 128。
      ],
    ),
    (
      key: "d_c",
      "context": "mla",
      preview: [
        KV 联合压缩的潜在维度，远小于 #mi(`n_{h} d_{h}`)；DeepSeek-V3 为 512。
      ],
    ),
    (
      key: "d_c'",
      "context": "mla",
      preview: [
        Query 低秩压缩的潜在维度，远小于 #mi(`n_{h} d_{h}`)；DeepSeek-V3 为 1536。
      ],
    ),
    (
      key: "d_h^R",
      "context": "mla",
      preview: [
        每头解耦 RoPE 子维度；DeepSeek-V3 为 64，RoPE Key 在所有头间共享。
      ],
    ),
    (
      key: "H",
      "context": "mla",
      preview: [
        按 token 行堆叠的输入隐藏矩阵，尺寸 #mi(`n\times d`)。
      ],
    ),
    (
      key: "M",
      "context": "mla",
      preview: [
        因果 mask 矩阵，尺寸 #mi(`n\times n`)；#mi(`j\le t`) 时 #mi(`M_{t,j}=0`)，否则为负无穷。
      ],
    ),
    (
      key: "M_tj",
      "context": "mla",
      preview: [
        因果 mask 在 Query #mi(`t`)、Key #mi(`j`) 处的标量；#mi(`j\le t`) 为 0，否则为负无穷。
      ],
    ),
    (
      key: "W_Q^(i)",
      "context": "mla",
      preview: [
        Llama MHA 第 #mi(`i`) 个头的 Query 投影矩阵，尺寸 #mi(`d_{h}\times d`)。
      ],
    ),
    (
      key: "W_K^(i)",
      "context": "mla",
      preview: [
        Llama MHA 第 #mi(`i`) 个头的 Key 投影矩阵，尺寸 #mi(`d_{h}\times d`)。
      ],
    ),
    (
      key: "W_V^(i)",
      "context": "mla",
      preview: [
        Llama MHA 第 #mi(`i`) 个头的 Value 投影矩阵，尺寸 #mi(`d_{h}\times d`)。
      ],
    ),
    (
      key: "W^O",
      "context": "mla",
      preview: [
        多头输出投影矩阵，尺寸 #mi(`d\times (n_{h} d_{h})`)；DeepSeek-V3 为 #mi(`7168\times 16384`)。
      ],
    ),
    (
      key: "W^DQ",
      "context": "mla",
      preview: [
        MLA Query 下投影矩阵，尺寸 #mi(`d_{c}^{\prime}\times d=1536\times 7168`)；实现中也供 indexer 复用。
      ],
    ),
    (
      key: "W^UQ",
      "context": "mla",
      preview: [
        MLA 全部头的内容 Query 上投影矩阵，尺寸 #mi(`(n_{h} d_{h})\times d_{c}^{\prime}=16384\times 1536`)。
      ],
    ),
    (
      key: "W^QR",
      "context": "mla",
      preview: [
        MLA 全部头的 Query RoPE 投影矩阵，尺寸 #mi(`(n_{h} d_{h}^{R})\times d_{c}^{\prime}=8192\times 1536`)。
      ],
    ),
    (
      key: "W^DKV",
      "context": "mla",
      preview: [
        KV 联合下投影矩阵，尺寸 #mi(`d_{c}\times d=512\times 7168`)。
      ],
    ),
    (
      key: "W^UK",
      "context": "mla",
      preview: [
        全部头的内容 Key 上投影矩阵，尺寸 #mi(`(n_{h} d_{h})\times d_{c}=16384\times 512`)。
      ],
    ),
    (
      key: "W^UV",
      "context": "mla",
      preview: [
        全部头的 Value 上投影矩阵，尺寸 #mi(`(n_{h} d_{h})\times d_{c}=16384\times 512`)。
      ],
    ),
    (
      key: "W^KR",
      "context": "mla",
      preview: [
        共享 RoPE Key 投影矩阵，尺寸 #mi(`d_{h}^{R}\times d=64\times 7168`)。
      ],
    ),
    (
      key: "W_i^UQ",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的内容 Query 上投影分块，尺寸 #mi(`d_{h}\times d_{c}^{\prime}`)。
      ],
    ),
    (
      key: "W_i^QR",
      "context": "mla",
      label: $W_i^(Q R)$,
      preview: [
        $W^(Q R)$ 中第 $i$ 个头的 Query RoPE 投影分块，尺寸 $d_h^R times d_c' = 64 times 1536$。其输入是 Query 压缩潜向量，投影后在位置 $t$ 施加 RoPE。
      ],
    ),
    (
      key: "W_i^UK",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的内容 Key 上投影分块，尺寸 #mi(`d_{h}\times d_{c}`)。
      ],
    ),
    (
      key: "W_i^UV",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的 Value 上投影分块，尺寸 #mi(`d_{h}\times d_{c}`)。
      ],
    ),
    (
      key: "W_i^QK",
      "context": "mla",
      preview: [
        吸收后的内容打分矩阵 #mi(`(W_{i}^{UQ})^T W_{i}^{UK}`)，尺寸 #mi(`d_{c}^{\prime}\times d_{c}`)。
      ],
    ),
    (
      key: "hat_q_ti^C",
      "context": "mla",
      label: [#mi(`\widehat{\mathbf q}_{t,i}^{C}`)],
      preview: [
        按框图报告记号，吸收后的内容 Query 为 #mi(`\widehat{\mathbf q}_{t,i}^{C}=(W_i^{QK})^T\mathbf c_t^Q=(W_i^{UK})^TW_i^{UQ}\mathbf c_t^Q`)，维度 #mi(`d_c=512`)。图中 #mi(`\mathbf c_t^Q`) 是融合块的归一化输出，对应 DSA 正文的 #mi(`\bar{\mathbf c}_t^Q`)。直接与 KV 缓存 #mi(`\mathbf c_j^{KV}`) 做点积，不再构造历史 Key 的内容上投影。
      ],
    ),
    (
      key: "S_tji^C",
      "context": "mla",
      label: [#mi(`S_{t,j,i}^{C}`)],
      preview: [
        未缩放的内容分数 #mi(`(\mathbf q_{t,i}^C)^T\mathbf k_{j,i}^C=(\widehat{\mathbf q}_{t,i}^C)^T\mathbf c_j^{KV}`)。它不含 RoPE 项，也未除以 #mi(`\sqrt{d_h+d_h^R}`)，不是完整的注意力分数 #mi(`S_{t,j,i}`)。
      ],
    ),
    (
      key: "W_blk^UV",
      "context": "mla",
      preview: [
        由各头 #mi(`W_{i}^{UV}`) 构成的分块对角矩阵，尺寸 #mi(`(n_{h} d_{h})\times (n_{h} d_{c})`)。
      ],
    ),
    (
      key: "W^O,UV",
      "context": "mla",
      preview: [
        吸收后的输出投影 #mi(`W^{O} W_{\mathrm{blk}}^{UV}`)，尺寸 #mi(`d\times (n_{h} d_{c})`)。
      ],
    ),
    (
      key: "c_t^KV",
      "context": "mla",
      preview: [
        token #mi(`t`) 的共享 KV 压缩潜向量 #mi(`W^{DKV} h_{t}`)，维度 #mi(`d_{c}=512`)；写入 KV Cache。
      ],
    ),
    (
      key: "c_j^KV",
      "context": "mla",
      preview: [
        历史位置 #mi(`j`) 的共享 KV 压缩潜向量，维度 #mi(`d_c=512`)；所有注意力头共用。正文报告公式省略归一化；框图及实际缓存使用 #mi(`\operatorname{RMSNorm}_{\gamma^{KV}}(W^{DKV}\mathbf h_j)`)。
      ],
    ),
    (
      key: "c_t^Q",
      "context": "mla",
      preview: [
        token #mi(`t`) 的 Query 压缩潜向量，维度 #mi(`d_c^{\prime}=1536`)。正文按报告公式写作 #mi(`W^{DQ}\mathbf h_t`)；框图的 MatMulRMSNorm 沿用此记号，但实际输出为 #mi(`\operatorname{RMSNorm}_{\gamma^Q}(W^{DQ}\mathbf h_t)`)，对应 DSA 正文显式记出的 #mi(`\bar{\mathbf c}_t^Q`)。
      ],
    ),
    (
      key: "C^Q",
      "context": "mla",
      preview: [
        按 token 行堆叠的 Query 压缩潜在矩阵，尺寸 #mi(`n\times d_{c}^{\prime}`)。
      ],
    ),
    (
      key: "C^KV",
      "context": "mla",
      preview: [
        按 token 行堆叠的共享 KV 压缩潜在矩阵，尺寸 #mi(`n\times d_{c}`)。
      ],
    ),
    (
      key: "q_t^C",
      "context": "mla",
      preview: [
        token #mi(`t`) 的多头内容 Query 拼接向量，维度 #mi(`n_{h} d_{h}`)。
      ],
    ),
    (
      key: "k_t^C",
      "context": "mla",
      preview: [
        token #mi(`t`) 的多头内容 Key 拼接向量，维度 #mi(`n_{h} d_{h}`)；不写入 MLA KV Cache。
      ],
    ),
    (
      key: "v_t^C",
      "context": "mla",
      preview: [
        token #mi(`t`) 的多头 Value 拼接向量，维度 #mi(`n_{h} d_{h}`)；无 RoPE，不写入 MLA KV Cache。
      ],
    ),
    (
      key: "q_t^R",
      "context": "mla",
      preview: [
        开头定义为各头 RoPE Query 的拼接，维度 #mi(`n_{h} d_{h}^{R}`)；矩阵吸收段省略头索引，存在原文记号歧义。
      ],
    ),
    (
      key: "k_t^R",
      "context": "mla",
      preview: [
        token #mi(`t`) 的共享 RoPE Key，维度 #mi(`d_{h}^{R}=64`)；投影旋转后写入 KV Cache。
      ],
    ),
    (
      key: "k_j^R",
      "context": "mla",
      preview: [
        历史位置 #mi(`j`) 的共享 RoPE Key，维度 #mi(`d_{h}^{R}=64`)。
      ],
    ),
    (
      key: "q_ti^C",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的内容 Query，维度 #mi(`d_{h}`)。
      ],
    ),
    (
      key: "k_ti^C",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的内容 Key，维度 #mi(`d_{h}`)。
      ],
    ),
    (
      key: "k_ji^C",
      "context": "mla",
      preview: [
        历史位置 #mi(`j`)、第 #mi(`i`) 个头的内容 Key，维度 #mi(`d_{h}`)。
      ],
    ),
    (
      key: "v_ti^C",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的 Value，维度 #mi(`d_{h}`)；仅含内容部分。
      ],
    ),
    (
      key: "v_ji^C",
      "context": "mla",
      preview: [
        历史位置 #mi(`j`)、第 #mi(`i`) 个头的 Value，维度 #mi(`d_{h}`)；由 #mi(`c_{j}^{KV}`) 上投影生成。
      ],
    ),
    (
      key: "q_ti^R",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的 RoPE Query，维度 #mi(`d_{h}^{R}`)。
      ],
    ),
    (
      key: "q_ti",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的完整 Query；MLA 为 #mi(`[q_{t,i}^{C};q_{t,i}^{R}]`)，维度 #mi(`d_{h}+d_{h}^{R}`)；Llama MHA 为 #mi(`d_{h}`)。
      ],
    ),
    (
      key: "k_ti",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的完整 Key；MLA 为 #mi(`[k_{t,i}^{C};k_{t}^{R}]`)，维度 #mi(`d_{h}+d_{h}^{R}`)。
      ],
    ),
    (
      key: "k_ji",
      "context": "mla",
      preview: [
        历史位置 #mi(`j`)、第 #mi(`i`) 个头的完整 Key；MLA 维度 #mi(`d_{h}+d_{h}^{R}`)，Llama MHA 为 #mi(`d_{h}`)。
      ],
    ),
    (
      key: "v_ji",
      "context": "mla",
      preview: [
        Llama MHA 历史位置 #mi(`j`)、第 #mi(`i`) 个头的 Value，维度 #mi(`d_{h}`)，不施加 RoPE。
      ],
    ),
    (
      key: "o_ti",
      "context": "mla",
      preview: [
        token #mi(`t`)、第 #mi(`i`) 个头的注意力输出，维度 #mi(`d_{h}`)；由历史 Value 按注意力权重加权求和。
      ],
    ),
    (
      key: "tilde_o_ti",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头在压缩潜空间的中间输出 Σ\_j #mi(`A_{t,j,i} c_{j}^{KV}`)，维度 #mi(`d_{c}`)。
      ],
    ),
    (
      key: "u_t",
      "context": "mla",
      preview: [
        多头注意力经输出投影后的结果，维度 #mi(`d`)；后文用作 FFN 输入。
      ],
    ),
    (
      key: "O_t",
      "context": "mla",
      preview: [
        比较表中的 token #mi(`t`) 多头注意力最终输出，维度 #mi(`d`)；由 #mi(`W^{O}`) 投影得到。
      ],
    ),
    (
      key: "S_tji",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头在 Query #mi(`t`)、Key #mi(`j`) 处的缩放注意力分数，为标量。
      ],
    ),
    (
      key: "A_tji",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头在 Query #mi(`t`)、Key #mi(`j`) 处的注意力权重，为沿 #mi(`j`) 归一化的标量。
      ],
    ),
    (
      key: "S_i",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的全序列注意力打分矩阵，尺寸 #mi(`n\times n`)；比较表中包含缩放和因果 mask。
      ],
    ),
    (
      key: "A_i",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的全序列注意力权重矩阵 #mi(`\operatorname{Softmax}(S_{i})`)，尺寸 #mi(`n\times n`)。
      ],
    ),
    (
      key: "Q_i",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的全序列 Query 矩阵；MHA 为 #mi(`n\times d_{h}`)，MLA 为 #mi(`n\times (d_{h}+d_{h}^{R})`)。
      ],
    ),
    (
      key: "K_i",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的全序列 Key 矩阵；MHA 为 #mi(`n\times d_{h}`)，MLA 为 #mi(`n\times (d_{h}+d_{h}^{R})`)。
      ],
    ),
    (
      key: "V_i",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的全序列 Value 矩阵，尺寸 #mi(`n\times d_{h}`)。
      ],
    ),
    (
      key: "O_i",
      "context": "mla",
      preview: [
        第 #mi(`i`) 个头的全序列注意力输出矩阵，尺寸 #mi(`n\times d_{h}`)。
      ],
    ),
    (
      key: "Q_i^C",
      "context": "mla",
      preview: [
        MLA 第 #mi(`i`) 个头的全序列内容 Query 矩阵，尺寸 #mi(`n\times d_{h}`)。
      ],
    ),
    (
      key: "K_i^C",
      "context": "mla",
      preview: [
        MLA 第 #mi(`i`) 个头的全序列内容 Key 矩阵，尺寸 #mi(`n\times d_{h}`)。
      ],
    ),
    (
      key: "V_i^C",
      "context": "mla",
      preview: [
        MLA 第 #mi(`i`) 个头的全序列 Value 矩阵，尺寸 #mi(`n\times d_{h}`)。
      ],
    ),
    (
      key: "Q_i^R",
      "context": "mla",
      preview: [
        MLA 第 #mi(`i`) 个头的全序列 RoPE Query 矩阵，尺寸 #mi(`n\times d_{h}^{R}`)。
      ],
    ),
    (
      key: "Q^C",
      "context": "mla",
      preview: [
        所有头沿最后一维拼接的内容 Query 矩阵，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "K^C",
      "context": "mla",
      preview: [
        所有头沿最后一维拼接的内容 Key 矩阵，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "V^C",
      "context": "mla",
      preview: [
        所有头沿最后一维拼接的 Value 矩阵，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "Q^R",
      "context": "mla",
      preview: [
        全序列多头 RoPE Query 矩阵，尺寸 #mi(`n\times (n_{h} d_{h}^{R})=n\times 8192`)。
      ],
    ),
    (
      key: "K^R",
      "context": "mla",
      preview: [
        全序列所有头共享的 RoPE Key 矩阵，尺寸 #mi(`n\times d_{h}^{R}=n\times 64`)。
      ],
    ),
    (
      key: "Q",
      "context": "mla",
      preview: [
        各头完整 Query 沿最后一维拼接；MHA 为 #mi(`n\times (n_{h} d_{h})`)，MLA 为 #mi(`n\times [n_{h}(d_{h}+d_{h}^{R})]`)。
      ],
    ),
    (
      key: "K",
      "context": "mla",
      preview: [
        各头完整 Key 沿最后一维拼接；MHA 为 #mi(`n\times (n_{h} d_{h})`)，MLA 为 #mi(`n\times [n_{h}(d_{h}+d_{h}^{R})]`)。
      ],
    ),
    (
      key: "V",
      "context": "mla",
      preview: [
        各头 Value 沿最后一维拼接的矩阵，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "O_cat",
      "context": "mla",
      preview: [
        各头输出 #mi(`[O_{1},\ldots,O_{n_h}]`) 的拼接矩阵，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "O",
      "context": "mla",
      preview: [
        全序列最终注意力输出 #mi(`O_{\mathrm{cat}} (W^{O})^T`)，尺寸 #mi(`n\times d`)。
      ],
    ),
    (
      key: "K_cache",
      "context": "mla",
      preview: [
        Llama MHA 实际缓存的完整 Key 矩阵 #mi(`K`)，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "V_cache",
      "context": "mla",
      preview: [
        Llama MHA 实际缓存的完整 Value 矩阵 #mi(`V`)，尺寸 #mi(`n\times (n_{h} d_{h})`)。
      ],
    ),
    (
      key: "C_cache^KV",
      "context": "mla",
      preview: [
        MLA 实际缓存的共享 KV 潜在矩阵 #mi(`C^{KV}`)，尺寸 #mi(`n\times d_{c}`)。
      ],
    ),
    (
      key: "K_cache^R",
      "context": "mla",
      preview: [
        MLA 实际缓存的共享 RoPE Key 矩阵 #mi(`K^{R}`)，尺寸 #mi(`n\times d_{h}^{R}`)。
      ],
    ),
    (
      key: "KVCache_j",
      "context": "mla",
      preview: [
        位置 #mi(`j`) 的缓存内容；MHA 存各头完整 #mi(`K/V`)，MLA 只存 #mi(`c_{j}^{KV}`) 与 #mi(`k_{j}^{R}`)。
      ],
    ),
    (
      key: "r_cache",
      "context": "mla",
      preview: [
        MLA 相对同配置 MHA 的缓存标量数比例：#mi(`\frac{d_c+d_h^R}{n_h(2d_h+d_h^R)}`)，示例约 1.41%。
      ],
    ),
    (
      key: "p",
      "context": "mla",
      preview: [
        RoPE 旋转所对应的 token 位置。
      ],
    ),
    (
      key: "R_p",
      "context": "mla",
      preview: [
        位置 #mi(`p`) 的 RoPE 旋转矩阵；随位置变化，原文未为吸收段该简写单独给出统一尺寸。
      ],
    ),
    (
      key: "R_t",
      "context": "mla",
      preview: [
        Query 位置 #mi(`t`) 的 RoPE 旋转矩阵。
      ],
    ),
    (
      key: "R_j",
      "context": "mla",
      preview: [
        历史 Key 位置 #mi(`j`) 的 RoPE 旋转矩阵；#mi(`R_{t}^{T} R_{j}`) 依赖相对位置，不能作固定权重吸收。
      ],
    ),
    (
      key: "x",
      "context": "mla",
      preview: [
        #mi(`\operatorname{RoPE}(x,p)=R_{p} x`) 中待旋转的输入向量。
      ],
    ),
    (
      key: "u_t",
      "context": "moe",
      preview: [
        当前 token 的 FFN/MoE 输入，维度 #mi(`d=7168`)。
      ],
    ),
    (
      key: "N_r",
      "context": "moe",
      preview: [
        当前层路由专家总数；DeepSeek-V3 为 256。
      ],
    ),
    (
      key: "N_s",
      "context": "moe",
      preview: [
        当前层始终激活的共享专家数量；DeepSeek-V3 为 1。
      ],
    ),
    (
      key: "K_r",
      "context": "moe",
      preview: [
        每个 token 选择的路由专家数量；DeepSeek-V3 为 8，不含共享专家。
      ],
    ),
    (
      key: "e_i",
      "context": "moe",
      preview: [
        第 #mi(`i`) 个路由专家的中心向量，用于与 #mi(`d`) 维 FFN 输入计算内积；路由权重每行对应一个专家。
      ],
    ),
    (
      key: "s_it",
      "context": "moe",
      preview: [
        token #mi(`t`) 与路由专家 #mi(`i`) 的原始亲和分数 #mi(`\operatorname{Sigmoid}(u_{t}^T e_{i})`)，为标量。
      ],
    ),
    (
      key: "b_i",
      "context": "moe",
      preview: [
        第 #mi(`i`) 个路由专家的负载均衡选择偏置；只影响 Top-K，不参与最终门控权重，推理时固定。
      ],
    ),
    (
      key: "g'_it",
      "context": "moe",
      preview: [
        路由选择后保留的原始亲和分数；专家入选时为 #mi(`s_{i,t}`)，否则为 0。
      ],
    ),
    (
      key: "g_it",
      "context": "moe",
      preview: [
        由 #mi(`g_{i,t}^{\prime}`) 在所有路由专家间归一化得到的门控标量；未选专家为 0。
      ],
    ),
    (
      key: "h'_t",
      "context": "moe",
      preview: [
        MoE 层残差合并后的输出：输入、共享专家输出及路由专家加权输出之和。
      ],
    ),
    (
      key: "FFN_i^(s)",
      "context": "moe",
      preview: [
        当前层第 #mi(`i`) 个共享专家的 FFN；不经过 Top-K，始终参与计算。
      ],
    ),
    (
      key: "FFN_i^(r)",
      "context": "moe",
      preview: [
        当前层第 #mi(`i`) 个路由专家的 FFN；仅在被 Top-K 选中时计算。
      ],
    ),
    (
      key: "u_t",
      "context": "ffn",
      preview: [
        当前 token 的 FFN 输入向量，维度 #mi(`d`)。
      ],
    ),
    (
      key: "d_ff",
      "context": "ffn",
      preview: [
        SwiGLU 每条输入分支的中间维度；Llama 2-7B 为 11008，DeepSeek-V3 单个路由专家为 2048。
      ],
    ),
    (
      key: "d_dense",
      "context": "ffn",
      preview: [
        DeepSeek-V3 前 3 层稠密 FFN 的中间维度，为 18432。
      ],
    ),
    (
      key: "d_moe",
      "context": "ffn",
      preview: [
        DeepSeek-V3 MoE 单个专家 FFN 的中间维度，为 2048。
      ],
    ),
    (
      key: "W_1",
      "context": "ffn",
      preview: [
        SwiGLU 门控分支投影矩阵，尺寸 #mi(`d_{\mathrm{ff}}\times d`)；投影后施加 SiLU。
      ],
    ),
    (
      key: "W_3",
      "context": "ffn",
      preview: [
        SwiGLU 另一条并行线性分支的投影矩阵，尺寸 #mi(`d_{\mathrm{ff}}\times d`)；不施加激活。
      ],
    ),
    (
      key: "W_2",
      "context": "ffn",
      preview: [
        SwiGLU 输出投影矩阵，尺寸 #mi(`d\times d_{\mathrm{ff}}`)；三个 FFN 投影均无偏置。
      ],
    ),
    (
      key: "W_1i^(r)",
      "context": "ffn",
      preview: [
        第 #mi(`i`) 个路由专家独立的门控分支投影矩阵，对应其 #mi(`W_{1}`)。
      ],
    ),
    (
      key: "W_2i^(r)",
      "context": "ffn",
      preview: [
        第 #mi(`i`) 个路由专家独立的输出投影矩阵，对应其 #mi(`W_{2}`)。
      ],
    ),
    (
      key: "W_3i^(r)",
      "context": "ffn",
      preview: [
        第 #mi(`i`) 个路由专家独立的线性分支投影矩阵，对应其 #mi(`W_{3}`)。
      ],
    ),
    (
      key: "a_t",
      "context": "ffn",
      preview: [
        门控分支激活 #mi(`\operatorname{SiLU}(W_{1} u_{t})`)，维度 #mi(`d_{\mathrm{ff}}`)。
      ],
    ),
    (
      key: "v_t^ff",
      "context": "ffn",
      preview: [
        并行线性分支结果 #mi(`W_{3} u_{t}`)，维度 #mi(`d_{\mathrm{ff}}`)；不是注意力 Value。
      ],
    ),
    (
      key: "z_t",
      "context": "ffn",
      preview: [
        SwiGLU 逐元素门控结果 #mi(`a_{t}\odot v_{t}^{ff}`)，维度 #mi(`d_{\mathrm{ff}}`)。
      ],
    ),
    (
      key: "y_t",
      "context": "ffn",
      preview: [
        单个 FFN 的输出 #mi(`W_{2} z_{t}`)，维度 #mi(`d`)。
      ],
    ),
    (
      key: "P_FFN",
      "context": "ffn",
      preview: [
        单个无偏置 FFN 的三个投影权重参数量 #mi(`3d\,d_{\mathrm{ff}}`)；不含路由器、归一化及量化元数据。
      ],
    ),
    (
      key: "FFN",
      "context": "ffn",
      preview: [
        无偏置 SwiGLU：#mi(`W_{2}[\operatorname{SiLU}(W_{1} u_{t})\odot (W_{3} u_{t})]`)；两条输入分支并行。
      ],
    ),
    (
      key: "SiLU",
      "context": "ffn",
      preview: [
        逐元素激活 #mi(`\operatorname{SiLU}(x)=x \operatorname{Sigmoid}(x)`)。
      ],
    ),
    (
      key: "n_h^I",
      "context": "dsa",
      preview: [
        Lightning Indexer 的头数，为 64；不同于 MLA 的 #mi(`n_{h}=128`)。
      ],
    ),
    (
      key: "d_h^I",
      "context": "dsa",
      preview: [
        Lightning Indexer 每头维度，为 128。
      ],
    ),
    (
      key: "C",
      "context": "dsa",
      preview: [
        上标 #mi(`C`) 表示内容部分。
      ],
    ),
    (
      key: "R",
      "context": "dsa",
      preview: [
        上标 #mi(`R`) 表示 RoPE 位置部分。
      ],
    ),
    (
      key: "I",
      "context": "dsa",
      preview: [
        上标 #mi(`I`) 表示 indexer 相关变量。
      ],
    ),
    (
      key: "bar_c_t^Q",
      "context": "mla",
      preview: [
        共享 Query RMSNorm 后的潜向量，维度 #mi(`d_{c}^{\prime}=1536`)；由 MLA 与 indexer 共用。
      ],
    ),
    (
      key: "gamma^Q",
      "context": "mla",
      preview: [
        MLA Query RMSNorm 的可学习逐维缩放，维度 #mi(`d_{c}^{\prime}=1536`)；不是 indexer 独有参数。
      ],
    ),
    (
      key: "gamma^KV",
      "context": "mla",
      label: [#mi(`\boldsymbol{\gamma}^{KV}`)],
      preview: [
        MLA KV 压缩后的 RMSNorm 可学习逐维缩放，维度 #mi(`d_c=512`)，对应官方 `kv_norm.weight` / HF `kv_a_layernorm.weight`。不作用于独立的 RoPE Key 分支。
      ],
    ),
    (
      key: "epsilon_Q",
      "context": "dsa",
      preview: [
        Query RMSNorm 分母中的稳定项；原文以 #mi(`\varepsilon_{Q}`) 表示，未给出数值。
      ],
    ),
    (
      key: "W^IQ",
      "context": "dsa",
      preview: [
        Indexer Query 上投影矩阵，尺寸 #mi(`(n_{h}^{I} d_{h}^{I})\times d_{c}^{\prime}=8192\times 1536`)。
      ],
    ),
    (
      key: "W^IK",
      "context": "dsa",
      preview: [
        Indexer 共享 Key 投影矩阵，尺寸 #mi(`d_{h}^{I}\times d=128\times 7168`)。
      ],
    ),
    (
      key: "W^IW",
      "context": "dsa",
      preview: [
        Indexer 动态头权重投影矩阵，尺寸 #mi(`n_{h}^{I}\times d=64\times 7168`)；本身为可学习参数。
      ],
    ),
    (
      key: "gamma^I",
      "context": "dsa",
      preview: [
        Indexer Key LayerNorm 的可学习逐维缩放，维度 #mi(`d_{h}^{I}=128`)。
      ],
    ),
    (
      key: "beta^I",
      "context": "dsa",
      preview: [
        Indexer Key LayerNorm 的可学习逐维偏置，维度 #mi(`d_{h}^{I}=128`)。
      ],
    ),
    (
      key: "tilde_q_ti^I",
      "context": "dsa",
      preview: [
        由 #mi(`W^{IQ} \bar{c}_{t}^{Q}`) 按头拆分得到、尚未施加部分 RoPE 的 Query，维度 #mi(`d_{h}^{I}=128`)。
      ],
    ),
    (
      key: "z_j^I",
      "context": "dsa",
      preview: [
        Indexer Key 投影的原始结果 #mi(`W^{IK} h_{j}`)，维度 #mi(`d_{h}^{I}=128`)。
      ],
    ),
    (
      key: "z_ja^I",
      "context": "dsa",
      preview: [
        位置 #mi(`j`) 的原始 indexer Key 投影在第 #mi(`a`) 个特征维上的标量分量。
      ],
    ),
    (
      key: "mu_j",
      "context": "dsa",
      preview: [
        位置 #mi(`j`) 的 Key 投影向量 #mi(`z_{j}^{I}`) 沿 #mi(`d_{h}^{I}`) 个特征维的均值。
      ],
    ),
    (
      key: "sigma_j^2",
      "context": "dsa",
      preview: [
        位置 #mi(`j`) 的 Key 投影向量沿特征维的方差，以 #mi(`d_{h}^{I}`) 为分母。
      ],
    ),
    (
      key: "tilde_k_j^I",
      "context": "dsa",
      preview: [
        #mi(`z_{j}^{I}`) 经带 #mi(`\gamma^{I}`)、#mi(`\beta^{I}`) 的 LayerNorm 后的共享 Key，维度 #mi(`d_{h}^{I}=128`)；尚未施加部分 RoPE。
      ],
    ),
    (
      key: "q_ti^I",
      "context": "dsa",
      preview: [
        Query 位置 #mi(`t`)、第 #mi(`i`) 个 indexer 头的 Query，维度 #mi(`d_{h}^{I}=128`)；已完成部分 RoPE。
      ],
    ),
    (
      key: "k_j^I",
      "context": "dsa",
      preview: [
        位置 #mi(`j`)、所有 indexer 头共享的 Key，维度 #mi(`d_{h}^{I}=128`)；已完成部分 RoPE。
      ],
    ),
    (
      key: "q^I",
      "context": "dsa",
      preview: [
        Indexer Query 的统称；每个头的向量维度为 #mi(`d_{h}^{I}`)。
      ],
    ),
    (
      key: "k^I",
      "context": "dsa",
      preview: [
        Indexer 所有头共享的 Key 的统称；向量维度为 #mi(`d_{h}^{I}`)。
      ],
    ),
    (
      key: "a_ti^I",
      "context": "dsa",
      preview: [
        头权重投影 #mi(`[W^{IW} h_{t}]`);\_i 的标量结果，随 token 改变。
      ],
    ),
    (
      key: "w_ti^I",
      "context": "dsa",
      preview: [
        Indexer MatMul / #mi(`\sqrt{n_h^I d_h^I}`) 的标量激活 #mi(`w_{t,i}^I=[W^{IW}\mathbf h_t]_i/\sqrt{n_h^I d_h^I}`)；固定缩放可等价折叠到权重 #mi(`\bar W^{IW}=W^{IW}/\sqrt{n_h^I d_h^I}`)。不是可学习权重。
      ],
    ),
    (
      key: "w^I",
      "context": "dsa",
      preview: [
        索引评分中的动态头权重激活的统称；由 #mi(`W^{IW}`) 对输入投影并缩放得到，不是独立可学习参数。
      ],
    ),
    (
      key: "I_tj",
      "context": "dsa",
      preview: [
        Query #mi(`t`) 对历史位置 #mi(`j`) 的标量索引分数：各 indexer 头的 #mi(`\operatorname{ReLU}(Q\cdot K)`) 乘动态头权重后求和。
      ],
    ),
    (
      key: "I_tj^HF",
      "context": "dsa",
      preview: [
        HF 路径的标量索引分数；将头数缩放与头维缩放展开后等于所定义的 #mi(`w_{t,i}^{I}`) 加权评分。
      ],
    ),
    (
      key: "I_tj^official",
      "context": "dsa",
      preview: [
        官方 Hadamard/FP8 路径的标量索引分数；因量化仅近似等于 HF 分数。
      ],
    ),
    (
      key: "calS_t",
      "context": "dsa",
      preview: [
        Query #mi(`t`) 选中的 KV 位置集合；是集合而非打分矩阵 #mi(`S_{t}`)，也不是位置对数 #mi(`S`)。
      ],
    ),
    (
      key: "k",
      "context": "dsa",
      preview: [
        DSA 每个 Query 所选 KV 位置数量的上限；V3.2 为 2048。
      ],
    ),
    (
      key: "m",
      "context": "dsa",
      preview: [
        部分 RoPE 的半宽 #mi(`m=d_{h}^{R}/2=32`)；与复杂度段的 #mi(`m=\operatorname{min}(n,k)`) 不同。
      ],
    ),
    (
      key: "a",
      "context": "dsa",
      preview: [
        Key 归一化中的特征维索引；部分 RoPE 配对中 #mi(`a=1,\ldots,m`)，配对维为 #mi(`(a,a+m)`)。
      ],
    ),
    (
      key: "p",
      "context": "dsa",
      preview: [
        部分 RoPE 所在的 token 位置；Query 用 #mi(`t`)，Key 用 #mi(`j`)。
      ],
    ),
    (
      key: "D_cos,p^I",
      "context": "dsa",
      preview: [
        位置 #mi(`p`) 的前 #mi(`m`) 个余弦值构成的对角矩阵，尺寸 #mi(`m\times m=32\times 32`)。
      ],
    ),
    (
      key: "D_sin,p^I",
      "context": "dsa",
      preview: [
        位置 #mi(`p`) 的前 #mi(`m`) 个正弦值构成的对角矩阵，尺寸 #mi(`m\times m=32\times 32`)。
      ],
    ),
    (
      key: "cos_p",
      "context": "dsa",
      preview: [
        实现中位置 #mi(`p`) 的余弦位置编码；前 #mi(`m`) 个值用于构造 #mi(`D_{\cos,p}^{I}`)。
      ],
    ),
    (
      key: "sin_p",
      "context": "dsa",
      preview: [
        实现中位置 #mi(`p`) 的正弦位置编码；前 #mi(`m`) 个值用于构造 #mi(`D_{\sin,p}^{I}`)。
      ],
    ),
    (
      key: "R_p^I",
      "context": "dsa",
      preview: [
        Indexer 前 #mi(`d_{h}^{R}`) 维的 half-split RoPE 旋转矩阵，尺寸 #mi(`d_{h}^{R}\times d_{h}^{R}=64\times 64`)。
      ],
    ),
    (
      key: "P_p^I",
      "context": "dsa",
      preview: [
        Indexer 部分 RoPE 变换 #mi(`\operatorname{diag}(R_{p}^{I},`);Id)，尺寸 #mi(`d_{h}^{I}\times d_{h}^{I}=128\times 128`)；后 64 维直通。
      ],
    ),
    (
      key: "P_t^I",
      "context": "dsa",
      preview: [
        Query 位置 #mi(`t`) 的 indexer 部分 RoPE 变换，尺寸 #mi(`128\times 128`)。
      ],
    ),
    (
      key: "P_j^I",
      "context": "dsa",
      preview: [
        Key 位置 #mi(`j`) 的 indexer 部分 RoPE 变换，尺寸 #mi(`128\times 128`)。
      ],
    ),
    (
      key: "Id",
      "context": "dsa",
      preview: [
        指定阶数的单位矩阵；部分 RoPE 的直通分块阶数为 #mi(`d_h^I-d_h^R`)，Hadamard 正交关系中为 #mi(`d_h^I`)。
      ],
    ),
    (
      key: "1",
      "context": "dsa",
      preview: [
        Key LayerNorm 中与 #mi(`z_{j}^{I}`) 同维的全 1 向量。
      ],
    ),
    (
      key: "calH^I",
      "context": "dsa",
      preview: [
        归一化 Hadamard 矩阵 #mi(`\operatorname{Hadamard}_{d_h^I}/\sqrt{d_h^I}`)，尺寸 #mi(`d_h^I\times d_h^I=128\times128`)，保持未量化点积。
      ],
    ),
    (
      key: "Hadamard",
      "context": "dsa",
      preview: [
        用于官方旋转激活路径的 Hadamard 变换；除以 #mi(`\sqrt{d_h^I}`) 后为正交变换。
      ],
    ),
    (
      key: "hat_q_ti^I",
      "context": "dsa",
      preview: [
        官方路径中 Query 经归一化 Hadamard 和 FP8 量化—反量化后的有效向量，维度 #mi(`d_{h}^{I}=128`)。
      ],
    ),
    (
      key: "hat_k_j^I",
      "context": "dsa",
      preview: [
        官方路径中共享 Key 经归一化 Hadamard 和 FP8 量化—反量化后的有效向量，维度 #mi(`d_{h}^{I}=128`)。
      ],
    ),
    (
      key: "calQ_FP8",
      "context": "dsa",
      preview: [
        含 scale 的 FP8 量化—反量化有效值映射；量化后不保证索引分数或 Top-K 与 HF 完全相同。
      ],
    ),
    (
      key: "RMSNorm",
      "context": "mla",
      preview: [
        沿潜向量特征维做均方根归一化，再施加可学习逐维缩放；Query 为 #mi(`\gamma^Q\odot\mathbf c_t^Q/\sqrt{\lVert\mathbf c_t^Q\rVert_2^2/d_c^{\prime}+\varepsilon_Q}`)，KV 为同样的 #mi(`d_c=512`) 维运算。它依赖当前激活，不能合入固定投影权重。
      ],
    ),
    (
      key: "LayerNorm",
      "context": "dsa",
      preview: [
        Indexer Key 沿特征维减均值、除标准差后施加 #mi(`\gamma^{I}`) 和 #mi(`\beta^{I}`)；稳定项为 #mi(`10^-6`)。
      ],
    ),
    (
      key: "MatMulLayerNorm",
      "context": "dsa",
      label: [MatMulLayerNorm],
      preview: [
        #strong[融合算子的数学展开（MatMul → LayerNorm）]

        #mi(`\operatorname{MatMulLayerNorm}(\mathbf x,W,\mathbf g,\mathbf b)`)
        \ #mi(`=\operatorname{LayerNorm}_{\mathbf g,\mathbf b}(W\mathbf x)`)
        \ #mi(`=\mathbf g\odot\frac{W\mathbf x-\mu}{\sqrt{\sigma^2+\epsilon}}+\mathbf b`)

        将统计量归约与逐元素后处理拆开：

        #mi(`\mathbf z=W\mathbf x,\quad \mu=\frac{1}{m}\sum_{a=1}^{m}z_a`)
        \ #mi(`\sigma^2=\frac{1}{m}\sum_{a=1}^{m}(z_a-\mu)^2`)
        \ #mi(`\rho=\operatorname{rsqrt}(\sigma^2+\epsilon)`)
        \ #mi(`\operatorname{MatMulLayerNorm}(\mathbf x,W,\mathbf g,\mathbf b)`)
        \ #mi(`=(\mathbf z-\mu)\odot(\mathbf g\rho)+\mathbf b`)

        #strong[Prologue：准备 MatMul 输入]

        读取 hidden state #mi(`\mathbf x`) 和投影权重 #mi(`W`)，准备矩阵乘法的数据块；读取缩放 #mi(`\mathbf g`) 和偏置 #mi(`\mathbf b`) 供 Epilogue 使用。随后 MatMul 主循环累加得到 #mi(`\mathbf z=W\mathbf x`)。

        #strong[Epilogue：归约、归一化、仿射变换、写出]

        ① 在完整输出行上求均值 #mi(`\mu`)；② 求中心化平方偏差的均值 #mi(`\sigma^2`)；③ 计算倒数标准差 #mi(`\rho=\operatorname{rsqrt}(\sigma^2+\epsilon)`)；④ 逐元素计算 #mi(`y_a=(z_a-\mu)g_a\rho+b_a`)，直接写出最终结果，避免单独写回并重读未归一化的 #mi(`\mathbf z`)。跨 tile 的行归约、同步及数值稳定性仍需内核保证。

        #strong[均值与方差取 MatMul 输出，而非输入 hidden state。] 本图取 #mi(`\mathbf x=\mathbf h_j,\ W=W^{IK},\ m=128`)、#mi(`\mathbf g=\boldsymbol\gamma^I,\ \mathbf b=\boldsymbol\beta^I,\ \epsilon=10^{-6}`)，输出为部分 RoPE 前的 #mi(`\widetilde{\mathbf k}_j^I`)。公式忽略浮点舍入；逻辑融合不等于声明官方已实现单个融合 kernel。
      ],
    ),
    (
      key: "reshape",
      "context": "dsa",
      preview: [
        将 indexer Query 投影结果重排为 #mi(`n_{h}^{I}\times d_{h}^{I}`)，再取第 #mi(`i`) 行得到每头向量。
      ],
    ),
    (
      key: "ReLU",
      "context": "dsa",
      preview: [
        逐元素整流激活；索引评分中作用于每个头的 Query-Key 标量内积。
      ],
    ),
    (
      key: "norm_2",
      "context": "dsa",
      preview: [
        欧氏范数；Query RMSNorm 使用其平方计算逐维均方值。
      ],
    ),
    (
      key: "n",
      "context": "complexity",
      preview: [
        可见 token 数，包含当前 token；完整 Prefill 为长度 #mi(`n`) 的单序列，Decode 为单个新 token。
      ],
    ),
    (
      key: "m",
      "context": "complexity",
      preview: [
        末位置实际保留数量 #mi(`m=\operatorname{min}(n,k)`)；不是部分 RoPE 的半宽 #mi(`m`)。
      ],
    ),
    (
      key: "m_t",
      "context": "dsa",
      preview: [
        Query 位置 #mi(`t`) 实际选中 KV 数量 #mi(`|\mathcal{S}_{t}|=\operatorname{min}(t,k)`)。
      ],
    ),
    (
      key: "P",
      "context": "complexity",
      preview: [
        完整因果 Prefill 的可见 Query-Key 位置对总数 #mi(`n(n+1)/2`)；为标量计数。
      ],
    ),
    (
      key: "S",
      "context": "complexity",
      preview: [
        稀疏主注意力的 Prefill 位置对总数 #mi(`\sum_t\min(t,k)=nm-\frac{m(m-1)}{2}`)；不是矩阵 #mi(`S_t`) 或集合 #mi(`\mathcal{S}_t`)。
      ],
    ),
    (
      key: "Q_t^QK",
      "context": "complexity",
      preview: [
        固定 Query #mi(`t`) 的吸收后内容查询矩阵；各行是 #mi(`\bar{c}_{t}^{Q}`) 转置乘 #mi(`W_{i}^{QK}`)，尺寸 #mi(`n_{h}\times d_{c}=128\times 512`)。
      ],
    ),
    (
      key: "Q_t^R",
      "context": "complexity",
      preview: [
        固定 Query #mi(`t`) 的各头 RoPE Query 按行堆叠，尺寸 #mi(`n_{h}\times d_{h}^{R}=128\times 64`)。
      ],
    ),
    (
      key: "C_le_t^KV",
      "context": "complexity",
      preview: [
        历史 #mi(`1,\ldots,t`) 的共享 KV 潜向量按行堆叠，尺寸 #mi(`t\times d_{c}=t\times 512`)。
      ],
    ),
    (
      key: "K_le_t^R",
      "context": "complexity",
      preview: [
        历史 #mi(`1,\ldots,t`) 的共享 RoPE Key 按行堆叠，尺寸 #mi(`t\times d_{h}^{R}=t\times 64`)。
      ],
    ),
    (
      key: "C_calS_t^KV",
      "context": "complexity",
      preview: [
        按所选位置集合 #mi(`\mathcal{S}_{t}`) 取出的共享 KV 潜在矩阵，尺寸 #mi(`m_{t}\times d_{c}=m_{t}\times 512`)。
      ],
    ),
    (
      key: "K_calS_t^R",
      "context": "complexity",
      preview: [
        按所选位置集合 #mi(`\mathcal{S}_{t}`) 取出的共享 RoPE Key 矩阵，尺寸 #mi(`m_{t}\times d_{h}^{R}=m_{t}\times 64`)。
      ],
    ),
    (
      key: "S_t",
      "context": "complexity",
      preview: [
        固定 Query #mi(`t`) 的未缩放 MLA 打分矩阵；Full 为 #mi(`n_{h}\times t`)，Sparse 为 #mi(`n_{h}\times m_{t}`)；包含内容与 RoPE 点积。
      ],
    ),
    (
      key: "A_t",
      "context": "complexity",
      preview: [
        #mi(`\operatorname{Softmax}_j(S_t/\sqrt{d_h+d_h^R})`)；Full 尺寸 #mi(`n_h\times t`)，Sparse 尺寸 #mi(`n_h\times m_t`)。
      ],
    ),
    (
      key: "tilde_O_t",
      "context": "complexity",
      preview: [
        压缩空间的 Value 聚合结果 #mi(`A_{t} C`)，尺寸 #mi(`n_{h}\times d_{V}=128\times 512`)；不是最终 #mi(`d`) 维输出。
      ],
    ),
    (
      key: "Q_t^I",
      "context": "complexity",
      preview: [
        位置 #mi(`t`) 的各 indexer 头已做部分 RoPE 的 Query 按行堆叠，尺寸 #mi(`n_{h}^{I}\times d_{h}^{I}=64\times 128`)。
      ],
    ),
    (
      key: "K_le_t^I",
      "context": "complexity",
      preview: [
        历史共享 indexer Key 按行堆叠，尺寸 #mi(`t\times d_{h}^{I}=t\times 128`)。
      ],
    ),
    (
      key: "B_t",
      "context": "complexity",
      preview: [
        Indexer 各头的全历史内积矩阵 #mi(`Q_{t}^{I}(K_{\le t}^{I})^T`)，尺寸 #mi(`n_{h}^{I}\times t=64\times t`)；尚非跨头汇总后的 #mi(`I_{t,j}`)。
      ],
    ),
    (
      key: "d_K",
      "context": "complexity",
      preview: [
        MLA 吸收后每头 QK 点积有效维度 #mi(`d_{c}+d_{h}^{R}=576`)；不是未吸收头维度 192。
      ],
    ),
    (
      key: "d_V",
      "context": "complexity",
      preview: [
        MLA 吸收后的 Value 聚合维度 #mi(`d_{c}=512`)。
      ],
    ),
    (
      key: "F_Indexer",
      "context": "complexity",
      preview: [
        Indexer 的位置对点积 FLOPs 主项；只计 #mi(`n_{h}^{I} d_{h}^{I}`) 维点积，不含预处理及 Top-K 等。
      ],
    ),
    (
      key: "F_Full",
      "context": "complexity",
      preview: [
        Full MLA 的 QK 点积与 Value 聚合 FLOPs 主项；用于相对成本基准。
      ],
    ),
    (
      key: "r_prefill",
      "context": "complexity",
      preview: [
        DSA 完整因果 Prefill 相对 Full MLA 的主项成本比 #mi(`1/17+S/P`)。
      ],
    ),
    (
      key: "r_decode",
      "context": "complexity",
      preview: [
        DSA 单 token Decode 相对 Full MLA 的主项成本比 #mi(`1/17+m/n`)。
      ],
    ),
    (
      key: "r",
      "context": "complexity",
      preview: [
        相对主项成本比的泛指；净节省比例为 1-r，不是实测整模型加速比。
      ],
    ),
    (
      key: "O",
      "context": "complexity",
      preview: [
        大 #mi(`O`) 渐近时间复杂度记号；不是注意力输出矩阵 #mi(`O`)。
      ],
    ),
    (
      key: "log",
      "context": "complexity",
      preview: [
        对数因子；用于快速 Hadamard、排序及堆式 Top-K 的复杂度，原文未指定底数。
      ],
    ),
    (
      key: "min",
      "context": "complexity",
      preview: [
        取较小值；实际稀疏位置数受可见位置数与 #mi(`k`) 共同限制。
      ],
    ),
    (
      key: "MAC",
      "context": "complexity",
      preview: [
        一次乘加；本节按 1 MAC=2 FLOPs 统计。
      ],
    ),
    (
      key: "FLOPs",
      "context": "complexity",
      preview: [
        浮点运算次数；本节统计单层、单序列推理的指定主项，并非硬件吞吐率。
      ],
    ),
  ),
)
