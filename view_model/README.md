https://sebastianraschka.com/llm-architecture-gallery/

uv venv --python 3.12 --seed --managed-python
source gitee-proxy.sh

```bash
euporie-preview --run --save linear.ipynb
jupyter lab --ip=0.0.0.0 --custom-css
```

## 经典模型

- Llama2
- DeepSeek V3.1: GQA, MHA -> MLA, MoE
- DeepSeek V3.2: topk稀疏注意力
    - GLM5.1
    - GLM5.2
    - GLM5.3
    - GLM5.3 flash
- DeepSeek V4, V4.1
- Qwen-Next: mamba full
    - Qwen3.5
    - Qwen3.6
    - Qwen3.8
- Kimi K3: KDA

TP + X特性
## DeepSeek V3.2 Indexer

在 `view_model` 目录执行：

```bash
.venv/bin/python deepseek/deepseek_v32_indexer.py
```

按官方配置的真实维度构建单个 `DeepseekV32Indexer`，以 BF16 随机权重和随机输入执行 forward，打印模块、参数形状与 token 索引。不下载配置或模型权重；需安装支持 `deepseek_v32` 的 Transformers。默认输入为 8 个 token，可修改脚本中的 `SEQUENCE_LENGTH`；`index_topk` 保持官方值 2048。


## LLaMA Attention 文档

`llama2_7b/llama_attention.typ` 保留原 Markdown 的全部文字、公式、代码和链接，与 DeepSeek 文档共同引用 `typst_template/web-template.typ`；样式和交互代码统一维护在该目录，不在各文档中复制。

在 `view_model` 目录编译或启动实时预览：

```bash
typst compile --root . --features html llama2_7b/llama_attention.typ llama2_7b/llama_attention.html
typst watch --root . --features html --port 3000 llama2_7b/llama_attention.typ llama2_7b/llama_attention.html
```

从 `llama2_7b` 或 `deepseek` 子目录构建时使用 `--root ..`，以允许引用公共模板。

## Attention 静态图

图中蓝色圆角框表示激活／张量，紫色直角框表示算子，橙色小卡片表示权重参数。权重从侧面接入 MatMul，浅灰透明的 Head 外框只表示分组；右上角图例与成品一起生成。布局采用紧凑间距，保留三列及输出链对齐，不缩小公式字号。

- [MHA](llama2_7b/mha.svg)：各 Query 头使用各自的 K/V。
- [Llama 2 70B GQA](llama2_7b/gqa.svg)：64 个 Query 头、8 个 KV 头、每头 128 维；每 8 个 Query 头共享一对 K/V，但独立计算 softmax 和输出。图中展开第 1 组（Q1–Q8）和第 8 组（Q57–Q64），其余组和组内部分头用省略号表示。Q 投影宽度为 8192，K/V 投影宽度各为 1024；图中省略 batch，展示 prefill 自注意力。

编辑 `llama2_7b/mha.source.svg` 或 `llama2_7b/gqa.source.svg` 中的布局和 `data-tex` 公式，使用共用生成器 `render_attention.mjs` 生成对应静态 SVG：

```bash
cd llama2_7b
npm ci
npm run render:mha
npm run render:gqa
```

MathJax 仅在生成时使用，成品可以直接在 Markdown 中嵌入。生成器将公式输出为 `<g>` 路径组，只保留一个最外层 `<svg>`；不要重新引入嵌套 `<svg>`，避免 Live Server 在多个 `</svg>` 处重复注入自动刷新脚本，导致 HTTP 响应截断和 XML 解析失败。

Llama 2 70B 参数可对照[公开 config.json](https://huggingface.co/NousResearch/Llama-2-70b-hf/blob/main/config.json)：`hidden_size=8192`、`num_attention_heads=64`、`num_key_value_heads=8`。
