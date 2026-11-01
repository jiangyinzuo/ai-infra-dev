uv venv --python 3.12 --seed --managed-python
source gitee-proxy.sh

```bash
euporie-preview --run --save linear.ipynb
jupyter lab --ip=0.0.0.0 --custom-css
```

## MHA 静态图

图中蓝色圆角框表示激活／张量，紫色直角框表示算子，橙色小卡片表示权重参数。权重从侧面接入 MatMul，浅灰透明的 Head 外框只表示分组；右上角图例与成品一起生成。布局采用紧凑间距，保留三列及输出链对齐，不缩小公式字号。

编辑 `llama2_7b/mha.source.svg` 中的布局和 `data-tex` 公式，重新生成 `llama2_7b/mha.svg`：

```bash
cd llama2_7b
npm ci
npm run render:mha
```

MathJax 仅在生成时使用，成品可以直接在 Markdown 中嵌入。生成器将公式输出为 `<g>` 路径组，只保留一个最外层 `<svg>`；不要重新引入嵌套 `<svg>`，避免 Live Server 在多个 `</svg>` 处重复注入自动刷新脚本，导致 HTTP 响应截断和 XML 解析失败。
