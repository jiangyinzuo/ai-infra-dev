// MLA before matrix absorption, following the equations in the article.
// Compile standalone: typst compile diagrams/deepseek-mla.typ diagrams/deepseek-mla.svg
// Query uses token t; KV uses every visible token j <= t. RMSNorm is omitted as in the report formulas.
// Node-sized SVG links carry glossary keys only; the webpage resolves rich previews from symbols.typ.
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/mitex:0.2.7": mi

#let mla-dimensions = (
  hidden: 7168,
  heads: 128,
  content: 128,
  rope: 64,
  query-latent: 1536,
  kv-latent: 512,
)

// Only keys are stored here. Descriptions and rich previews come from the article glossary.
#let mla-node-symbols = (
  output: ("u_t", "d", "t"),
  heads: ("o_ti", "n_h", "d_h", "t"),
  head-out: ("o_ti", "d_h", "t", "i"),
  attention: ("A_tji", "t", "j", "i"),
  scores: ("S_tji", "t", "j", "i"),
  q: ("q_ti", "q_ti^C", "q_ti^R", "d_h", "d_h^R", "t", "i"),
  k: ("k_ji", "k_ji^C", "k_j^R", "d_h", "d_h^R", "j", "i"),
  qc: ("q_ti^C", "d_h", "t", "i"),
  qr: ("q_ti^R", "d_h^R", "t", "i"),
  kc: ("k_ji^C", "d_h", "j", "i"),
  v: ("v_ji^C", "d_h", "j", "i"),
  kr: ("k_j^R", "d_h^R", "j"),
  cq: ("c_t^Q", "d_c'", "t"),
  ckv: ("c_j^KV", "d_c", "j"),
  input-q: ("h_t", "d", "t"),
  input-kv: ("h_j", "d", "j", "t"),
  heads-concat: ("concat", "n_h"),
  q-concat: ("concat",),
  k-concat: ("concat",),
  weighted-sum: ("A_tji", "v_ji^C", "j", "t", "i"),
  softmax: ("Softmax_j", "j", "t"),
  qk: ("q_ti", "k_ji", "T", "d_h", "d_h^R", "sqrt", "t", "j", "i"),
  q-rope: ("RoPE", "t"),
  k-rope: ("RoPE", "j"),
  w-out-proj: ("W^O", "d", "n_h", "d_h"),
  w-q-up: ("W_i^UQ", "d_h", "d_c'", "i"),
  w-qr-proj: ("W_i^QR", "d_h^R", "d_c'", "i"),
  w-k-up: ("W_i^UK", "d_h", "d_c", "i"),
  w-v-up: ("W_i^UV", "d_h", "d_c", "i"),
  w-q-down: ("W^DQ", "d_c'", "d"),
  w-kv-down: ("W^DKV", "d_c", "d"),
  w-kr-proj: ("W^KR", "d_h^R", "d"),
)

#let mla-diagram() = {
  set text(font: ("Noto Sans CJK SC", "New Computer Modern"), size: 10pt, fill: rgb("202735"))
  set par(leading: 4pt)
  let d = mla-dimensions
  let full-head = d.content + d.rope
  let all-values = d.heads * d.content
  assert.eq(full-head, 192)
  assert.eq(all-values, 16384)

  // Keep layout rows top-to-bottom; Fletcher physical coordinates use an upward y axis.
  let at(x, y) = (x * 1pt, -y * 1pt)
  let ink = rgb("485466")
  let tensor-ink = rgb("245e91")
  let parameter-ink = rgb("79517d")
  let cache-ink = rgb("1665ac")
  let preview-region(pos, id, width, height, keys: none) = {
    let keys = if keys == none { mla-node-symbols.at(id, default: ()) } else { keys }
    if keys.len() > 0 {
      // Typst exports the transparent link rectangle at the exact Fletcher node bounds.
      let target = "#diagram-symbols/mla/" + id + "/" + keys.join("|")
      node(pos, link(target, box(width: width, height: height)),
        name: label("preview-" + id), shape: rect, width: width, height: height,
        inset: 0pt, fill: none, stroke: none, snap: false, layer: 10)
    }
  }
  let tensor(pos, id, symbol, dimension, cache: false, width: 136pt) = {
    node(
    pos,
    align(center)[#mi(symbol) \ #text(9pt, mi(dimension))],
    name: label(id), shape: rect, width: width, height: 46pt,
    corner-radius: 7pt, inset: 5pt,
    fill: if cache { rgb("e4f2ff") } else { rgb("f1f7fc") },
    stroke: (paint: if cache { cache-ink } else { tensor-ink }, thickness: if cache { 1.3pt } else { 0.85pt }),
    extrude: if cache { (0, 2.5) } else { (0,) },
    )
    preview-region(pos, id, width, 46pt)
  }
  let operator(pos, id, title, formula: none, width: 90pt) = {
    node(
    pos,
    align(center)[#text(weight: "bold", title)#if formula != none [\ #text(9pt, mi(formula))]],
    name: label(id), shape: rect, width: width,
    height: if formula == none { 29pt } else { 43pt },
    inset: 5pt, fill: rgb("f3f3f4"), stroke: 0.85pt + ink,
    )
    preview-region(pos, id, width, if formula == none { 29pt } else { 43pt })
  }
  let projection(x, y, id, weight, rows, cols, shape, weight-offset: (-108, 0), weight-width: 100pt) = {
    node(
      at(x + weight-offset.at(0), y + weight-offset.at(1)),
      align(center)[#mi(weight) \ #text(8.5pt, mi(shape.text + "=" + str(rows) + "\\times" + str(cols)))],
      name: label("w-" + id), shape: rect, width: weight-width, height: 42pt,
      inset: 4pt, fill: rgb("faf4fa"),
      stroke: (paint: parameter-ink, thickness: 0.85pt, dash: "dashed"),
    )
    preview-region(at(x + weight-offset.at(0), y + weight-offset.at(1)), "w-" + id, weight-width, 42pt)
    operator(at(x, y), id, [MatMul], width: 68pt)
    preview-region(at(x, y), id, 68pt, 29pt, keys: ("MatMul",))
    edge(label("w-" + id), label(id), "--|>", stroke: parameter-ink + 0.8pt)
  }
  let wire(..args) = edge(..args, crossing: true, crossing-thickness: 6)

  diagram(
    node-shape: rect,
    edge-stroke: 0.9pt + ink,
    edge-corner-radius: 5pt,
    mark-scale: 70%,
    crossing-fill: white,
    {
      // Legend: shapes encode tensor / parameter / operator; blue double frames mark stored KV.
      tensor(at(55, 70), "legend-tensor", `\mathbf{x}`, `\mathrm{Activation}`, width: 128pt)
      node(at(55, 132), align(center)[#mi(`W`) \ 可学习权重], shape: rect,
        width: 128pt, height: 42pt, fill: rgb("faf4fa"),
        stroke: (paint: parameter-ink, thickness: 0.85pt, dash: "dashed"))
      operator(at(55, 192), "legend-operator", [算子 Operator], width: 128pt)
      tensor(at(55, 255), "legend-cache", `\mathrm{KV\ Cache}`, `\mathrm{stored\ activation}`, cache: true, width: 128pt)
      edge(at(15, 325), at(95, 325))
      wire(at(55, 345), at(55, 305))
      node(at(175, 325), [跨线不连接], stroke: none, width: 120pt, snap: false)
      node(at(750, 145), align(left)[
        *DeepSeek MLA · 未吸收路径* \
        数据自下向上流动；虚线为权重输入。 \
        #mi(`i=1,\ldots,n_h,\quad j=1,\ldots,t`) \
        #mi(`n_h=`.text + str(d.heads) + `,\ d_h=`.text + str(d.content) + `,\ d_h^R=`.text + str(d.rope)) \
        #mi(`W_i^{UQ},W_i^{QR},W_i^{UK},W_i^{UV}`) 为按头分块。 \
        按正文报告公式省略压缩后的 RMSNorm。 \
        不包含残差连接、前置归一化或 FFN。
      ], stroke: none, width: 290pt, inset: 4pt, snap: false)

      // Final multi-head output.
      tensor(at(415, 0), "output", `\mathbf{u}_t\in\mathbb{R}^{d}`.text, `d=`.text + str(d.hidden))
      projection(415, 65, "out-proj", `W^O`, d.hidden, all-values, `d\times(n_h d_h)`, weight-offset: (-155, 0), weight-width: 155pt)
      tensor(at(415, 130), "heads", `[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_h}]`.text, `\in\mathbb{R}^{n_h d_h}=\mathbb{R}^{`.text + str(all-values) + `}`.text, width: 200pt)
      operator(at(415, 195), "heads-concat", [Concat · 全部头], width: 136pt)
      tensor(at(415, 260), "head-out", `\mathbf{o}_{t,i}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content))
      operator(at(415, 325), "weighted-sum", [Value 加权聚合], formula: `\sum_{j=1}^{t}A_{t,j,i}\mathbf{v}_{j,i}^{C}`, width: 240pt)
      tensor(at(415, 395), "attention", `[A_{t,1,i},\ldots,A_{t,t,i}]`, `\in\mathbb{R}^{t}`, width: 190pt)
      operator(at(415, 460), "softmax", [Causal Softmax], formula: `\operatorname{Softmax}_{j\le t}`, width: 155pt)
      tensor(at(415, 525), "scores", `[S_{t,1,i},\ldots,S_{t,t,i}]`, `\in\mathbb{R}^{t}`, width: 190pt)
      operator(at(415, 595), "qk", [QK 点积与缩放], formula: `\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_h+d_h^R}}`, width: 240pt)

      // Full Q/K and their content / positional components.
      tensor(at(185, 665), "q", `\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}]`.text, `\in\mathbb{R}^{d_h+d_h^R}=\mathbb{R}^{`.text + str(full-head) + `}`.text, width: 230pt)
      tensor(at(645, 665), "k", `\mathbf{k}_{j,i}=[\mathbf{k}_{j,i}^{C};\mathbf{k}_{j}^{R}]`.text, `\in\mathbb{R}^{d_h+d_h^R}=\mathbb{R}^{`.text + str(full-head) + `}`.text, width: 230pt)
      operator(at(185, 730), "q-concat", [Concat], width: 90pt)
      operator(at(645, 730), "k-concat", [Concat], width: 90pt)
      tensor(at(95, 800), "qc", `\mathbf{q}_{t,i}^{C}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content))
      tensor(at(280, 800), "qr", `\mathbf{q}_{t,i}^{R}\in\mathbb{R}^{d_h^R}`.text, `d_h^R=`.text + str(d.rope))
      tensor(at(560, 800), "kc", `\mathbf{k}_{j,i}^{C}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content))
      tensor(at(900, 800), "v", `\mathbf{v}_{j,i}^{C}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content))
      operator(at(280, 865), "q-rope", [RoPE], formula: `\mathrm{position}\ t`, width: 90pt)
      projection(95, 935, "q-up", `W_i^{UQ}`, d.content, d.query-latent, `d_h\times d_c^{\prime}`)
      projection(280, 935, "qr-proj", `W_i^{QR}`, d.rope, d.query-latent, `d_h^R\times d_c^{\prime}`, weight-offset: (0, 45))
      projection(560, 935, "k-up", `W_i^{UK}`, d.content, d.kv-latent, `d_h\times d_c`)
      projection(900, 935, "v-up", `W_i^{UV}`, d.content, d.kv-latent, `d_h\times d_c`, weight-offset: (108, 0))
      tensor(at(730, 935), "kr", `\mathbf{k}_{j}^{R}\in\mathbb{R}^{d_h^R}`.text, `d_h^R=`.text + str(d.rope) + `\quad\text{（所有头共享）}`.text, cache: true, width: 150pt)

      // Shared low-rank representations and independent RoPE Key projection.
      tensor(at(195, 1025), "cq", `\mathbf{c}_t^{Q}\in\mathbb{R}^{d_c^{\prime}}`.text, `d_c^{\prime}=`.text + str(d.query-latent))
      tensor(at(560, 1025), "ckv", `\mathbf{c}_j^{KV}\in\mathbb{R}^{d_c}`.text, `d_c=`.text + str(d.kv-latent), cache: true)
      operator(at(730, 1025), "k-rope", [RoPE], formula: `\mathrm{position}\ j`, width: 90pt)
      projection(195, 1115, "q-down", `W^{DQ}`, d.query-latent, d.hidden, `d_c^{\prime}\times d`)
      projection(560, 1115, "kv-down", `W^{DKV}`, d.kv-latent, d.hidden, `d_c\times d`)
      projection(730, 1115, "kr-proj", `W^{KR}`, d.rope, d.hidden, `d_h^R\times d`, weight-offset: (108, 0))
      tensor(at(195, 1200), "input-q", `\mathbf{h}_t\in\mathbb{R}^{d}`.text, `d=`.text + str(d.hidden))
      tensor(at(645, 1200), "input-kv", `\mathbf{h}_j\in\mathbb{R}^{d}`.text, `d=`.text + str(d.hidden) + `,\quad j\le t`.text)

      // Data edges, listed separately so repositioning nodes cannot change dependencies.
      edge(<input-q>, <q-down>, "-|>")
      edge(<q-down>, <cq>, "-|>")
      edge(<cq.north>, at(195, 980), at(95, 980), <q-up.south>, "-|>")
      edge(<cq.east>, at(350, 1025), at(350, 935), <qr-proj.east>, "-|>")
      edge(<q-up>, <qc>, "-|>")
      edge(<qr-proj>, <q-rope>, "-|>")
      edge(<q-rope>, <qr>, "-|>")
      edge(<qc.north>, at(95, 760), at(160, 760), <q-concat.south>, "-|>")
      edge(<qr.north>, at(280, 760), at(210, 760), <q-concat.south>, "-|>")
      edge(<q-concat>, <q>, "-|>")

      edge(<input-kv.north>, at(645, 1160), at(560, 1160), <kv-down.south>, "-|>")
      edge(<input-kv.north>, at(645, 1160), at(730, 1160), <kr-proj.south>, "-|>")
      edge(<kv-down>, <ckv>, "-|>")
      edge(<ckv>, <k-up>, "-|>")
      edge(<k-up>, <kc>, "-|>")
      edge(<v-up>, <v>, "-|>")
      edge(<kr-proj>, <k-rope>, "-|>")
      edge(<k-rope>, <kr>, "-|>")
      // This Value branch crosses the independent RoPE-Key branch; the white halo means no connection.
      wire(<ckv.north>, at(560, 985), at(900, 985), <v-up.south>, "-|>")
      edge(<kc.north>, at(560, 760), at(620, 760), <k-concat.south>, "-|>")
      edge(<kr.north>, at(730, 760), at(670, 760), <k-concat.south>, "-|>")
      edge(<k-concat>, <k>, "-|>")

      edge(<q.north>, at(185, 595), <qk.west>, "-|>")
      edge(<k.north>, at(645, 595), <qk.east>, "-|>")
      edge(<qk>, <scores>, "-|>")
      edge(<scores>, <softmax>, "-|>")
      edge(<softmax>, <attention>, "-|>")
      edge(<attention>, <weighted-sum>, "-|>")
      edge(<v.north>, at(900, 325), <weighted-sum.east>, "-|>")
      edge(<weighted-sum>, <head-out>, "-|>")
      edge(<head-out>, <heads-concat>, "-|>")
      edge(<heads-concat>, <heads>, "-|>")
      edge(<heads>, <out-proj>, "-|>")
      edge(<out-proj>, <output>, "-|>")
    },
  )
}

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#mla-diagram()
