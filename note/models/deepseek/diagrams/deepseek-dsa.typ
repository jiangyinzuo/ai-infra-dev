#import "@preview/fletcher:0.5.8": node, edge
#import "@preview/mitex:0.2.7": mi
#import "mla-common.typ": mla-layout, mla-render, mla-restyle, attention-row, attention-input-row, query-latent-width, mla-output-shift
#import "deepseek-absorption.typ": mla-absorption-changes

#set page(width: auto, height: auto, margin: 14pt, fill: white)

#let dsa-diagram(absorbed: false) = context {
  let symbols = (
    attention-core: ("SparseAttention", "o_ti", "q_ti", "k_ji", "v_ji^C", "d_h", "d_h^R", "calS_t", "t", "j", "i"),
    w-index-q-proj: ("W^IQ", "n_h^I", "d_h^I", "d_c'"),
    w-index-k-proj: ("W^IK", "d_h^I", "d"),
    w-index-weight-proj: ("W^IW", "n_h^I", "d"),
    index-reshape: ("reshape", "tilde_q_ti^I", "n_h^I", "d_h^I", "i"),
    index-q-rope: ("RoPE", "P_p^I", "q_ti^I", "tilde_q_ti^I", "d_h^R", "d_h^I", "t", "i"),
    index-q: ("q_ti^I", "n_h^I", "d_h^I", "t", "i"),
    g-index-k-proj: ("gamma^I", "d_h^I"),
    b-index-k-proj: ("beta^I", "d_h^I"),
    index-k-normalized: ("tilde_k_j^I", "d_h^I", "j"),
    index-k-rope: ("RoPE", "P_p^I", "k_j^I", "tilde_k_j^I", "d_h^R", "d_h^I", "j"),
    index-k: ("k_j^I", "d_h^I", "n_h^I", "j"),
    index-w: ("w_ti^I", "n_h^I", "t", "i"),
    index-score: ("I_tj", "w_ti^I", "q_ti^I", "k_j^I", "ReLU", "n_h^I", "T", "t", "j", "i"),
    index-scores: ("I_tj", "t", "j"),
    index-topk: ("calS_t", "I_tj", "m_t", "k", "t", "j"),
    index-selected: ("calS_t", "m_t", "k", "j", "t"),
  )
  let layout = mla-layout(scope: "dsa", symbols: symbols,
    title: [DSA · 选中位置上的 MLA 等价展开],
    note: [
      *Lightning Indexer · 位置选择* \
      #mi(`\mathcal S_t=\operatorname{TopK}_{1\le j\le t}(I_{t,j},m_t),\quad m_t=\min(t,2048)`) \
      部分 RoPE 仅旋转前 64 维，后 64 维直通；Key 为全部 64 个 indexer 头共享。 \
      本图展开未量化评分；官方额外 Hadamard / FP8 路径见正文，量化后不保证精确等价。
    ])
  let p = layout.paint
  let (at, tensor, operator, parameter, projection, matmul-rmsnorm, wire) = (p.at, p.tensor, p.operator, p.parameter, p.projection, p.matmul-rmsnorm, p.wire)
  // MiTeX's explicit math colors need their alpha before conversion; metadata handles the frames.
  let colored(tex, hex) = raw("\\color[HTML]{" + hex + if absorbed { "66" } else { "" } + "}{" + tex.text + "}")
  let replace = (
    input-q: tensor(at(195, 1200), "input-q", colored(`\mathbf h_t\in\mathbb R^d`, "ff0000"), colored(`d=7168`, "ff0000")),
    input-kv: tensor(at(645, 1200), "input-kv", colored(`\mathbf h_j\in\mathbb R^d`, "ff0000"), colored(`d=7168,\quad j\le t`, "ff0000")),
    cq: tensor(at(195, 1025), "cq", `\mathrm{Latent}\ `.text + colored(`\mathbf c_t^Q\in\mathbb R^{d_c^{\prime}}`, "ff0000").text,
      colored(`d_c^{\prime}=1536`, "ff0000"), width: query-latent-width),
    q-down: matmul-rmsnorm(195, 1115, "q-down", colored(`W^{DQ}`, "ff0000"), 1536, 7168, colored(`d_c^{\prime}\times d`, "ff0000"),
      colored(`\boldsymbol{\gamma}^Q`, "ff0000"), colored(`d_c^{\prime}=1536`, "ff0000"), title: [MatMulRMSNorm（压缩）]),
    attention-core: operator(at(415, attention-row), "attention-core", [Sparse Attention],
      formula: `\mathbf o_{t,i}=\sum_{j\in\mathcal S_t}\operatorname{Softmax}_{j\in\mathcal S_t}\!\left(\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}\right)\mathbf v_{j,i}^C\in\mathbb R^{d_h}=\mathbb R^{128}`, width: 340pt),
  )
  // Left to right: dynamic head weights, Query, Key.
  let columns = (weights: 1075, query: 1280, key: 1590)
  // Both operators' activation inputs use attention-input-row; derive earlier Indexer stages from it.
  let rows = (projection: attention-input-row + 250, latent: attention-input-row + 160, transform: attention-input-row + 80, ready: attention-input-row)
  let extra = {
    node(at(columns.query - 70, attention-row - 310), align(left)[
      *Lightning Indexer · 自下向上* \
      #mi(`n_h^I=64,\quad d_h^I=128`)；左侧 MLA 为 #mi(`n_h=128`) 个头。 \
      红色公式：共享参数 / 激活；绿色公式：Indexer 自有可学习参数。 \
      所有框的形状、填色、边框与 MLA 使用同一套样式；颜色只标记公式。 \
      部分 RoPE 仅旋转前 64 维，后 64 维直通；Key 为全部 64 个 indexer 头共享。 \
      本图展开未量化评分；官方额外 Hadamard / FP8 路径见正文，量化后不保证精确等价。
    ], width: 980pt, stroke: none, inset: 4pt, snap: false)

    // CSS named green is #008000; MiTeX's named green is the brighter #00ff00.
    parameter(at(columns.query - 90, rows.projection), "w-index-q-proj", colored(`W^{IQ}`, "008000"),
      `\begin{gathered}n_h^I d_h^I\times d_c^{\prime}\\8192\times1536\end{gathered}`, width: 90pt)
    operator(at(columns.query, rows.projection), "index-q-proj", [MatMul], width: 68pt, keys: ("MatMul",))
    edge(<w-index-q-proj.east>, <index-q-proj.west>, "--|>", stroke: rgb("79517d") + 0.8pt)
    operator(at(columns.query, rows.latent), "index-reshape", [按头 Reshape],
      formula: `\tilde{\mathbf{q}}_{t,i}^I=[\operatorname{reshape}_{64\times128}(\cdot)]_{i,:}`, width: 260pt)
    operator(at(columns.query, rows.transform), "index-q-rope", [部分 RoPE · 前 64 维],
      formula: `\mathbf{q}_{t,i}^I=P_t^I\tilde{\mathbf{q}}_{t,i}^I`, width: 220pt)
    tensor(at(columns.query, rows.ready), "index-q", `\mathbf{q}_{t,i}^I\in\mathbb{R}^{128}`,
      `d_h^I=128`, width: 210pt, head-count: `n_h^I=64`)

    parameter(at(columns.key - 175, rows.projection - 35), "w-index-k-proj", colored(`W^{IK}`, "008000"), `128\times7168`, width: 130pt)
    parameter(at(columns.key - 175, rows.projection), "g-index-k-proj", colored(`\boldsymbol\gamma^I`, "008000"),
      `d_h^I=128`, width: 130pt)
    parameter(at(columns.key - 175, rows.projection + 35), "b-index-k-proj", colored(`\boldsymbol\beta^I`, "008000"),
      `d_h^I=128`, width: 130pt)
    operator(at(columns.key, rows.projection), "index-k-proj", [MatMulLayerNorm], width: 135pt, keys: ("MatMulLayerNorm",))
    edge(<w-index-k-proj.east>, at(columns.key - 67.5, rows.projection - 35), <index-k-proj.north-west>, "--|>", stroke: rgb("79517d") + 0.8pt)
    edge(<g-index-k-proj.east>, <index-k-proj.west>, "--|>", stroke: rgb("79517d") + 0.8pt)
    edge(<b-index-k-proj.east>, at(columns.key - 67.5, rows.projection + 35), <index-k-proj.south-west>, "--|>", stroke: rgb("79517d") + 0.8pt)
    tensor(at(columns.key, rows.latent), "index-k-normalized", `\tilde{\mathbf{k}}_j^I`, `\in\mathbb{R}^{128}`, width: 210pt)
    operator(at(columns.key, rows.transform), "index-k-rope", [部分 RoPE · 前 64 维],
      formula: `\mathbf{k}_j^I=P_j^I\tilde{\mathbf{k}}_j^I`, width: 220pt)
    tensor(at(columns.key, rows.ready), "index-k", `\mathbf{k}_j^I\in\mathbb{R}^{128}`,
      `\text{全部 }64\text{ 个 indexer 头共享，类似MQA但没有Value}`, cache: true, width: 225pt)

    parameter(at(columns.weights - 100, rows.projection - 45), "w-index-weight-proj", colored(`W^{IW}`, "008000"),
      `n_h^I\times d=64\times7168`, width: 130pt)
    operator(at(columns.weights, rows.projection), "index-weight-proj", [MatMul / #mi(`\sqrt{n_h^I d_h^I}`)],
      width: 130pt, keys: ("MatMul", "n_h^I", "d_h^I", "IndexerWeightScaleFold"))
    edge(<w-index-weight-proj.south>, at(columns.weights - 100, rows.projection),
      <index-weight-proj.west>, "--|>", stroke: rgb("79517d") + 0.8pt)
    tensor(at(columns.weights, rows.ready), "index-w", `w_{t,i}^I\in\mathbb R`, `i=1,\ldots,n_h^I`, width: 136pt, head-count: `n_h^I=64`)

    operator(at(columns.query, attention-row), "index-score", [Indexer 评分 · 对全部因果历史候选],
      formula: `I_{t,j}=\sum_{i=1}^{64}w_{t,i}^I\operatorname{ReLU}\!\left((\mathbf{q}_{t,i}^I)^T\mathbf{k}_j^I\right)\in\mathbb R`, width: 600pt)
    tensor(at(columns.query, attention-row - 80), "index-scores", `[I_{t,j}]_{j=1}^{t}`, `\in\mathbb{R}^{t}`, width: 220pt)
    operator(at(columns.query, attention-row - 160), "index-topk", [因果 Top-k · 上限 2048],
      formula: `\mathcal{S}_t=\operatorname{TopK}_{1\le j\le t}(I_{t,j},m_t)`, width: 300pt)
    tensor(at(columns.query, attention-row - 240), "index-selected", `\mathcal{S}_t\subseteq\{1,\ldots,t\}`,
      `|\mathcal{S}_t|=m_t=\min(t,2048)`, width: 230pt)

    // Shared sources leave only from their top or sides, never from their bottom edge.
    // Reuse the upward latent-to-RoPE stem; the shared Indexer input never leaves downward.
    wire(at(280, 970), at(columns.query, 970), <index-q-proj.south>, "-|>")
    node(at(280, 970), [], shape: circle, width: 3pt, height: 3pt, inset: 0pt,
      fill: rgb("485466"), stroke: none, snap: false)
    wire(<input-kv.east>, at(columns.key, 1200), <index-k-proj.south>, "-|>")
    wire(at(195, 1145), at(columns.weights, 1145), <index-weight-proj.south>, "-|>")
    node(at(195, 1145), [], shape: circle, width: 3pt, height: 3pt, inset: 0pt,
      fill: rgb("485466"), stroke: none, snap: false)
    edge(<index-q-proj>, <index-reshape>, "-|>")
    edge(<index-reshape>, <index-q-rope>, "-|>")
    edge(<index-q-rope>, <index-q>, "-|>")
    edge(<index-k-proj>, <index-k-normalized>, "-|>")
    edge(<index-k-normalized>, <index-k-rope>, "-|>")
    edge(<index-k-rope>, <index-k>, "-|>")
    edge(<index-weight-proj>, <index-w>, "-|>")
    wire(<index-w.north>, at(columns.weights, attention-row + 14.5), "-|>")
    edge(<index-q.north>, <index-score.south>, "-|>")
    wire(<index-k.north>, at(columns.key, attention-row + 45), at(columns.query + 200, attention-row + 45), at(columns.query + 200, attention-row + 14.5), "-|>")
    edge(<index-score>, <index-scores>, "-|>")
    edge(<index-scores>, <index-topk>, "-|>")
    edge(<index-topk>, <index-selected>, "-|>")

    // Selected positions are an ordinary fourth input, not a separate K/V control bus.
    wire(<index-selected.west>, at(585, attention-row - 240), <attention-core.north-east>, "-|>")
  }
  if not absorbed {
    mla-render(layout, replace: replace, extra: extra, bounds-y: (mla-output-shift - 14.5, 1223))
  } else {
    let change = mla-absorption-changes(layout, paths: ("qk", "value"), sparse: true)
    let caption = (layout.nodes.note + []).children.first().value
    caption.label = align(left)[
      *DSA · MLA 矩阵吸收* \
      #mi(`W_i^{QK}=(W_i^{UQ})^TW_i^{UK}`) \
      #mi(`W^{O,UV}=W^O\operatorname{diag}(W_i^{UV})`) \
      MLA 的 QK 内容投影与 Value / 输出投影在潜空间中合并。 \
      选中位置仍由 Indexer 输入 Sparse Attention。 \
      RoPE 不变，Softmax 缩放仍为 #mi(`1/\sqrt{192}`)。
    ]
    change.replace.insert("note", metadata(caption))
    change.focus += ("note",)
    // Fixed Indexer scaling is already fused in the original diagram; only MLA changes here.
    let transformed = mla-restyle(extra, faded: true)
    mla-render(layout, ids: change.ids, replace: replace + change.replace, routes: change.routes,
      extra: transformed, focus: change.focus, bounds-y: (mla-output-shift - 14.5, 1223))
}
}

#dsa-diagram()
