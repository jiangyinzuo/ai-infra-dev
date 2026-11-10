#import "mla-common.typ": mla-layout, mla-paint, mla-render, mla-output-shift, attention-row, attention-input-row
#import "@preview/fletcher:0.5.8": edge
#import "@preview/mitex:0.2.7": mi

// Shared cutover for the standalone MLA comparisons and the combined DSA absorption diagram.
// The caller's original nodes supply geometry; only labels, weights and eliminated paths change.
#let mla-absorption-changes(layout, paths: ("qk",), sparse: false) = {
  let qk = "qk" in paths
  let value = "value" in paths
  let symbols = (:)
  if qk {
    symbols += (
      w-q-up: ("W_i^QK", "T", "d_c", "d_c'", "i"),
      qc: ("hat_q_ti^C", "d_c", "t", "i"),
      q: ("hat_q_ti^C", "q_ti^R", "d_c", "d_h^R", "t", "i"),
      k: ("c_j^KV", "k_j^R", "d_c", "d_h^R", "j"),
    )
  }
  if value {
    symbols += (
      v: ("c_j^KV", "d_c", "j"),
      head-out: ("tilde_o_ti", "d_c", "t", "i"),
      heads: ("tilde_o_ti", "n_h", "d_c", "t"),
      w-out-proj: ("W^O,UV", "d", "n_h", "d_c"),
    )
  }
  symbols.insert("attention-core", if sparse { ("SparseAttentionAbsorbed", "calS_t", "tilde_o_ti", "d_c") }
    else if qk { ("AttentionQKAbsorbed", "hat_q_ti^C", "q_ti^R", "c_j^KV", "k_j^R", "v_ji^C") }
    else { ("AttentionValueAbsorbed", "q_ti", "k_ji", "c_j^KV", "tilde_o_ti") })
  let p = mla-paint(scope: if sparse { "dsa" } else { "mla" }, symbols: symbols)
  let (at, tensor, operator, projection, wire) = (p.at, p.tensor, p.operator, p.projection, p.wire)
  let ids = layout.nodes.keys()
  let replace = (:)
  let routes = layout.routes
  let focus = ("attention-core",)
  // Focus both endpoints of every rerouted or absorbed path so its connections remain opaque.

  if qk {
    // Eliminate W_i^UK and the historical content Key; all heads read the shared latent.
    ids = ids.filter(id => id not in ("k-up", "kc"))
    focus += ("q-up", "qc", "k-up", "kc", "q-concat", "k-concat", "q", "k")
    replace.insert("q-up", projection(95, 935, "q-up", `(W_i^{QK})^T`, 512, 1536, `d_c\times d_c^{\prime}`))
    replace.insert("qc", tensor(at(95, 800), "qc", `\widehat{\mathbf q}_{t,i}^C\in\mathbb R^{d_c}`, `d_c=512`, head-count: `n_h=128`))
    replace.insert("q", tensor(at(185, attention-input-row), "q",
      `[\widehat{\mathbf q}_{t,i}^C;\mathbf q_{t,i}^R]`, `\in\mathbb R^{d_c+d_h^R}=\mathbb R^{576}`, width: 230pt, head-count: `n_h=128`))
    replace.insert("k", tensor(at(645, attention-input-row), "k",
      `[\mathbf c_j^{KV};\mathbf k_j^R]`, `\in\mathbb R^{d_c+d_h^R}=\mathbb R^{576}`, width: 230pt))
    routes += ((from: "ckv", to: "k-concat", body:
      edge(<ckv.north>, at(560, 760), at(620, 760), at(620, 744.5), "-|>")),)
  }
  if value {
    // Move Value aggregation into latent space and precombine W^O with the head-wise W_i^UV.
    ids = ids.filter(id => id != "v-up")
    focus += ("v-up", "v", "head-out", "heads-concat", "heads", "out-proj")
    replace.insert("v", tensor(at(900, attention-input-row), "v", `\mathbf c_j^{KV}\in\mathbb R^{d_c}`, `d_c=512`))
    replace.insert("head-out", tensor(at(415, 260 + mla-output-shift), "head-out", `\widetilde{\mathbf o}_{t,i}\in\mathbb R^{d_c}`, `d_c=512`, head-count: `n_h=128`))
    replace.insert("heads", tensor(at(415, 130 + mla-output-shift), "heads",
      `[\widetilde{\mathbf o}_{t,1};\ldots;\widetilde{\mathbf o}_{t,n_h}]`, `\in\mathbb R^{n_h d_c}=\mathbb R^{65536},\quad n_h=128`, width: 200pt))
    replace.insert("out-proj", projection(415, 65 + mla-output-shift, "out-proj", `W^{O,UV}`, 7168, 65536,
      `d\times(n_h d_c)`, weight-offset: (-155, 0), weight-width: 155pt))
    routes += ((from: "ckv", to: "v", body:
      wire(<ckv.north>, at(560, 985), at(900, 985), <v.south>, "-|>")),)
  }
  let candidates = if sparse { `j\in\mathcal S_t`.text } else { `j\le t`.text }
  let dot = if qk { `\widehat{\mathbf q}_{t,i}^T\widehat{\mathbf k}_j`.text } else { `\mathbf q_{t,i}^T\mathbf k_{j,i}`.text }
  let values = if value { `\mathbf c_j^{KV}`.text } else { `\mathbf v_{j,i}^C`.text }
  let output = if not sparse { "" } else if value { `\widetilde{\mathbf o}_{t,i}=`.text } else { `\mathbf o_{t,i}=`.text }
  let dimension = if not sparse { "" } else if value { `\in\mathbb R^{d_c}=\mathbb R^{512}`.text } else { `\in\mathbb R^{d_h}=\mathbb R^{128}`.text }
  replace.insert("attention-core", operator(at(415, attention-row), "attention-core", if sparse { [Sparse Attention] } else { [Attention] },
    formula: output + "\\sum_{" + candidates + "}\\operatorname{Softmax}_{" + candidates + "}\\!\\left(\\frac{" + dot + "}{\\sqrt{d_h+d_h^R}}\\right)" + values + dimension,
    width: 340pt))
  (ids: ids, replace: replace, routes: routes, focus: focus)
}

// Full MLA comparisons: unchanged modules and connections are translucent, never repositioned.
#let absorption-diagram(path: "qk", stage: "before") = context {
  assert(path in ("qk", "value"), message: "path must be qk or value")
  assert(stage in ("before", "after"), message: "stage must be before or after")
  let absorbed = stage == "after"
  let layout = mla-layout()
  let change = mla-absorption-changes(layout, paths: (path,))
  let replace = if absorbed { change.replace } else { (:) }
  let caption = (layout.nodes.note + []).children.first().value
  caption.label = align(left)[
    *MLA · #if path == "qk" [QK 内容路径] else [Value / 输出路径] · #if absorbed [吸收后] else [吸收前]* \
    实色：本组对比路径；半透明：其余模块，位置与原图一致。 \
    #if path == "qk" [
      #mi(`W_i^{QK}=(W_i^{UQ})^TW_i^{UK}`) \
      #mi(`\widehat{\mathbf q}_{t,i}^C=(W_i^{QK})^T\mathbf c_t^Q`) \
      RoPE 不变；吸收后内容点积在 512 维潜空间完成。 \
      #mi(`\widehat{\mathbf q}=[\widehat{\mathbf q}^C;\mathbf q^R],\quad \widehat{\mathbf k}=[\mathbf c^{KV};\mathbf k^R]`)
    ] else [
      #mi(`W^{O,UV}=W^O\operatorname{diag}(W_1^{UV},\ldots,W_{n_h}^{UV})`) \
      先在 512 维潜空间聚合，再使用吸收后的输出投影。 \
      Query / Key 与 RoPE 分支保持不变。
    ] \
    Softmax 缩放始终保留 #mi(`\sqrt{d_h+d_h^R}=\sqrt{192}`)。
  ]
  replace.insert("note", metadata(caption))
  mla-render(layout, ids: if absorbed { change.ids } else { layout.nodes.keys() }, replace: replace,
    routes: if absorbed { change.routes } else { layout.routes }, focus: change.focus + ("note",),
    bounds-y: (mla-output-shift - 14.5, 1214.5))
}

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#absorption-diagram(path: sys.inputs.at("path", default: "qk"), stage: sys.inputs.at("stage", default: "before"))
