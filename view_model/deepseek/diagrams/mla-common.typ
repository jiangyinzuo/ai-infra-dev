// Shared fixed-coordinate MLA nodes, edges and Fletcher styling.
// Subdiagrams select nodes rather than laying them out again. Coordinates are in pt.
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/mitex:0.2.7": mi

#let mla-dimensions = (hidden: 7168, heads: 128, content: 128, rope: 64, query-latent: 1536, kv-latent: 512)
// Attention and Lightning Indexer scoring are peers on the same row.
// DSA imports this constant: change it here, never move the two operators independently.
#let attention-row = 565
// Q/K/V for Attention and w/q/k for Indexer scoring must share this activation row.
#let attention-input-row = 665
// Keep the upper output chain together, 90pt above the Attention operator.
#let mla-output-shift = attention-row - 350
// Span both Query projections so their inputs can rise straight from the latent's top edge.
#let query-latent-width = 220pt
#let mla-node-symbols = (
  output: ("u_t", "d", "t"), heads: ("o_ti", "n_h", "d_h", "t"),
  head-out: ("o_ti", "d_h", "t", "i"),
  attention-core: ("Attention", "q_ti", "k_ji", "v_ji^C", "d_h", "d_h^R", "t", "j", "i"),
  q: ("q_ti", "q_ti^C", "q_ti^R", "d_h", "d_h^R", "t", "i"), k: ("k_ji", "k_ji^C", "k_j^R", "d_h", "d_h^R", "j", "i"),
  qc: ("q_ti^C", "d_h", "t", "i"), qr: ("q_ti^R", "d_h^R", "t", "i"), kc: ("k_ji^C", "d_h", "j", "i"),
  v: ("v_ji^C", "d_h", "j", "i"), kr: ("k_j^R", "d_h^R", "j"),
  cq: ("c_t^Q", "d_c'", "t"), ckv: ("c_j^KV", "d_c", "j"),
  input-q: ("h_t", "d", "t"), input-kv: ("h_j", "d", "j", "t"),
  heads-concat: ("concat", "n_h"), q-concat: ("concat",), k-concat: ("concat",),
  q-rope: ("RoPE", "t"), k-rope: ("RoPE", "j"),
  w-out-proj: ("W^O", "d", "n_h", "d_h"), w-q-up: ("W_i^UQ", "d_h", "d_c'", "i"),
  w-qr-proj: ("W_i^QR", "d_h^R", "d_c'", "i"), w-k-up: ("W_i^UK", "d_h", "d_c", "i"),
  w-v-up: ("W_i^UV", "d_h", "d_c", "i"), w-q-down: ("W^DQ", "d_c'", "d"),
  w-kv-down: ("W^DKV", "d_c", "d"), w-kr-proj: ("W^KR", "d_h^R", "d"),
  g-q-down: ("gamma^Q", "d_c'"), g-kv-down: ("gamma^KV", "d_c"),
)

#let mla-paint(scope: "mla", symbols: (:)) = {
  let ink = rgb("485466")
  let parameter-ink = rgb("79517d")
  let keys-by-node = mla-node-symbols + symbols
  let at(x, y) = (x * 1pt, -y * 1pt)
  let preview-region(pos, id, width, height, keys: none) = {
    let keys = if keys == none { keys-by-node.at(id, default: ()) } else { keys }
    if keys.len() > 0 {
      node(pos, link("#diagram-symbols/" + scope + "/" + id + "/" + keys.join("|"), box(width: width, height: height)),
        name: label("preview-" + id), shape: rect, width: width, height: height,
        inset: 0pt, fill: none, stroke: none, snap: false, layer: 10)
    }
  }
  // Match the rendered font when deciding whether a label and its detail fit on one line.
  let compact-label(first, second, width, inset) = {
    let styled(body) = text(font: ("Noto Sans CJK SC", "New Computer Modern"), size: 10pt, body)
    let inline = if second == none { first } else { [#first#h(5pt)#second] }
    let body = if second == none or measure(styled(inline)).width <= width - 2 * inset {
      inline
    } else { [#first \ #second] }
    let body = [#set par(leading: 3pt)
      #styled(body)]
    (body: body, height: calc.max(29pt, measure(box(width: width - 2 * inset, body)).height + 2 * inset))
  }
  let tensor(pos, id, symbol, dimension, cache: false, width: 136pt, keys: none) = {
    let caption = compact-label(mi(symbol), text(9pt, mi(dimension)), width, 5pt)
    node(pos, align(center, caption.body),
      name: label(id), shape: rect, width: width, height: caption.height, corner-radius: 7pt, inset: 5pt,
      fill: if cache { rgb("e4f2ff") } else { rgb("f1f7fc") },
      stroke: (paint: if cache { rgb("1665ac") } else { rgb("245e91") }, thickness: if cache { 1.3pt } else { 0.85pt }),
      extrude: if cache { (0, 2.5) } else { (0,) })
    preview-region(pos, id, width, caption.height, keys: keys)
  }
  let operator(pos, id, title, formula: none, width: 90pt, keys: none) = {
    let caption = compact-label(text(weight: "bold", title), if formula == none { none } else { text(9pt, mi(formula)) }, width, 5pt)
    let height = caption.height
    node(pos, align(center, caption.body),
      name: label(id), shape: rect, width: width, height: height, inset: 5pt,
      fill: rgb("f3f3f4"), stroke: 0.85pt + ink)
    preview-region(pos, id, width, height, keys: keys)
  }
  let parameter(pos, id, symbol, dimension, width: 100pt, color: parameter-ink, keys: none) = {
    let caption = compact-label(mi(symbol), text(8.5pt, mi(dimension)), width, 4pt)
    node(pos, align(center, caption.body),
      name: label(id), shape: rect, width: width, height: caption.height, inset: 4pt,
      fill: rgb("faf4fa"), stroke: (paint: color, thickness: 0.85pt, dash: "dashed"))
    preview-region(pos, id, width, caption.height, keys: keys)
  }
  let projection(x, y, id, weight, rows, cols, shape, weight-offset: (-108, 0), weight-width: 100pt, color: parameter-ink) = {
    parameter(at(x + weight-offset.at(0), y + weight-offset.at(1)), "w-" + id, weight,
      shape.text + "=" + str(rows) + "\\times" + str(cols), width: weight-width, color: color)
    operator(at(x, y), id, [MatMul], width: 68pt, keys: ("MatMul",))
    if weight-offset.at(0) != 0 and weight-offset.at(1) != 0 {
      // Vertically offset weights still enter through an orthogonal side connection.
      let left = weight-offset.at(0) < 0
      let inlet-x = x + if left { -34 } else { 34 }
      edge(label("w-" + id + if left { ".east" } else { ".west" }),
        at(inlet-x, y + weight-offset.at(1)), label(id + if left { ".west" } else { ".east" }),
        "--|>", stroke: color + 0.8pt)
    } else {
      edge(label("w-" + id), label(id), "--|>", stroke: color + 0.8pt)
    }
  }
  // Three inputs: hidden state, down-projection weight, and RMSNorm scale. One normalized output.
  let matmul-rmsnorm(x, y, id, weight, rows, cols, shape, gamma, gamma-dimension, weight-offset: (-150, 0), gamma-offset: (-150, -55)) = {
    parameter(at(x + weight-offset.at(0), y + weight-offset.at(1)), "w-" + id, weight,
      shape.text + "=" + str(rows) + "\\times" + str(cols))
    parameter(at(x + gamma-offset.at(0), y + gamma-offset.at(1)), "g-" + id, gamma, gamma-dimension)
    operator(at(x, y), id, [MatMulRMSNorm], width: 110pt, keys: ("MatMulRMSNorm",))
    edge(label("w-" + id), label(id), "--|>", stroke: parameter-ink + 0.8pt)
    edge(label("g-" + id + ".east"), at(x - 25, y + gamma-offset.at(1)), at(x - 25, y - 14.5), "--|>", stroke: parameter-ink + 0.8pt)
  }
  let wire(..args) = edge(..args, crossing: true, crossing-thickness: 6)
  (at: at, tensor: tensor, operator: operator, parameter: parameter, projection: projection,
    matmul-rmsnorm: matmul-rmsnorm, wire: wire, preview-region: preview-region)
}

#let mla-figure(body) = {
  set text(font: ("Noto Sans CJK SC", "New Computer Modern"), size: 10pt, fill: rgb("202735"))
  set par(leading: 4pt)
  diagram(node-shape: rect, edge-stroke: 0.9pt + rgb("485466"), edge-corner-radius: 5pt,
    mark-scale: 70%, crossing-fill: white, body)
}

#let mla-layout(scope: "mla", symbols: (:), title: [DeepSeek MLA · 未吸收路径], note: none) = {
  let paint = mla-paint(scope: scope, symbols: symbols)
  let (at, tensor, operator, projection, matmul-rmsnorm, wire) = (paint.at, paint.tensor, paint.operator, paint.projection, paint.matmul-rmsnorm, paint.wire)
  let d = mla-dimensions
  let full-head = d.content + d.rope
  let all-values = d.heads * d.content
  // Keep legend and captions beside the output chain instead of extending the canvas upward.
  let legend-shift = attention-row - 405
  let nodes = (
    legend: {
      tensor(at(55, 70 + legend-shift), "legend-tensor", `\mathbf{x}`, `\mathrm{Activation}`, width: 128pt)
      node(at(55, 132 + legend-shift), align(center)[#mi(`W`)#h(6pt)可学习权重], shape: rect,
        width: 128pt, height: 29pt, fill: rgb("faf4fa"), stroke: (paint: rgb("79517d"), thickness: 0.85pt, dash: "dashed"))
      operator(at(55, 192 + legend-shift), "legend-operator", [算子 Operator], width: 128pt)
      tensor(at(55, 255 + legend-shift), "legend-cache", `\mathrm{KV\ Cache}`, `\mathrm{stored\ activation}`, cache: true, width: 128pt)
    },
    note: node(at(750, attention-row - 125), align(left)[
      *#title* \
      数据自下向上流动；紫色虚线为权重输入。 \
      #mi(`i=1,\ldots,n_h,\quad j=1,\ldots,t`) \
      #mi(`n_h=128,\ d_h=128,\ d_h^R=64`) \
      #if note == none [
        #mi(`W_i^{UQ},W_i^{QR},W_i^{UK},W_i^{UV}`) 为按头分块。 \
        MatMulRMSNorm 合并投影与归一化（报告公式省略后者）。 \
        输入已过 attention 前置 RMSNorm；不画残差或 FFN。
      ] else { note }
    ], stroke: none, width: 290pt, inset: 4pt, snap: false),
    output: tensor(at(415, mla-output-shift), "output", `\mathbf{u}_t\in\mathbb{R}^{d}`.text, `d=`.text + str(d.hidden)),
    out-proj: projection(415, 65 + mla-output-shift, "out-proj", `W^O`, d.hidden, all-values, `d\times(n_h d_h)`, weight-offset: (-155, 0), weight-width: 155pt),
    heads: tensor(at(415, 130 + mla-output-shift), "heads", `[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_h}]`.text, `\in\mathbb{R}^{n_h d_h}=\mathbb{R}^{`.text + str(all-values) + `}`.text, width: 200pt),
    heads-concat: operator(at(415, 195 + mla-output-shift), "heads-concat", [Concat · 全部头], width: 136pt),
    head-out: tensor(at(415, 260 + mla-output-shift), "head-out", `\mathbf{o}_{t,i}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content)),
    attention-core: operator(at(415, attention-row), "attention-core", [Attention],
      formula: `\sum_{j\le t}\operatorname{Softmax}_{j\le t}\!\left(\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}\right)\mathbf v_{j,i}^C`, width: 340pt),
    q: tensor(at(185, attention-input-row), "q", `\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}]`.text, `\in\mathbb{R}^{d_h+d_h^R}=\mathbb{R}^{`.text + str(full-head) + `}`.text, width: 230pt),
    k: tensor(at(645, attention-input-row), "k", `\mathbf{k}_{j,i}=[\mathbf{k}_{j,i}^{C};\mathbf{k}_{j}^{R}]`.text, `\in\mathbb{R}^{d_h+d_h^R}=\mathbb{R}^{`.text + str(full-head) + `}`.text, width: 230pt),
    q-concat: operator(at(185, 730), "q-concat", [Concat], width: 90pt),
    k-concat: operator(at(645, 730), "k-concat", [Concat], width: 90pt),
    qc: tensor(at(95, 800), "qc", `\mathbf{q}_{t,i}^{C}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content)),
    qr: tensor(at(280, 800), "qr", `\mathbf{q}_{t,i}^{R}\in\mathbb{R}^{d_h^R}`.text, `d_h^R=`.text + str(d.rope)),
    kc: tensor(at(560, 800), "kc", `\mathbf{k}_{j,i}^{C}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content)),
    v: tensor(at(900, attention-input-row), "v", `\mathbf{v}_{j,i}^{C}\in\mathbb{R}^{d_h}`.text, `d_h=`.text + str(d.content)),
    q-rope: operator(at(280, 865), "q-rope", [RoPE], formula: `\mathrm{position}\ t`, width: 90pt),
    q-up: projection(95, 935, "q-up", `W_i^{UQ}`, d.content, d.query-latent, `d_h\times d_c^{\prime}`),
    qr-proj: projection(280, 935, "qr-proj", `W_i^{QR}`, d.rope, d.query-latent, `d_h^R\times d_c^{\prime}`, weight-offset: (-95, 0), weight-width: 90pt),
    k-up: projection(560, 935, "k-up", `W_i^{UK}`, d.content, d.kv-latent, `d_h\times d_c`),
    v-up: projection(900, 935, "v-up", `W_i^{UV}`, d.content, d.kv-latent, `d_h\times d_c`, weight-offset: (-108, -45)),
    kr: tensor(at(730, 935), "kr", `\mathbf{k}_{j}^{R}\in\mathbb{R}^{d_h^R}`.text, `d_h^R=`.text + str(d.rope) + `\quad\text{（所有头共享）}`.text, cache: true, width: 150pt),
    cq: tensor(at(195, 1025), "cq", `\mathrm{Latent}\ \mathbf{c}_t^{Q}\in\mathbb{R}^{d_c^{\prime}}`.text, `d_c^{\prime}=`.text + str(d.query-latent), width: query-latent-width),
    ckv: tensor(at(560, 1025), "ckv", `\mathrm{Latent}\ \mathbf{c}_j^{KV}\in\mathbb{R}^{d_c}`.text, `d_c=`.text + str(d.kv-latent), cache: true),
    k-rope: operator(at(730, 1025), "k-rope", [RoPE], formula: `\mathrm{position}\ j`, width: 90pt),
    q-down: matmul-rmsnorm(195, 1115, "q-down", `W^{DQ}`, d.query-latent, d.hidden, `d_c^{\prime}\times d`,
      `\boldsymbol{\gamma}^{Q}`, `d_c^{\prime}=1536`),
    kv-down: matmul-rmsnorm(560, 1115, "kv-down", `W^{DKV}`, d.kv-latent, d.hidden, `d_c\times d`,
      `\boldsymbol{\gamma}^{KV}`, `d_c=512`),
    kr-proj: projection(730, 1115, "kr-proj", `W^{KR}`, d.rope, d.hidden, `d_h^R\times d`, weight-offset: (108, 0)),
    input-q: tensor(at(195, 1200), "input-q", `\mathbf{h}_t\in\mathbb{R}^{d}`.text, `d=`.text + str(d.hidden)),
    input-kv: tensor(at(645, 1200), "input-kv", `\mathbf{h}_j\in\mathbb{R}^{d}`.text, `d=`.text + str(d.hidden) + `,\quad j\le t`.text),
  )
  let route(from, to, body) = (from: from, to: to, body: body)
  let routes = (
    route("input-q", "q-down", edge(<input-q>, <q-down>, "-|>")),
    route("q-down", "cq", edge(<q-down>, <cq>, "-|>")),
    route("cq", "q-up", edge(at(95, 1010.5), <q-up.south>, "-|>")),
    route("cq", "qr-proj", edge(at(280, 1010.5), <qr-proj.south>, "-|>")),
    route("q-up", "qc", edge(<q-up>, <qc>, "-|>")),
    route("qr-proj", "q-rope", edge(<qr-proj>, <q-rope>, "-|>")),
    route("q-rope", "qr", edge(<q-rope>, <qr>, "-|>")),
    route("qc", "q-concat", edge(<qc.north>, at(95, 760), at(160, 760), at(160, 744.5), "-|>")),
    route("qr", "q-concat", edge(<qr.north>, at(280, 760), at(210, 760), at(210, 744.5), "-|>")),
    route("q-concat", "q", edge(<q-concat>, <q>, "-|>")),
    route("input-kv", "kv-down", edge(<input-kv.north>, at(645, 1160), at(560, 1160), <kv-down.south>, "-|>")),
    route("input-kv", "kr-proj", edge(<input-kv.north>, at(645, 1160), at(730, 1160), <kr-proj.south>, "-|>")),
    route("kv-down", "ckv", edge(<kv-down>, <ckv>, "-|>")),
    route("ckv", "k-up", edge(<ckv>, <k-up>, "-|>")),
    route("k-up", "kc", edge(<k-up>, <kc>, "-|>")),
    route("v-up", "v", edge(<v-up>, <v>, "-|>")),
    route("kr-proj", "k-rope", edge(<kr-proj>, <k-rope>, "-|>")),
    route("k-rope", "kr", edge(<k-rope>, <kr>, "-|>")),
    route("ckv", "v-up", wire(<ckv.north>, at(560, 985), at(900, 985), <v-up.south>, "-|>")),
    route("kc", "k-concat", edge(<kc.north>, at(560, 760), at(620, 760), at(620, 744.5), "-|>")),
    route("kr", "k-concat", edge(<kr.north>, at(730, 760), at(670, 760), at(670, 744.5), "-|>")),
    route("k-concat", "k", edge(<k-concat>, <k>, "-|>")),
    route("q", "attention-core", edge(<q.north>, at(185, attention-row), <attention-core.west>, "-|>")),
    route("k", "attention-core", edge(<k.north>, at(645, attention-input-row - 40), at(415, attention-input-row - 40), <attention-core.south>, "-|>")),
    route("v", "attention-core", edge(<v.north>, at(900, attention-row), <attention-core.east>, "-|>")),
    route("attention-core", "head-out", edge(<attention-core>, <head-out>, "-|>")),
    route("head-out", "heads-concat", edge(<head-out>, <heads-concat>, "-|>")),
    route("heads-concat", "heads", edge(<heads-concat>, <heads>, "-|>")),
    route("heads", "out-proj", edge(<heads>, <out-proj>, "-|>")),
    route("out-proj", "output", edge(<out-proj>, <output>, "-|>")),
  )
  (paint: paint, nodes: nodes, routes: routes)
}

// Restyle Fletcher metadata, not a second set of drawn nodes. Geometry and hit regions stay shared.
#let mla-restyle(body, reference: none, faded: false) = {
  let source = if reference == none { () } else { (reference + []).children }
  let fade-stroke(value) = {
    if value == none { return none }
    let spec = if value == auto { (paint: rgb("485466"), thickness: 0.9pt) }
      else if type(value) == dictionary { value }
      else if type(value) == stroke { (paint: value.paint, thickness: value.thickness, cap: value.cap,
        join: value.join, dash: value.dash, miter-limit: value.miter-limit) }
      else { (paint: value, thickness: 0.9pt) }
    spec.paint = spec.at("paint", default: rgb("485466")).transparentize(60%)
    stroke(..spec)
  }
  for item in (body + []).children {
    assert(item.func() == metadata)
    let record = item.value
    if record.class == "node" and reference != none {
      let original = source.find(item => item.value.class == "node" and item.value.name == record.name)
      assert(original != none, message: "Replacement must reuse an existing MLA node name")
      for key in ("pos", "size", "radius", "shape", "corner-radius", "inset", "outset", "extrude", "snap", "layer") {
        record.insert(key, original.value.at(key))
      }
      if auto in record.size {
        let inset = original.value.inset
        let inner = original.value.size.map(size => if size == auto { auto } else { size - 2 * inset })
        let measure-original() = {
          let label = [#set text(font: ("Noto Sans CJK SC", "New Computer Modern"), size: 10pt)
            #set par(leading: 4pt)
            #original.value.label]
          measure(box(width: inner.at(0), height: inner.at(1), label))
        }
        let measured = measure-original()
        record.size = (if record.size.at(0) == auto { measured.width + 2 * inset } else { record.size.at(0) },
          if record.size.at(1) == auto { measured.height + 2 * inset } else { record.size.at(1) })
      }
      if record.label.func() == link {
        record.label = link(record.label.dest, box(width: record.size.at(0), height: record.size.at(1)))
      }
    }
    if faded {
      if record.class == "node" {
        if type(record.fill) == color { record.fill = record.fill.transparentize(60%) }
        record.stroke = fade-stroke(record.stroke)
        let label = record.label
        record.label = [#show text: it => context {
          let paint = text.fill
          // Explicit MiTeX math colors may already carry the comparison alpha.
          text(fill: if paint.components().last() <= 40% { paint } else { paint.transparentize(60%) }, it)
        }
          #label]
      } else if not record.is-crossing-background {
        record.stroke = fade-stroke(record.stroke)
      }
    }
    metadata(record)
  }
}

// Horizontal bounds keep extracted paths aligned with the original, even when most nodes are absent.
// Existing nodes occupy x=-63..1058; adding a right-hand branch expands the SVG without moving them.
#let mla-render(layout, ids: none, replace: (:), extra: [], routes: none, bounds-y: none, focus: none) = {
  let ids = if ids == none { layout.nodes.keys() } else { ids }
  mla-figure({
    for id in ids {
      let original = layout.nodes.at(id)
      let body = if id in replace { mla-restyle(replace.at(id), reference: original) } else { original }
      if focus != none and id not in focus { mla-restyle(body, faded: true) } else { body }
    }
    for route in if routes == none { layout.routes } else { routes } {
      if route.from in ids and route.to in ids {
        if focus != none and (route.from not in focus or route.to not in focus) {
          mla-restyle(route.body, faded: true)
        } else { route.body }
      }
    }
    if bounds-y != none {
      let at = layout.paint.at
      for (x-index, x) in (-63, 1058).enumerate() {
        for (y-index, y) in bounds-y.enumerate() {
          // Fletcher requires positive size; inset each invisible 1pt square to keep exact bounds.
          node(at(x + if x-index == 0 { 0.5 } else { -0.5 }, y + if y-index == 0 { 0.5 } else { -0.5 }),
            [], width: 1pt, height: 1pt, inset: 0pt, fill: none, stroke: none, snap: false)
        }
      }
    }
    extra
  })
}
