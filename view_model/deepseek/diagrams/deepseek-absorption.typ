#import "mla-common.typ": mla-layout, mla-render, mla-output-shift
#import "@preview/fletcher:0.5.8": edge

// Extracted paths retain the complete MLA coordinate system and node geometry.
#let absorption-diagram(path: "qk", stage: "before") = context {
  assert(path in ("qk", "value"), message: "path must be qk or value")
  assert(stage in ("before", "after"), message: "stage must be before or after")
  let absorbed = stage == "after"
  let symbols = if path == "qk" {
    (
      scores: ("S_tji^C", "t", "j", "i"),
      qk: if absorbed { ("hat_q_ti^C", "c_j^KV", "T", "t", "j", "i") }
        else { ("q_ti^C", "k_ji^C", "T", "t", "j", "i") },
      qc: if absorbed { ("hat_q_ti^C", "d_c", "t", "i") } else { ("q_ti^C", "d_h", "t", "i") },
      w-q-up: if absorbed { ("W_i^QK", "T", "d_c", "d_c'", "i") } else { ("W_i^UQ", "d_h", "d_c'", "i") },
    )
  } else {
    (
      head-out: if absorbed { ("tilde_o_ti", "d_c", "t", "i") } else { ("o_ti", "d_h", "t", "i") },
      heads: if absorbed { ("tilde_o_ti", "n_h", "d_c", "t") } else { ("o_ti", "n_h", "d_h", "t") },
      weighted-sum: if absorbed { ("A_tji", "c_j^KV", "j", "t", "i") } else { ("A_tji", "v_ji^C", "j", "t", "i") },
      w-out-proj: if absorbed { ("W^O,UV", "d", "n_h", "d_c") } else { ("W^O", "d", "n_h", "d_h") },
    )
  }
  let layout = mla-layout(scope: "mla", symbols: symbols)
  let p = layout.paint
  let at = p.at
  let ids = if path == "qk" {
    if absorbed { ("cq", "ckv", "q-up", "qc", "qk", "scores") }
    else { ("cq", "ckv", "q-up", "k-up", "qc", "kc", "qk", "scores") }
  } else {
    if absorbed { ("output", "out-proj", "heads", "heads-concat", "head-out", "weighted-sum", "attention", "ckv") }
    else { ("output", "out-proj", "heads", "heads-concat", "head-out", "weighted-sum", "attention", "v", "v-up", "ckv") }
  }
  let replace = (:)
  let extra = []
  let routes = layout.routes
  if path == "qk" {
    replace.insert("scores", (p.tensor)(at(415, 525), "scores", `S_{t,j,i}^{C}`, `\in\mathbb{R}\quad\text{（内容分数）}`, width: 190pt))
    replace.insert("qk", (p.operator)(at(415, 595), "qk", [内容点积],
      formula: if absorbed { `(\widehat{\mathbf{q}}_{t,i}^{C})^T\mathbf{c}_{j}^{KV}` }
        else { `(\mathbf{q}_{t,i}^{C})^{T}\mathbf{k}_{j,i}^{C}` }, width: 240pt))
    routes += ((from: "qc", to: "qk", body: edge(<qc.north>, at(95, 595), <qk.west>, "-|>")),)
    if absorbed {
      replace.insert("q-up", {
        (p.parameter)(at(-13, 935), "w-q-up", `(W_i^{QK})^{T}`, `512\times1536`)
        (p.operator)(at(95, 935), "q-up", [MatMul], width: 68pt, keys: ("MatMul",))
        edge(<w-q-up>, <q-up>, "--|>", stroke: rgb("79517d") + 0.8pt)
      })
      replace.insert("qc", (p.tensor)(at(95, 800), "qc", `\widehat{\mathbf{q}}_{t,i}^{C}`, `d_c=512`))
      routes += ((from: "ckv", to: "qk", body: edge(<ckv.north>, at(560, 595), <qk.east>, "-|>")),)
    } else {
      routes += ((from: "kc", to: "qk", body: edge(<kc.north>, at(560, 595), <qk.east>, "-|>")),)
    }
    extra = {
      (p.operator)(at(825, 690), "qk-identity", [QK 权重吸收 · 离线预计算],
        formula: `W_i^{QK}=(W_i^{UQ})^T W_i^{UK}`, width: 380pt,
        keys: ("W_i^QK", "W_i^UQ", "W_i^UK", "T", "i"))
      (p.operator)(at(825, 765), "qk-rope-note", [RoPE 分支保持不变，此处不绘制；内容分数未缩放。],
        width: 380pt, keys: ("RoPE", "S_tji^C"))
      (p.operator)(at(825, 855), "qk-norm-note", [源潜向量已过 RMSNorm；报告公式省略归一化。],
        width: 380pt, keys: ("c_t^Q", "c_j^KV", "RMSNorm"))
      (p.operator)(at(825, 900), "qk-norm-limit", [RMSNorm 依赖激活，不能吸收到固定权重中。],
        width: 380pt, keys: ("RMSNorm",))
      if absorbed {
        (p.operator)(at(825, 970), "absorbed-query", [吸收后的 Query · 512 维],
          formula: `\widehat{\mathbf{q}}_{t,i}^{C}=(W_i^{QK})^T\mathbf{c}_t^Q`,
          width: 380pt, keys: ("hat_q_ti^C", "W_i^QK", "c_t^Q", "T", "d_c"))
      }
    }
  } else {
    if absorbed {
      replace.insert("head-out", (p.tensor)(at(415, 260 + mla-output-shift), "head-out", `\widetilde{\mathbf{o}}_{t,i}`, `d_c=512`))
      replace.insert("heads", (p.tensor)(at(415, 130 + mla-output-shift), "heads",
        `[\widetilde{\mathbf{o}}_{t,1};\ldots;\widetilde{\mathbf{o}}_{t,n_h}]`,
        `\in\mathbb{R}^{n_h d_c}=\mathbb{R}^{65536}`, width: 200pt))
      replace.insert("out-proj", {
        (p.parameter)(at(260, 65 + mla-output-shift), "w-out-proj", `W^{O,UV}`, `7168\times65536`, width: 155pt)
        (p.operator)(at(415, 65 + mla-output-shift), "out-proj", [MatMul], width: 68pt, keys: ("MatMul",))
        edge(<w-out-proj>, <out-proj>, "--|>", stroke: rgb("79517d") + 0.8pt)
      })
      replace.insert("weighted-sum", (p.operator)(at(415, 325 + mla-output-shift), "weighted-sum", [潜空间加权聚合],
        formula: `\sum_{j=1}^{t}A_{t,j,i}\mathbf{c}_{j}^{KV}`, width: 240pt))
      routes += ((from: "ckv", to: "weighted-sum", body:
        (p.wire)(<ckv.east>, at(900, 1025), at(900, 325 + mla-output-shift), <weighted-sum.east>, "-|>")),)
    }
    extra = {
      (p.operator)(at(400, 510 + mla-output-shift), "value-identity", [Value 权重吸收 · 离线预计算],
        formula: `W^{O,UV}=W^O\operatorname{diag}(W_1^{UV},\ldots,W_{n_h}^{UV})`,
        width: 560pt, keys: ("W^O,UV", "W^O", "diag", "W_i^UV", "n_h"))
      (p.operator)(at(400, 580 + mla-output-shift), "value-norm-note", [源潜向量已过 RMSNorm；报告公式省略归一化。],
        width: 560pt, keys: ("c_j^KV", "RMSNorm"))
      (p.operator)(at(400, 625 + mla-output-shift), "value-norm-limit", [RMSNorm 依赖激活，不能吸收到固定权重中。],
        width: 560pt, keys: ("RMSNorm",))
    }
  }
  mla-render(layout, ids: ids, replace: replace, extra: extra, routes: routes,
    bounds-y: if path == "qk" { (480, 1060) } else { (-35 + mla-output-shift, 1060) })
}

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#absorption-diagram(path: sys.inputs.at("path", default: "qk"), stage: sys.inputs.at("stage", default: "before"))
