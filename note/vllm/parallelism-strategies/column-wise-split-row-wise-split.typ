#set page(width: 330mm, height: 260mm, margin: 10mm)
#set text(size: 10pt)
#set par(leading: 0.55em)

#let ink = rgb("#334155")
#let line = rgb("#64748b")
#let blue = rgb("#dbeafe")
#let orange = rgb("#ffedd5")
#let cell(body, fill: none) = table.cell(fill: fill)[#body]
#let caption(name, dims) = [#text(weight: "bold")[#name] #text(size: 8pt, fill: ink)[(#dims)]]

#let x_plain = table(
  columns: 4,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($x_(1,1)$), cell($x_(1,2)$), cell($x_(1,3)$), cell($x_(1,4)$),
  cell($x_(2,1)$), cell($x_(2,2)$), cell($x_(2,3)$), cell($x_(2,4)$),
)

#let x_cols = table(
  columns: 4,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($x_(1,1)$, fill: blue), cell($x_(1,2)$, fill: blue), cell($x_(1,3)$, fill: orange), cell($x_(1,4)$, fill: orange),
  cell($x_(2,1)$, fill: blue), cell($x_(2,2)$, fill: blue), cell($x_(2,3)$, fill: orange), cell($x_(2,4)$, fill: orange),
)

#let w_cols = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($w_(1,1)$, fill: blue), cell($w_(1,2)$, fill: blue), cell($w_(1,3)$, fill: blue), cell($w_(1,4)$, fill: orange), cell($w_(1,5)$, fill: orange), cell($w_(1,6)$, fill: orange),
  cell($w_(2,1)$, fill: blue), cell($w_(2,2)$, fill: blue), cell($w_(2,3)$, fill: blue), cell($w_(2,4)$, fill: orange), cell($w_(2,5)$, fill: orange), cell($w_(2,6)$, fill: orange),
  cell($w_(3,1)$, fill: blue), cell($w_(3,2)$, fill: blue), cell($w_(3,3)$, fill: blue), cell($w_(3,4)$, fill: orange), cell($w_(3,5)$, fill: orange), cell($w_(3,6)$, fill: orange),
  cell($w_(4,1)$, fill: blue), cell($w_(4,2)$, fill: blue), cell($w_(4,3)$, fill: blue), cell($w_(4,4)$, fill: orange), cell($w_(4,5)$, fill: orange), cell($w_(4,6)$, fill: orange),
)

#let w_rows = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($w_(1,1)$, fill: blue), cell($w_(1,2)$, fill: blue), cell($w_(1,3)$, fill: blue), cell($w_(1,4)$, fill: blue), cell($w_(1,5)$, fill: blue), cell($w_(1,6)$, fill: blue),
  cell($w_(2,1)$, fill: blue), cell($w_(2,2)$, fill: blue), cell($w_(2,3)$, fill: blue), cell($w_(2,4)$, fill: blue), cell($w_(2,5)$, fill: blue), cell($w_(2,6)$, fill: blue),
  cell($w_(3,1)$, fill: orange), cell($w_(3,2)$, fill: orange), cell($w_(3,3)$, fill: orange), cell($w_(3,4)$, fill: orange), cell($w_(3,5)$, fill: orange), cell($w_(3,6)$, fill: orange),
  cell($w_(4,1)$, fill: orange), cell($w_(4,2)$, fill: orange), cell($w_(4,3)$, fill: orange), cell($w_(4,4)$, fill: orange), cell($w_(4,5)$, fill: orange), cell($w_(4,6)$, fill: orange),
)

#let y_cols = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($y_(1,1)$, fill: blue), cell($y_(1,2)$, fill: blue), cell($y_(1,3)$, fill: blue), cell($y_(1,4)$, fill: orange), cell($y_(1,5)$, fill: orange), cell($y_(1,6)$, fill: orange),
  cell($y_(2,1)$, fill: blue), cell($y_(2,2)$, fill: blue), cell($y_(2,3)$, fill: blue), cell($y_(2,4)$, fill: orange), cell($y_(2,5)$, fill: orange), cell($y_(2,6)$, fill: orange),
)

#let y_plain = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($y_(1,1)$), cell($y_(1,2)$), cell($y_(1,3)$), cell($y_(1,4)$), cell($y_(1,5)$), cell($y_(1,6)$),
  cell($y_(2,1)$), cell($y_(2,2)$), cell($y_(2,3)$), cell($y_(2,4)$), cell($y_(2,5)$), cell($y_(2,6)$),
)
#let x_left = table(
  columns: 2,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($x_(1,1)$, fill: blue), cell($x_(1,2)$, fill: blue),
  cell($x_(2,1)$, fill: blue), cell($x_(2,2)$, fill: blue),
)

#let x_right = table(
  columns: 2,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($x_(1,3)$, fill: orange), cell($x_(1,4)$, fill: orange),
  cell($x_(2,3)$, fill: orange), cell($x_(2,4)$, fill: orange),
)

#let w_top = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($w_(1,1)$, fill: blue), cell($w_(1,2)$, fill: blue), cell($w_(1,3)$, fill: blue), cell($w_(1,4)$, fill: blue), cell($w_(1,5)$, fill: blue), cell($w_(1,6)$, fill: blue),
  cell($w_(2,1)$, fill: blue), cell($w_(2,2)$, fill: blue), cell($w_(2,3)$, fill: blue), cell($w_(2,4)$, fill: blue), cell($w_(2,5)$, fill: blue), cell($w_(2,6)$, fill: blue),
)

#let w_bottom = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($w_(3,1)$, fill: orange), cell($w_(3,2)$, fill: orange), cell($w_(3,3)$, fill: orange), cell($w_(3,4)$, fill: orange), cell($w_(3,5)$, fill: orange), cell($w_(3,6)$, fill: orange),
  cell($w_(4,1)$, fill: orange), cell($w_(4,2)$, fill: orange), cell($w_(4,3)$, fill: orange), cell($w_(4,4)$, fill: orange), cell($w_(4,5)$, fill: orange), cell($w_(4,6)$, fill: orange),
)

#let y_top = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($y^T_(1,1)$, fill: blue), cell($y^T_(1,2)$, fill: blue), cell($y^T_(1,3)$, fill: blue), cell($y^T_(1,4)$, fill: blue), cell($y^T_(1,5)$, fill: blue), cell($y^T_(1,6)$, fill: blue),
  cell($y^T_(2,1)$, fill: blue), cell($y^T_(2,2)$, fill: blue), cell($y^T_(2,3)$, fill: blue), cell($y^T_(2,4)$, fill: blue), cell($y^T_(2,5)$, fill: blue), cell($y^T_(2,6)$, fill: blue),
)

#let y_bottom = table(
  columns: 6,
  stroke: 0.7pt + line,
  inset: 4pt,
  align: center + horizon,
  cell($y^B_(1,1)$, fill: orange), cell($y^B_(1,2)$, fill: orange), cell($y^B_(1,3)$, fill: orange), cell($y^B_(1,4)$, fill: orange), cell($y^B_(1,5)$, fill: orange), cell($y^B_(1,6)$, fill: orange),
  cell($y^B_(2,1)$, fill: orange), cell($y^B_(2,2)$, fill: orange), cell($y^B_(2,3)$, fill: orange), cell($y^B_(2,4)$, fill: orange), cell($y^B_(2,5)$, fill: orange), cell($y^B_(2,6)$, fill: orange),
)


#align(center)[#text(size: 19pt, weight: "bold")[矩阵乘法：W 的两种切分方式]]
#v(3pt)
#align(center)[#text(size: 10pt, fill: ink)[$X_(2 times 4) times W_(4 times 6) = Y_(2 times 6)$]]
#v(10pt)

#text(size: 13pt, weight: "bold")[1. W 纵向切分（按列切，输出拼接）]
#v(5pt)
#align(center)[#grid(
  columns: (auto, auto, auto, auto, auto),
  column-gutter: 10pt,
  align: center,
  stack(dir: ttb, spacing: 4pt, caption("X", "2 × 4"), x_plain),
  [$times$],
  stack(dir: ttb, spacing: 4pt, caption("W", "4 × 6"), w_cols),
  [$=$],
  stack(dir: ttb, spacing: 4pt, caption("Y", "2 × 6"), y_cols),
)]
#align(center)[#text(size: 9pt, fill: ink)[蓝色：W_L (4 × 3) → Y_L (2 × 3)　　橙色：W_R (4 × 3) → Y_R (2 × 3)]]
#align(center)[#text(size: 10pt)[$Y = [Y_L | Y_R] = [X W_L | X W_R]$]]

#v(9pt)
#std.line(length: 100%, stroke: 0.7pt + line)
#v(8pt)

#text(size: 13pt, weight: "bold")[2. W 横向切分（按行切，部分结果相加）]
#v(5pt)
#align(center)[#grid(
  columns: (auto, auto, auto, auto, auto),
  column-gutter: 10pt,
  align: center,
  stack(dir: ttb, spacing: 4pt, caption("X", "2 × 4"), x_cols),
  [$times$],
  stack(dir: ttb, spacing: 4pt, caption("W", "4 × 6"), w_rows),
  [$=$],
  stack(dir: ttb, spacing: 4pt, caption("Y", "2 × 6"), y_plain),
)]
#align(center)[#text(size: 9pt, fill: ink)[蓝色对应：X_L (2 × 2) 与 W_T (2 × 6)　　橙色对应：X_R (2 × 2) 与 W_B (2 × 6)]]
#v(3pt)
#align(center)[#text(size: 10pt, weight: "bold")[分别计算两个部分积]]
#align(center)[#grid(
  columns: (auto, auto, auto, auto, auto),
  column-gutter: 10pt,
  align: center,
  stack(dir: ttb, spacing: 4pt, caption("X_L", "2 × 2"), x_left),
  [$times$],
  stack(dir: ttb, spacing: 4pt, caption("W_T", "2 × 6"), w_top),
  [$=$],
  stack(dir: ttb, spacing: 4pt, caption("Y_T", "2 × 6"), y_top),
)]
#align(center)[#text(size: 10pt)[$Y_T = X_L W_T$]]
#v(4pt)
#align(center)[#grid(
  columns: (auto, auto, auto, auto, auto),
  column-gutter: 10pt,
  align: center,
  stack(dir: ttb, spacing: 4pt, caption("X_R", "2 × 2"), x_right),
  [$times$],
  stack(dir: ttb, spacing: 4pt, caption("W_B", "2 × 6"), w_bottom),
  [$=$],
  stack(dir: ttb, spacing: 4pt, caption("Y_B", "2 × 6"), y_bottom),
)]
#align(center)[#text(size: 10pt)[$Y_B = X_R W_B$]]
#v(4pt)
#align(center)[#text(size: 10pt, weight: "bold")[将两个同形状的部分积逐项相加]]
#align(center)[#grid(
  columns: (auto, auto, auto, auto, auto),
  column-gutter: 10pt,
  align: center,
  stack(dir: ttb, spacing: 4pt, caption("Y_T", "2 × 6"), y_top),
  [$+$],
  stack(dir: ttb, spacing: 4pt, caption("Y_B", "2 × 6"), y_bottom),
  [$=$],
  stack(dir: ttb, spacing: 4pt, caption("Y", "2 × 6"), y_plain),
)]
#align(center)[#text(size: 10pt)[$Y = Y_T + Y_B = X_L W_T + X_R W_B$]]