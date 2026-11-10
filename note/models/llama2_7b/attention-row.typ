// Export SVG: typst compile attention-row.typ attention-row.svg
// The document embeds the exported SVG; all grids and formulas are drawn here.
#set page(width: 960pt, height: 930pt, margin: 0pt, fill: none)
#set text(font: ("Noto Sans CJK SC", "New Computer Modern"), size: 17pt)
#set math.equation(numbering: none)
#set par(leading: 0pt)

#let label(x, y, width, height, body, size: 17pt) = place(
  top + left, dx: x * 1pt, dy: y * 1pt,
  box(width: width * 1pt, height: height * 1pt,
    align(center + horizon, text(size: size, body))),
)

#let grid-at(x, y, cell-width, row-height, cells, selected: none) = {
  let columns = cells.first().len()
  place(top + left, dx: x * 1pt, dy: y * 1pt,
    table(
      columns: (cell-width * 1pt,) * columns,
      rows: (row-height * 1pt,) * cells.len(),
      align: center + horizon, inset: 0pt, stroke: 0.5pt,
      ..cells.flatten(),
    ),
  )
  if selected != none {
    place(top + left, dx: x * 1pt, dy: (y + selected * row-height) * 1pt,
      rect(width: columns * cell-width * 1pt, height: row-height * 1pt,
        fill: none, stroke: 1.7pt),
    )
  }
}

#label(24, 12, 912, 32, [矩阵乘法 → 取一行 → 向量线性组合], size: 21pt)
#label(62, 54, 270, 30, $A in bb(R)^(4 times 4)$)
#label(400, 54, 210, 30, $V in bb(R)^(4 times 3)$)
#label(696, 54, 210, 30, $O in bb(R)^(4 times 3)$)

#for (i, column) in ($1$, $2$, $3$, $4$).enumerate() {
  label(62 + i * 67.5, 84, 67.5, 22, column, size: 14pt)
}
#for (i, column) in ($1$, $2$, $3$).enumerate() {
  label(400 + i * 70, 84, 70, 22, column, size: 14pt)
  label(696 + i * 70, 84, 70, 22, column, size: 14pt)
}
#for (i, row) in ($1$, $2$, $3$, $4$).enumerate() {
  label(16, 106 + i * 47.5, 34, 47.5, row, size: 15pt)
}

#grid-at(62, 106, 67.5, 47.5, (
  ($alpha_(11)$, $alpha_(12)$, $alpha_(13)$, $alpha_(14)$),
  ($alpha_(21)$, $alpha_(22)$, $alpha_(23)$, $alpha_(24)$),
  ($alpha_(31)$, $alpha_(32)$, $alpha_(33)$, $alpha_(34)$),
  ($alpha_(41)$, $alpha_(42)$, $alpha_(43)$, $alpha_(44)$),
), selected: 2)
#label(340, 106, 48, 190, $times$, size: 27pt)
#grid-at(400, 106, 70, 47.5, (
  ($v_(11)$, $v_(12)$, $v_(13)$),
  ($v_(21)$, $v_(22)$, $v_(23)$),
  ($v_(31)$, $v_(32)$, $v_(33)$),
  ($v_(41)$, $v_(42)$, $v_(43)$),
))
#label(626, 106, 48, 190, $=$, size: 27pt)
#grid-at(696, 106, 70, 47.5, (
  ($o_(11)$, $o_(12)$, $o_(13)$),
  ($o_(21)$, $o_(22)$, $o_(23)$),
  ($o_(31)$, $o_(32)$, $o_(33)$),
  ($o_(41)$, $o_(42)$, $o_(43)$),
), selected: 2)

#label(24, 308, 912, 30, [↓ 第 3 行（$t = 3$）], size: 16pt)
#label(24, 344, 912, 42,
  $O_(3 comma :) = sum_(j=1)^4 A_(3 j) V_(j comma :) = sum_(j=1)^4 alpha_(3 j) v_j = o_3$,
  size: 20pt,
)
#label(24, 386, 912, 28,
  $A_(3 j) = alpha_(3 j), quad V_(j comma :) = v_j, quad O_(3 comma :) = o_3$,
  size: 16pt,
)
#place(top + left, dx: 24pt, dy: 426pt, line(length: 912pt, stroke: 0.5pt))

#label(24, 442, 120, 30, [$t = 3$], size: 20pt)
#label(160, 436, 758, 42,
  $o_3 = alpha_(31) v_1 + alpha_(32) v_2 + alpha_(33) v_3 + alpha_(34) v_4$,
  size: 20pt,
)
#label(130, 480, 276, 24, $v_j$, size: 15pt)
#label(510, 480, 408, 24, $alpha_(3 j) v_j$, size: 15pt)

#for j in range(1, 5) {
  let y = 512 + (j - 1) * 56
  label(24, y, 98, 34, $alpha_(3 #j) times$)
  grid-at(130, y, 92, 34, ((
    $v_(#j 1)$, $v_(#j 2)$, $v_(#j 3)$,
  ),))
  label(420, y, 44, 34, $=$, size: 22pt)
  grid-at(510, y, 136, 34, ((
    $alpha_(3 #j) v_(#j 1)$,
    $alpha_(3 #j) v_(#j 2)$,
    $alpha_(3 #j) v_(#j 3)$,
  ),))
  if j > 1 {
    label(698, y - 22, 32, 22, $+$, size: 18pt)
  }
}
#place(top + left, dx: 470pt, dy: 732pt, line(length: 448pt, stroke: 1pt))
#label(24, 746, 382, 36, [按分量相加], size: 16pt)
#label(408, 746, 94, 36, $o_3 =$)
#grid-at(510, 746, 136, 36, ((
  $o_(31)$, $o_(32)$, $o_(33)$,
),), selected: 0)
#label(24, 790, 912, 30,
  $o_(3 r) = alpha_(31) v_(1 r) + alpha_(32) v_(2 r) + alpha_(33) v_(3 r) + alpha_(34) v_(4 r), quad r = 1, 2, 3$,
  size: 16pt,
)

#place(top + left, dx: 24pt, dy: 836pt, line(length: 912pt, stroke: 0.5pt))
#label(24, 846, 912, 32,
  [$alpha_(3 j)$（标量）：Attention 中，第 3 个 Query 对第 $j$ 个 token 的注意力权重],
  size: 16pt,
)
#label(24, 884, 912, 32,
  [$v_j = x_j W_V$（行向量）：第 $j$ 个 token 经 Value 投影后的内容表示],
  size: 16pt,
)
