// Compile standalone: typst compile diagrams/deepseek-mla.typ diagrams/deepseek-mla.svg
#import "mla-common.typ": mla-layout, mla-render, mla-output-shift

// Preserve the shared horizontal scale even when a parameter moves to the other side.
#let mla-diagram() = context mla-render(mla-layout(), bounds-y: (mla-output-shift - 14.5, 1214.5))

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#mla-diagram()
