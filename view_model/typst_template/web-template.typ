// HTML-only document template. The emitted page embeds all styles, scripts and symbol data.
// Edit web-style.css and web-runtime.js; read() embeds both files and tracks them for live reload.
// Usage: import the helpers and your symbols.typ, then #show: web-document.with(symbols: symbols)
// symbols: {definitions: [{key, preview, context, label?}], defaultContext, sections: [{heading, context}], fallbacks}.
// TOC: sidebar at >=1100px; collapsible above the text on narrower screens.
// Write native $...$ equations; use vbold(c) for upright-bold vectors and #boxed[$ ... $] for frames.
// Symbol previews are native Typst content blocks; they can contain text, math, images and tables.
#let horizontalrule = html.elem("hr")
#let Red(body) = html.elem("mstyle", attrs: (mathcolor: "red"), body)
#let Green(body) = html.elem("mstyle", attrs: (mathcolor: "green"), body)
#let Blue(body) = html.elem("mstyle", attrs: (mathcolor: "blue"), body)
#let TextRed(body) = html.elem("span", attrs: (style: "color: red"), body)
#let vbold(body) = math.upright(math.bold(body))
#let boxed(body) = box(stroke: black, inset: 3pt, body)

#let web-style = read("web-style.css")
#let web-runtime = read("web-runtime.js")

#let web-document(
  symbols: (definitions: (), defaultContext: "any", sections: (), fallbacks: (:)),
  body,
) = {
  set text(lang: "zh")
  set smartquote(enabled: false)
  // HTML frames use paged layout; preserve the native drawing elements inside them.
  show box: it => context {
    if target() == "html" {
      html.elem("mrow", attrs: (style: "border: 1px solid currentColor; padding: 0.15em 0.25em"), it.body)
    } else { it }
  }
  show scale: it => context {
    if target() == "html" {
      html.elem("mstyle", attrs: (mathsize: str(it.x / 1%) + "%"), it.body)
    } else { it }
  }
  html.elem("style", web-style)
  // Only the symbol index is JSON; rich preview content stays in HTML templates.
  let metadata = (
    defaultContext: symbols.defaultContext,
    sections: symbols.sections,
    fallbacks: symbols.fallbacks,
    definitions: symbols.definitions.map(entry => (key: entry.key, "context": entry.at("context"))),
  )
  let payload = json.encode(metadata).replace("<", "\\u003c")
  html.elem("script", attrs: (id: "document-symbols", type: "application/json"), payload)
  for (index, entry) in symbols.definitions.enumerate() {
    html.elem("template", attrs: (id: "symbol-definition-" + str(index)), entry.preview)
    if "label" in entry {
      html.elem("template", attrs: (id: "symbol-label-" + str(index)), entry.label)
    }
  }
  html.elem("script", web-runtime)
  body
}
