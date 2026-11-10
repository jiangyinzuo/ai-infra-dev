import { readFile, writeFile } from 'node:fs/promises';
import { mathjax } from 'mathjax-full/js/mathjax.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import { SVG } from 'mathjax-full/js/output/svg.js';
import { liteAdaptor } from 'mathjax-full/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';
import 'mathjax-full/js/input/tex/ams/AmsConfiguration.js';

const name = process.argv[2];
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mathDocument = mathjax.document('', {
  InputJax: new TeX({ packages: ['base', 'ams'] }),
  OutputJax: new SVG({ fontCache: 'none' }),
});
const source = await readFile(new URL(`${name}.source.svg`, import.meta.url), 'utf8');
const parsed = adaptor.parse(source, 'text/html');
const diagram = adaptor.tags(adaptor.body(parsed), 'svg')[0];
let formulaCount = 0;

for (const label of adaptor.tags(diagram, 'text')) {
  const tex = adaptor.getAttribute(label, 'data-tex');
  if (tex === undefined) continue;

  const output = mathDocument.convert(tex, { display: false });
  const math = adaptor.tags(output, 'svg')[0];
  if (adaptor.tags(math, 'g').some(node => adaptor.getAttribute(node, 'data-mml-node') === 'merror')) {
    throw new Error(`Cannot render LaTeX: ${tex}`);
  }
  const [, , width] = adaptor.getAttribute(math, 'viewBox').split(/\s+/).map(Number);
  const fontSize = Number(adaptor.getAttribute(label, 'font-size') || adaptor.getAttribute(diagram, 'font-size'));
  // MathJax uses 1000 SVG units per em; preserve the source label's center and baseline.
  const scale = fontSize / 1000;
  const x = Number(adaptor.getAttribute(label, 'x')) - width * scale / 2;
  const y = Number(adaptor.getAttribute(label, 'y'));
  // Avoid nested </svg> tags: live-server injects a reload script at each closing tag.
  const group = adaptor.node('g', {
    transform: `translate(${x} ${y}) scale(${scale})`,
    color: adaptor.getAttribute(label, 'fill'),
    role: 'img', 'data-tex': tex, 'aria-label': tex,
  }, adaptor.childNodes(math));
  adaptor.replace(group, label);
  formulaCount++;
}

const result = `<!-- Generated from ${name}.source.svg by npm run render:${name}. -->\n`
  + adaptor.outerHTML(diagram) + '\n';
await writeFile(new URL(`${name}.svg`, import.meta.url), result, 'utf8');
console.log(`Rendered ${formulaCount} formulas to ${name}.svg`);
