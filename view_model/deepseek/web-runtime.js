// Browser interactions for Typst HTML; web-template.typ embeds this file into HTML.
// Inline SVG node regions and individual MathML symbols share the same glossary preview templates.
(() => {
  'use strict';
  const configuration = JSON.parse(document.getElementById('document-symbols').textContent);
  const definitions = configuration.definitions;
  const byContext = new Map();
  definitions.forEach(({ key, context }, index) => {
    if (!byContext.has(context)) byContext.set(context, new Map());
    byContext.get(context).set(key, { templateId: 'symbol-definition-' + index });
  });
  const greek = { 'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'ε': 'epsilon', 'ϵ': 'epsilon', 'μ': 'mu', 'σ': 'sigma', 'ℓ': 'ell' };
  const script = { '𝒮': 'calS', '𝓢': 'calS', 'ℋ': 'calH', '𝓗': 'calH', '𝒬': 'calQ', '𝓠': 'calQ', '𝒪': 'calO', '𝓞': 'calO' };
  const clean = text => [...text].map(char => char === 'ℝ' ? 'real' : script[char] || greek[char.normalize('NFKC')] || char.normalize('NFKC')).join('').replace(/[\s\u2061-\u2064\uFE00-\uFE0F]/g, '').replace(/[′’]/g, "'").replace(/≤/g, 'le_');
  function keyOf(node) {
    if (!node || node.nodeType !== Node.ELEMENT_NODE) return '';
    const parts = [...node.children];
    const key = child => keyOf(child);
    const sub = child => key(child).replace(/,/g, '');
    switch (node.localName) {
      case 'msub': return key(parts[0]) + '_' + sub(parts[1]);
      case 'msup': return key(parts[0]) + '^' + key(parts[1]);
      case 'msubsup': return key(parts[0]) + '_' + sub(parts[1]) + '^' + key(parts[2]);
      case 'mover': {
        const accent = parts[1]?.textContent || '';
        const prefix = /[¯‾ˉ̄]/.test(accent) ? 'bar_' : /[~˜∼̃]/.test(accent) ? 'tilde_' : /[ˆ^̂]/.test(accent) ? 'hat_' : '';
        return prefix ? prefix + key(parts[0]) : key(parts[0]) + '^' + key(parts[1]);
      }
      case 'mi': case 'mn': case 'mo': case 'mtext': return clean(node.textContent);
      default: return parts.map(key).join('');
    }
  }
  function lookup(key, context) {
    key = key.replace(/\^'/g, "'");
    const scopes = configuration.fallbacks[context] || [context, 'any'];
    const variants = [key];
    if (key.endsWith('^T')) variants.push(key.slice(0, -2));
    variants.push(key.replace(/_(t|j)(?:[0-9]+|n_h(?:\^I)?)(?=\^|$)/g, '_$1i'));
    variants.push(key.replace(/_[0-9]+(?=\^|$)/g, '_i'));
    variants.push(key.replace(/^([a-z])_([^'^]+)'/, "$1'_$2"));
    for (const variant of variants) {
      for (const scope of scopes) {
        const entry = byContext.get(scope)?.get(variant);
        if (entry) return { key: variant, ...entry, transposed: key.endsWith('^T') && variant !== key };
      }
    }
    return null;
  }
  function buildTableOfContents() {
    const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')];
    const main = document.createElement('main');
    main.id = 'document-content';
    for (const node of [...document.body.childNodes]) {
      if (node.nodeType === Node.ELEMENT_NODE && ['SCRIPT', 'STYLE'].includes(node.tagName)) continue;
      main.append(node);
    }
    document.body.append(main);
    const nav = document.createElement('nav');
    nav.id = 'document-toc';
    nav.setAttribute('aria-label', '目录');
    const details = document.createElement('details');
    const desktop = window.matchMedia('(min-width: 1100px)');
    details.open = desktop.matches;
    desktop.addEventListener('change', event => { details.open = event.matches; });
    const summary = document.createElement('summary');
    summary.textContent = '目录';
    const root = document.createElement('ul');
    const parents = [];
    headings.forEach((heading, index) => {
      if (!heading.id) heading.id = 'section-' + (index + 1);
      const level = Number(heading.tagName.slice(1));
      while (parents.length && parents.at(-1).level >= level) parents.pop();
      let list = root;
      if (parents.length) {
        const parent = parents.at(-1);
        if (!parent.children) {
          parent.children = document.createElement('ul');
          parent.item.append(parent.children);
        }
        list = parent.children;
      }
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = '#' + encodeURIComponent(heading.id);
      link.textContent = heading.textContent;
      item.append(link);
      list.append(item);
      parents.push({ level, item });
    });
    nav.addEventListener('click', event => {
      if (!desktop.matches && event.target.closest('a')) details.open = false;
    });
    details.append(summary, root);
    nav.append(details);
    document.body.insertBefore(nav, main);
    // A freshly loaded fragment needs its target after generated heading IDs exist.
    if (location.hash) {
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (target && main.contains(target)) target.scrollIntoView();
    }
  }
  function initialize() {
    buildTableOfContents();
    let context = configuration.defaultContext;
    for (const node of document.querySelectorAll('h1,h2,h3,h4,h5,h6,math')) {
      if (/^H[1-6]$/.test(node.tagName)) {
        const heading = node.textContent;
        for (const section of configuration.sections) {
          if (heading.includes(section.heading)) context = section.context;
        }
        continue;
      }
      node.dataset.symbolContext = context;
      for (const symbol of node.querySelectorAll('msub,msup,msubsup,mover,mi,mtext,mo')) {
        if (symbol.parentElement.closest('[data-symbol]')) continue;
        // Superscripts such as I, R and C belong to their full symbol, not to a separate parameter.
        if (symbol.localName === 'mo' && !['∑', '⊙'].includes(symbol.textContent)) continue;
        const key = symbol.textContent === '∑' ? 'sum' : symbol.textContent === '⊙' ? 'odot' : keyOf(symbol);
        const entry = lookup(key, context);
        if (!entry) continue;
        symbol.dataset.symbol = entry.key;
        symbol.dataset.definitionTemplate = entry.templateId;
        symbol.dataset.definitionTranspose = String(entry.transposed);
        symbol.setAttribute('tabindex', '0');
      }
    }
    const diagrams = [...document.querySelectorAll('.model-diagram svg')];
    if (diagrams.length) {
      document.getElementById('document-content').style.setProperty('--diagram-reference-width', diagrams[0].viewBox.baseVal.width);
      for (const svg of diagrams) svg.style.setProperty('--diagram-width', svg.viewBox.baseVal.width);
    }
    const diagramPreviews = new WeakMap();
    for (const region of document.querySelectorAll('.model-diagram svg a')) {
      const href = region.getAttribute('href') || region.getAttributeNS('http://www.w3.org/1999/xlink', 'href') || '';
      const prefix = '#diagram-symbols/';
      if (!href.startsWith(prefix)) continue;
      const [context, node, list] = href.slice(prefix.length).split('/');
      const entries = [...new Set(list.split('|'))].map(key => lookup(key, context)).filter(Boolean);
      if (!entries.length) continue;
      diagramPreviews.set(region, entries);
      region.dataset.diagramNode = node;
      region.setAttribute('tabindex', '0');
      region.setAttribute('role', 'button');
      region.setAttribute('aria-label', '框内符号说明：' + list.split('|').join('、'));
      region.removeAttribute('href');
      region.removeAttributeNS('http://www.w3.org/1999/xlink', 'href');
    }
    for (const table of document.querySelectorAll('table')) {
      const scroll = document.createElement('div');
      scroll.className = 'table-scroll';
      table.before(scroll);
      scroll.append(table);
    }
    const tooltip = document.createElement('div');
    tooltip.id = 'symbol-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.hidden = true;
    document.body.append(tooltip);
    let active = null;
    let hideTimer;
    function hide() {
      clearTimeout(hideTimer);
      if (active) active.removeAttribute('aria-describedby');
      active = null;
      tooltip.hidden = true;
    }
    function symbolLabel(entry) {
      const supplied = document.getElementById(entry.templateId.replace('symbol-definition-', 'symbol-label-'));
      if (supplied) return supplied.content.cloneNode(true);
      const source = document.querySelector('#document-content math [data-symbol="' + CSS.escape(entry.key) + '"]');
      if (!source) return document.createTextNode(entry.key);
      const copy = source.cloneNode(true);
      for (const node of [copy, ...copy.querySelectorAll('*')]) {
        for (const attr of [...node.attributes]) {
          if (attr.name.startsWith('data-') || ['tabindex', 'aria-describedby'].includes(attr.name)) node.removeAttribute(attr.name);
        }
      }
      const math = document.createElementNS('http://www.w3.org/1998/Math/MathML', 'math');
      math.append(copy);
      return math;
    }
    function show(symbol, x, y) {
      if (active && active !== symbol) active.removeAttribute('aria-describedby');
      active = symbol;
      clearTimeout(hideTimer);
      tooltip.classList.toggle('diagram-tooltip', diagramPreviews.has(symbol));
      tooltip.replaceChildren();
      if (diagramPreviews.has(symbol)) {
        for (const entry of diagramPreviews.get(symbol)) {
          const section = document.createElement('section');
          section.className = 'diagram-definition';
          section.dataset.definitionKey = entry.key;
          const label = document.createElement('div');
          label.className = 'tooltip-symbol';
          label.append(symbolLabel(entry));
          const preview = document.createElement('div');
          preview.className = 'definition-preview';
          preview.append(document.getElementById(entry.templateId).content.cloneNode(true));
          section.append(label, preview);
          tooltip.append(section);
        }
      } else {
        const definition = document.getElementById(symbol.dataset.definitionTemplate);
        tooltip.append(definition.content.cloneNode(true));
        if (symbol.dataset.definitionTranspose === 'true') tooltip.append(' 此处取转置。');
      }
      tooltip.hidden = false;
      symbol.setAttribute('aria-describedby', tooltip.id);
      tooltip.style.maxHeight = '';
      const naturalBox = tooltip.getBoundingClientRect();
      const anchor = symbol.getBoundingClientRect();
      const top = Math.min(y, anchor.top);
      const bottom = Math.max(y, anchor.bottom);
      const spaceBelow = Math.max(0, innerHeight - bottom - 24);
      const spaceAbove = Math.max(0, top - 20);
      const below = naturalBox.height <= spaceBelow || (naturalBox.height > spaceAbove && spaceBelow >= spaceAbove);
      tooltip.style.maxHeight = (below ? spaceBelow : spaceAbove) + 'px';
      const box = tooltip.getBoundingClientRect();
      tooltip.style.left = Math.max(8, Math.min(x + 12, innerWidth - box.width - 8)) + 'px';
      tooltip.style.top = (below ? bottom + 16 : Math.max(8, top - box.height - 12)) + 'px';
    }
    document.addEventListener('pointerover', event => {
      const symbol = event.target.closest?.('[data-symbol], [data-diagram-node]');
      if (symbol) show(symbol, event.clientX, event.clientY);
      else if (tooltip.contains(event.target)) clearTimeout(hideTimer);
    });
    document.addEventListener('pointerout', event => {
      if (active && !active.contains(event.relatedTarget) && !tooltip.contains(event.relatedTarget)) {
        clearTimeout(hideTimer);
        hideTimer = setTimeout(hide, 150);
      }
    });
    document.addEventListener('focusin', event => {
      const symbol = event.target.closest?.('[data-symbol], [data-diagram-node]');
      if (symbol) {
        const box = symbol.getBoundingClientRect();
        show(symbol, box.left, box.bottom);
      }
    });
    document.addEventListener('click', event => {
      const region = event.target.closest?.('[data-diagram-node]');
      if (!region) return;
      event.preventDefault();
      region.focus();
      const box = region.getBoundingClientRect();
      show(region, box.left, box.bottom);
    });
    tooltip.addEventListener('pointerleave', hide);
    document.addEventListener('focusout', hide);
    document.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
    window.addEventListener('scroll', event => {
      if (tooltip.contains(event.target)) return;
      if (active && active === document.activeElement) {
        const box = active.getBoundingClientRect();
        show(active, box.left, box.bottom);
      } else hide();
    }, true);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
  else initialize();
})();
