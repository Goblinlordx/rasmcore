#!/usr/bin/env node
/**
 * SSG Docs Builder — generates a static HTML documentation site from:
 *   1. V2 operation registry (dump_registry JSON → per-operation pages)
 *   2. Manual pages (docs/pages/*.adoc → guide/tutorial pages)
 *
 * Usage: node scripts/build-docs.mjs
 * Output: docs/build/
 *
 * Requires: npm install @asciidoctor/core (or asciidoctor.js)
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');
const outDir = join(projectRoot, 'docs', 'build');
const pagesDir = join(projectRoot, 'docs', 'pages');

// ─── Load Asciidoctor ────────────────────────────────────���─────────────────

let Asciidoctor;
try {
  const mod = await import('@asciidoctor/core');
  Asciidoctor = mod.default || mod;
} catch {
  console.error('Missing dependency: npm install @asciidoctor/core');
  process.exit(1);
}
const asciidoctor = Asciidoctor();

// ��── Get registry JSON ─────────────────────────────────────────────────────

console.log('Building registry dump...');
const registryJson = '/tmp/v2_registry_docs.json';
execSync(`cargo run --bin dump_registry -p rasmcore-v2-wasm 2>/dev/null > ${registryJson}`, { cwd: projectRoot });
const registry = JSON.parse(readFileSync(registryJson, 'utf8'));

const { filters, encoders, decoders } = registry;
console.log(`Registry: ${filters.length} filters, ${encoders.length} encoders, ${decoders.length} decoders`);

// ─── Helpers ────────────────────────────────────────────────────────────────

function snakeToTitle(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function snakeToKebab(s) {
  return s.replace(/_/g, '-');
}

function renderAdoc(content) {
  return asciidoctor.convert(content, { safe: 'safe', standalone: false });
}

function readAdocFile(docPath) {
  if (!docPath) return null;
  const fullPath = join(projectRoot, docPath);
  if (!existsSync(fullPath)) return null;
  return readFileSync(fullPath, 'utf8');
}

// ─── HTML Template ──────────────────────────────────────────────────────────

function htmlPage(title, content, nav, breadcrumb = '') {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title} — rasmcore docs</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <div class="layout">
    <nav class="sidebar">${nav}</nav>
    <main class="content">
      ${breadcrumb ? `<div class="breadcrumb">${breadcrumb}</div>` : ''}
      <h1>${title}</h1>
      ${content}
    </main>
  </div>
</body>
</html>`;
}

// ─── Build Nav Sidebar ──────────────────────────────────────────────────────

function buildNav(categories, manualSections) {
  let nav = '<div class="nav-section"><a href="/index.html" class="nav-home">rasmcore</a></div>\n';

  // Operation categories
  const sortedCats = Object.keys(categories).sort();
  for (const cat of sortedCats) {
    const ops = categories[cat].sort((a, b) => a.name.localeCompare(b.name));
    nav += `<div class="nav-section">
      <div class="nav-heading">${snakeToTitle(cat)}</div>
      <ul>${ops.map(op =>
        `<li><a href="/operations/${snakeToKebab(op.name)}.html">${op.displayName || snakeToTitle(op.name)}</a></li>`
      ).join('\n')}</ul>
    </div>\n`;
  }

  // Encoders
  if (encoders.length > 0) {
    nav += `<div class="nav-section">
      <div class="nav-heading">Encoders</div>
      <ul>${encoders.map(e =>
        `<li><a href="/codecs/${snakeToKebab(e.name)}-encoder.html">${e.displayName}</a></li>`
      ).join('\n')}</ul>
    </div>\n`;
  }

  // Decoders
  if (decoders.length > 0) {
    nav += `<div class="nav-section">
      <div class="nav-heading">Decoders</div>
      <ul>${decoders.map(d =>
        `<li><a href="/codecs/${snakeToKebab(d.name)}-decoder.html">${d.displayName}</a></li>`
      ).join('\n')}</ul>
    </div>\n`;
  }

  // Manual page sections
  for (const [section, pages] of Object.entries(manualSections)) {
    nav += `<div class="nav-section">
      <div class="nav-heading">${section}</div>
      <ul>${pages.sort((a, b) => a.order - b.order).map(p =>
        `<li><a href="/pages/${p.slug}.html">${p.title}</a></li>`
      ).join('\n')}</ul>
    </div>\n`;
  }

  return nav;
}

// ─── Param Table ────────────────────────────────────────────────────────────

function paramTable(params) {
  if (!params || params.length === 0) return '<p><em>No parameters.</em></p>';
  let html = `<table class="param-table">
    <thead><tr><th>Parameter</th><th>Type</th><th>Range</th><th>Default</th><th>Description</th></tr></thead>
    <tbody>`;
  for (const p of params) {
    const range = p.min != null && p.max != null ? `${p.min} – ${p.max}` : '—';
    const step = p.step != null ? ` (step ${p.step})` : '';
    const def = p.default != null ? String(p.default) : '—';
    const desc = p.description || '';
    html += `<tr>
      <td><code>${p.name}</code></td>
      <td><code>${p.type}</code></td>
      <td>${range}${step}</td>
      <td>${def}</td>
      <td>${desc}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  return html;
}

// ─── Group operations by category ───────────────────────────────────────────

const categories = {};
for (const f of filters) {
  const cat = f.category || 'uncategorized';
  if (!categories[cat]) categories[cat] = [];
  categories[cat].push(f);
}

// ─── Load manual pages ──────────────────────────────────────────────────────

const manualSections = {};
const manualPages = [];

if (existsSync(pagesDir)) {
  for (const file of readdirSync(pagesDir)) {
    if (!file.endsWith('.adoc')) continue;
    const raw = readFileSync(join(pagesDir, file), 'utf8');

    // Parse AsciiDoc document attributes for metadata
    let title = basename(file, '.adoc');
    let section = 'Guides';
    let order = 99;

    const titleMatch = raw.match(/^= (.+)$/m);
    if (titleMatch) title = titleMatch[1];

    const sectionMatch = raw.match(/^:section:\s*(.+)$/m);
    if (sectionMatch) section = sectionMatch[1].trim();

    const orderMatch = raw.match(/^:order:\s*(\d+)$/m);
    if (orderMatch) order = parseInt(orderMatch[1]);

    const slug = basename(file, '.adoc');
    const page = { title, section, order, slug, content: raw };
    manualPages.push(page);

    if (!manualSections[section]) manualSections[section] = [];
    manualSections[section].push(page);
  }
}

// ─── Build navigation ───────────────────────────────────────────────────────

const nav = buildNav(categories, manualSections);

// ─── Create output directories ──────────────────────────────────────────────

mkdirSync(join(outDir, 'operations'), { recursive: true });
mkdirSync(join(outDir, 'codecs'), { recursive: true });
mkdirSync(join(outDir, 'pages'), { recursive: true });

// ─── Generate index page ────────────────────────────────────────────────────

let indexContent = '<div class="op-index">\n';
const sortedCats = Object.keys(categories).sort();
for (const cat of sortedCats) {
  const ops = categories[cat].sort((a, b) => a.name.localeCompare(b.name));
  indexContent += `<h2>${snakeToTitle(cat)}</h2>\n<ul>\n`;
  for (const op of ops) {
    const name = op.displayName || snakeToTitle(op.name);
    indexContent += `  <li><a href="/operations/${snakeToKebab(op.name)}.html">${name}</a></li>\n`;
  }
  indexContent += '</ul>\n';
}

if (encoders.length > 0) {
  indexContent += '<h2>Encoders</h2>\n<ul>\n';
  for (const e of encoders) {
    indexContent += `  <li><a href="/codecs/${snakeToKebab(e.name)}-encoder.html">${e.displayName}</a></li>\n`;
  }
  indexContent += '</ul>\n';
}

if (decoders.length > 0) {
  indexContent += '<h2>Decoders</h2>\n<ul>\n';
  for (const d of decoders) {
    indexContent += `  <li><a href="/codecs/${snakeToKebab(d.name)}-decoder.html">${d.displayName}</a></li>\n`;
  }
  indexContent += '</ul>\n';
}
indexContent += '</div>';

writeFileSync(join(outDir, 'index.html'), htmlPage('Operations', indexContent, nav));

// ─── Generate per-operation pages ───────────────────────────────────────────

for (const f of filters) {
  const name = f.displayName || snakeToTitle(f.name);
  const slug = snakeToKebab(f.name);

  let content = '';

  // Render .adoc content if doc_path is set
  const adocContent = readAdocFile(f.docPath);
  if (adocContent) {
    content += `<div class="op-description">${renderAdoc(adocContent)}</div>\n`;
  }

  // Params table (always rendered from registry)
  content += '<h2>Parameters</h2>\n';
  content += paramTable(f.params);

  // Metadata
  content += `<div class="op-meta">
    <p><strong>Category:</strong> ${snakeToTitle(f.category || 'uncategorized')}</p>
    <p><strong>Registry name:</strong> <code>${f.name}</code></p>
  </div>`;

  const breadcrumb = `<a href="/index.html">Operations</a> &rsaquo; ${snakeToTitle(f.category || '')} &rsaquo; ${name}`;
  writeFileSync(join(outDir, 'operations', `${slug}.html`), htmlPage(name, content, nav, breadcrumb));
}

// ─── Generate encoder pages ─────────────────────────────────────────────────

for (const e of encoders) {
  const slug = snakeToKebab(e.name);
  let content = '';
  const adocContent = readAdocFile(e.docPath);
  if (adocContent) {
    content += `<div class="op-description">${renderAdoc(adocContent)}</div>\n`;
  }
  content += `<div class="op-meta">
    <p><strong>MIME type:</strong> <code>${e.mime}</code></p>
    <p><strong>Extensions:</strong> ${e.extensions.map(x => `<code>.${x}</code>`).join(', ')}</p>
  </div>`;
  const breadcrumb = `<a href="/index.html">Operations</a> &rsaquo; Encoders &rsaquo; ${e.displayName}`;
  writeFileSync(join(outDir, 'codecs', `${slug}-encoder.html`), htmlPage(`${e.displayName} Encoder`, content, nav, breadcrumb));
}

// ─── Generate decoder pages ─────────────────────────────────────────────────

for (const d of decoders) {
  const slug = snakeToKebab(d.name);
  let content = '';
  const adocContent = readAdocFile(d.docPath);
  if (adocContent) {
    content += `<div class="op-description">${renderAdoc(adocContent)}</div>\n`;
  }
  content += `<div class="op-meta">
    <p><strong>Extensions:</strong> ${d.extensions.map(x => `<code>.${x}</code>`).join(', ')}</p>
  </div>`;
  const breadcrumb = `<a href="/index.html">Operations</a> &rsaquo; Decoders &rsaquo; ${d.displayName}`;
  writeFileSync(join(outDir, 'codecs', `${slug}-decoder.html`), htmlPage(`${d.displayName} Decoder`, content, nav, breadcrumb));
}

// ─── Generate manual pages ──────────────────────────────────────────────────

for (const page of manualPages) {
  const html = renderAdoc(page.content);
  const breadcrumb = `<a href="/index.html">Home</a> &rsaquo; ${page.section} &rsaquo; ${page.title}`;
  writeFileSync(join(outDir, 'pages', `${page.slug}.html`), htmlPage(page.title, html, nav, breadcrumb));
}

// ─── Generate sitemap.json ──────────────────────────────────────────────────

const sitemap = {
  operations: filters.map(f => ({
    name: f.name,
    displayName: f.displayName || snakeToTitle(f.name),
    category: f.category,
    url: `/operations/${snakeToKebab(f.name)}.html`,
    hasDoc: !!f.docPath,
  })),
  encoders: encoders.map(e => ({
    name: e.name,
    displayName: e.displayName,
    url: `/codecs/${snakeToKebab(e.name)}-encoder.html`,
  })),
  decoders: decoders.map(d => ({
    name: d.name,
    displayName: d.displayName,
    url: `/codecs/${snakeToKebab(d.name)}-decoder.html`,
  })),
  pages: manualPages.map(p => ({
    title: p.title,
    section: p.section,
    url: `/pages/${p.slug}.html`,
  })),
};
writeFileSync(join(outDir, 'sitemap.json'), JSON.stringify(sitemap, null, 2));

// ─── Generate CSS ───────────────────────────────────────────────────────────

writeFileSync(join(outDir, 'style.css'), `
:root {
  --bg: #0d1117;
  --bg-sidebar: #161b22;
  --bg-content: #0d1117;
  --text: #c9d1d9;
  --text-muted: #8b949e;
  --link: #58a6ff;
  --border: #30363d;
  --heading: #e6edf3;
  --code-bg: #1c2128;
  --table-stripe: #161b22;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}

.layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 280px;
  min-width: 280px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  padding: 1rem 0;
  overflow-y: auto;
  position: sticky;
  top: 0;
  height: 100vh;
}

.nav-home {
  display: block;
  padding: 0.5rem 1rem;
  font-weight: 700;
  font-size: 1.1rem;
  color: var(--heading);
  text-decoration: none;
}

.nav-section { margin-bottom: 0.5rem; }

.nav-heading {
  padding: 0.3rem 1rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.sidebar ul {
  list-style: none;
  padding: 0;
}

.sidebar li a {
  display: block;
  padding: 0.2rem 1rem 0.2rem 1.5rem;
  color: var(--text);
  text-decoration: none;
  font-size: 0.85rem;
}

.sidebar li a:hover {
  color: var(--link);
  background: rgba(88, 166, 255, 0.08);
}

.content {
  flex: 1;
  padding: 2rem 3rem;
  max-width: 900px;
}

.breadcrumb {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-bottom: 1rem;
}

.breadcrumb a {
  color: var(--link);
  text-decoration: none;
}

h1 {
  color: var(--heading);
  font-size: 1.8rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}

h2 {
  color: var(--heading);
  font-size: 1.3rem;
  margin: 1.5rem 0 0.8rem;
}

a { color: var(--link); }

code {
  background: var(--code-bg);
  padding: 0.15em 0.4em;
  border-radius: 3px;
  font-size: 0.9em;
}

pre {
  background: var(--code-bg);
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
  margin: 1rem 0;
}

pre code {
  background: none;
  padding: 0;
}

.param-table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}

.param-table th {
  background: var(--bg-sidebar);
  padding: 0.5rem 0.8rem;
  text-align: left;
  font-size: 0.85rem;
  color: var(--text-muted);
  border-bottom: 2px solid var(--border);
}

.param-table td {
  padding: 0.5rem 0.8rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.9rem;
}

.param-table tr:nth-child(even) td {
  background: var(--table-stripe);
}

.op-meta {
  margin-top: 1.5rem;
  padding: 1rem;
  background: var(--bg-sidebar);
  border-radius: 6px;
  border: 1px solid var(--border);
}

.op-meta p { margin: 0.3rem 0; font-size: 0.9rem; }

.op-index ul {
  list-style: none;
  padding: 0;
  columns: 2;
}

.op-index li {
  padding: 0.2rem 0;
}

.op-description {
  margin-bottom: 1.5rem;
}

/* AsciiDoc admonitions */
.admonitionblock {
  margin: 1rem 0;
  padding: 0.8rem 1rem;
  border-left: 4px solid var(--link);
  background: var(--bg-sidebar);
  border-radius: 0 6px 6px 0;
}

.admonitionblock .title {
  font-weight: 700;
  color: var(--heading);
}
`);

// ─── Summary ────────────────────────────────────────────────────────────────

const totalPages = filters.length + encoders.length + decoders.length + manualPages.length + 1;
console.log(`\nGenerated: ${outDir}`);
console.log(`  ${filters.length} filter pages`);
console.log(`  ${encoders.length} encoder pages`);
console.log(`  ${decoders.length} decoder pages`);
console.log(`  ${manualPages.length} manual pages`);
console.log(`  1 index page + sitemap.json`);
console.log(`  ${totalPages} total pages`);
