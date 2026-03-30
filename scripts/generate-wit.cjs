#!/usr/bin/env node
/**
 * Generate WIT interface files from #[register_filter] annotations.
 *
 * Reads:  crates/rasmcore-image/src/domain/filters.rs
 *         wit/image/filters.wit.tmpl
 *         wit/image/pipeline.wit.tmpl
 *
 * Writes: wit/image/filters.wit
 *         wit/image/pipeline.wit
 *
 * No external dependencies — uses only Node.js built-ins.
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FILTERS_RS = path.join(ROOT, 'crates/rasmcore-image/src/domain/filters.rs');
const FILTERS_TMPL = path.join(ROOT, 'wit/image/filters.wit.tmpl');
const PIPELINE_TMPL = path.join(ROOT, 'wit/image/pipeline.wit.tmpl');
const FILTERS_WIT = path.join(ROOT, 'wit/image/filters.wit');
const PIPELINE_WIT = path.join(ROOT, 'wit/image/pipeline.wit');

// Rust type → WIT type
const TYPE_MAP = {
  'f32': 'f32', 'f64': 'f64',
  'u8': 'u8', 'u16': 'u16', 'u32': 'u32', 'u64': 'u64',
  'i32': 's32', 'i64': 's64',
  'bool': 'bool',
  'String': 'string', '&str': 'string',
  '&[f32]': 'list<f32>', '&[f64]': 'list<f64>',
  '&[u8]': 'list<u8>', '&[u32]': 'list<u32>',
};

function toWitType(rustType) { return TYPE_MAP[rustType] || 'f32'; }
function toWitName(rustName) { return rustName.replace(/^_+/, '').replace(/_/g, '-'); }

function parseRegisteredFilters(source) {
  const lines = source.split('\n');
  const filters = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line.includes('register_filter(') || line.startsWith('//')) continue;

    // Collect full attribute text (may span multiple lines)
    let attr = line;
    let k = i + 1;
    while (!attr.includes(')]') && k < lines.length) {
      attr += ' ' + lines[k].trim();
      k++;
    }

    const nameM = attr.match(/name\s*=\s*"([^"]+)"/);
    const catM = attr.match(/category\s*=\s*"([^"]+)"/);
    const refM = attr.match(/reference\s*=\s*"([^"]+)"/);
    if (!nameM || !catM) continue;

    // Find next pub fn after attribute closes
    let j = k;
    let fnSig = null;
    while (j < lines.length) {
      const fl = lines[j].trim();
      if (fl.startsWith('pub fn ')) {
        let sig = fl;
        while (!sig.includes('->') && j + 1 < lines.length) {
          j++;
          sig += ' ' + lines[j].trim();
        }
        fnSig = sig;
        break;
      }
      if (fl.startsWith('//') || fl.startsWith('#[') || fl === '') { j++; continue; }
      break;
    }
    if (!fnSig) continue;

    // Parse parameters (skip pixels and info)
    const parenStart = fnSig.indexOf('(');
    const parenEnd = fnSig.indexOf(')');
    const paramsStr = fnSig.slice(parenStart + 1, parenEnd);
    const params = [];
    for (const p of paramsStr.split(',')) {
      const trimmed = p.trim();
      if (!trimmed.includes(':')) continue;
      const [pname, ...ptypeArr] = trimmed.split(':');
      const ptype = ptypeArr.join(':').trim();
      const name = pname.trim();
      if (name === 'pixels' || name === 'info' || ptype === '&[u8]' || ptype === '&ImageInfo') continue;
      params.push({ name, type: ptype });
    }

    filters.push({
      name: nameM[1],
      category: catM[1],
      reference: refM ? refM[1] : '',
      params,
    });
  }
  return filters;
}

function generateFiltersWit(filters) {
  return filters.map(f => {
    const witParams = f.params.map(p => `${toWitName(p.name)}: ${toWitType(p.type)}`).join(', ');
    const doc = `    /// ${f.reference || f.name + ' filter'}`;
    const sig = witParams
      ? `    ${toWitName(f.name)}: func(pixels: buffer, info: image-info, ${witParams}) -> result<buffer, rasmcore-error>;`
      : `    ${toWitName(f.name)}: func(pixels: buffer, info: image-info) -> result<buffer, rasmcore-error>;`;
    return `${doc}\n${sig}\n`;
  }).join('');
}

function generatePipelineWit(filters) {
  return filters.map(f => {
    const witParams = f.params.map(p => `${toWitName(p.name)}: ${toWitType(p.type)}`).join(', ');
    return witParams
      ? `        ${toWitName(f.name)}: func(source: node-id, ${witParams}) -> result<node-id, rasmcore-error>;`
      : `        ${toWitName(f.name)}: func(source: node-id) -> result<node-id, rasmcore-error>;`;
  }).join('\n');
}

// Main
const source = fs.readFileSync(FILTERS_RS, 'utf8');
const filters = parseRegisteredFilters(source);
console.log(`Parsed ${filters.length} registered filters from filters.rs`);

if (fs.existsSync(FILTERS_TMPL)) {
  const tmpl = fs.readFileSync(FILTERS_TMPL, 'utf8');
  fs.writeFileSync(FILTERS_WIT, tmpl.replace('{{GENERATED_FILTERS}}', generateFiltersWit(filters)));
  console.log(`Generated: wit/image/filters.wit (${filters.length} filter functions)`);
}

if (fs.existsSync(PIPELINE_TMPL)) {
  const tmpl = fs.readFileSync(PIPELINE_TMPL, 'utf8');
  fs.writeFileSync(PIPELINE_WIT, tmpl.replace('{{GENERATED_PIPELINE_FILTERS}}', generatePipelineWit(filters)));
  console.log(`Generated: wit/image/pipeline.wit (${filters.length} pipeline methods)`);
}
