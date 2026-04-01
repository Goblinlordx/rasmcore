#!/usr/bin/env node
/**
 * Generate WIT interface files from #[register_filter] annotations.
 *
 * Reads:  crates/rasmcore-image/src/domain/filters.rs
 *         crates/rasmcore-image/src/domain/param_types.rs
 *         wit/image/filters.wit.tmpl
 *         wit/image/pipeline.wit.tmpl
 *
 * Writes: wit/image/filters.wit
 *         wit/image/pipeline.wit
 *
 * No external dependencies — uses only Node.js built-ins.
 *
 * ConfigParams structs are the WIT contract: every ConfigParams struct
 * becomes a WIT record. Fields must be primitives (f32, u32, u8, bool)
 * or other ConfigParams types (ColorRgba, ColorRgb, Point2D). If a field
 * type is not WIT-mappable, this script errors with a clear message.
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FILTERS_RS = path.join(ROOT, 'crates/rasmcore-image/src/domain/filters.rs');
const PARAM_TYPES_RS = path.join(ROOT, 'crates/rasmcore-image/src/domain/param_types.rs');
const FILTERS_TMPL = path.join(ROOT, 'wit/image/filters.wit.tmpl');
const PIPELINE_TMPL = path.join(ROOT, 'wit/image/pipeline.wit.tmpl');
const FILTERS_WIT = path.join(ROOT, 'wit/image/filters.wit');
const PIPELINE_WIT = path.join(ROOT, 'wit/image/pipeline.wit');

// Rust primitive type → WIT type
const PRIMITIVE_TYPE_MAP = {
  'f32': 'f32', 'f64': 'f64',
  'u8': 'u8', 'u16': 'u16', 'u32': 'u32', 'u64': 'u64',
  'i32': 's32', 'i64': 's64',
  'bool': 'bool',
  'String': 'string', '&str': 'string',
  '&[f32]': 'list<f32>', '&[f64]': 'list<f64>',
  '&[u8]': 'list<u8>', '&[u32]': 'list<u32>',
  '&[Point2D]': 'list<point2d>',
};

function toWitName(rustName) { return rustName.replace(/^_+/, '').replace(/_/g, '-'); }

// ─── ConfigParams struct parser ──────────────────────────────────────────

/**
 * Parse all #[derive(ConfigParams)] structs from source files.
 * Returns a map: structName → { fields: [{name, type}], witRecordName }
 */
function parseConfigParamsStructs(sources) {
  const structs = new Map();

  for (const source of sources) {
    const lines = source.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // Look for ConfigParams derive
      if (!line.includes('ConfigParams') || !line.includes('derive')) continue;

      // Find the struct name
      let j = i + 1;
      while (j < lines.length) {
        const sl = lines[j].trim();
        if (sl.startsWith('pub struct ')) {
          const nameMatch = sl.match(/pub struct (\w+)/);
          if (!nameMatch) { j++; continue; }
          const structName = nameMatch[1];

          // Parse fields until closing brace
          const fields = [];
          let k = j + 1;
          while (k < lines.length && !lines[k].trim().startsWith('}')) {
            const fl = lines[k].trim();
            // Match: pub field_name: Type,
            const fieldMatch = fl.match(/^pub\s+(\w+)\s*:\s*(.+?)\s*,?\s*$/);
            if (fieldMatch) {
              let fieldType = fieldMatch[2].replace(/,$/, '').trim();
              // Resolve fully-qualified paths: crate::domain::param_types::ColorRgba → ColorRgba
              const lastSeg = fieldType.split('::').pop();
              if (lastSeg && fieldType.includes('::')) {
                fieldType = lastSeg;
              }
              fields.push({ name: fieldMatch[1], type: fieldType });
            }
            k++;
          }

          const witRecordName = toWitName(structName.replace(/Params$/, '').replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase()) + '-config';
          structs.set(structName, { fields, witRecordName });
          break;
        }
        if (sl.startsWith('pub fn ') || sl.startsWith('impl ')) break; // not a struct
        j++;
      }
    }
  }
  return structs;
}

// ─── Filter parser ───────────────────────────────────────────────────────

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
      if (name === 'pixels' || name === 'info' || name === 'request' || name === 'upstream' || ptype === '&[u8]' || ptype === '&ImageInfo' || ptype.includes('UpstreamFn')) continue;
      // Normalize fully-qualified paths: &[crate::domain::param_types::Point2D] → &[Point2D]
      let normalizedType = ptype;
      const sliceMatch = ptype.match(/^&\[(.+)\]$/);
      if (sliceMatch) {
        const inner = sliceMatch[1].split('::').pop();
        normalizedType = `&[${inner}]`;
      } else if (ptype.includes('::')) {
        // e.g., crate::domain::param_types::Point2D → Point2D
        const refPrefix = ptype.startsWith('&') ? '&' : '';
        const base = ptype.replace(/^&/, '');
        normalizedType = refPrefix + base.split('::').pop();
      }
      params.push({ name, type: normalizedType });
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

// ─── WIT generation ──────────────────────────────────────────────────────

/**
 * Resolve a Rust type to a WIT type string.
 * Primitives map directly. ConfigParams struct types map to their WIT record name.
 * Unknown types cause an error.
 */
function resolveWitType(rustType, configStructs, context) {
  // Check primitives first
  if (PRIMITIVE_TYPE_MAP[rustType]) return PRIMITIVE_TYPE_MAP[rustType];

  // Check if it's a known ConfigParams struct (nested type like ColorRgba)
  if (configStructs.has(rustType)) return configStructs.get(rustType).witRecordName;

  // Config struct reference (e.g., &BlurParams) — not a WIT field, handled separately
  if (rustType.startsWith('&') && rustType.endsWith('Params')) return null;

  // Unknown type — error
  console.error(`ERROR: Unknown WIT type mapping for '${rustType}' in ${context}`);
  console.error(`  Field type must be a primitive (f32, u32, bool, ...) or a ConfigParams struct.`);
  console.error(`  If this is a new type, add #[derive(ConfigParams)] to it.`);
  process.exit(1);
}

/**
 * Generate WIT record definitions for all ConfigParams structs that are
 * used as filter config params.
 */
function generateWitRecords(configStructs, usedStructNames) {
  const output = [];
  const emitted = new Set();

  // Recursively emit records for nested types first
  function emitRecord(structName) {
    if (emitted.has(structName)) return;
    const info = configStructs.get(structName);
    if (!info) return;

    // Emit dependencies first
    for (const field of info.fields) {
      if (configStructs.has(field.type) && !emitted.has(field.type)) {
        emitRecord(field.type);
      }
    }

    const fieldLines = info.fields.map(f => {
      const witType = resolveWitType(f.type, configStructs, `${structName}.${f.name}`);
      return `        ${toWitName(f.name)}: ${witType},`;
    });
    output.push(`    record ${info.witRecordName} {`);
    output.push(...fieldLines);
    output.push(`    }\n`);
    emitted.add(structName);
  }

  for (const name of usedStructNames) {
    emitRecord(name);
  }
  return output.join('\n');
}

function generateFiltersWit(filters, configStructs) {
  // Collect all config struct names used by filters
  const usedStructs = new Set();
  for (const f of filters) {
    for (const p of f.params) {
      if (p.type.startsWith('&') && p.type.endsWith('Params')) {
        const structName = p.type.slice(1); // strip leading &
        usedStructs.add(structName);
        // Also collect nested struct types
        const info = configStructs.get(structName);
        if (info) {
          for (const field of info.fields) {
            if (configStructs.has(field.type)) usedStructs.add(field.type);
          }
        }
      }
    }
  }

  // Emit records
  const records = generateWitRecords(configStructs, usedStructs);

  // Emit functions
  const funcs = filters.map(f => {
    const configParam = f.params.find(p => p.type.startsWith('&') && p.type.endsWith('Params'));
    const extraParams = f.params.filter(p => !(p.type.startsWith('&') && p.type.endsWith('Params')));

    const parts = ['pixels: buffer', 'info: image-info'];
    for (const ep of extraParams) {
      const witType = resolveWitType(ep.type, configStructs, `${f.name}.${ep.name}`);
      if (witType) parts.push(`${toWitName(ep.name)}: ${witType}`);
    }
    if (configParam) {
      const structName = configParam.type.slice(1);
      const info = configStructs.get(structName);
      if (info) parts.push(`config: ${info.witRecordName}`);
    }

    const doc = `    /// ${f.reference || f.name + ' filter'}`;
    const sig = `    ${toWitName(f.name)}: func(${parts.join(', ')}) -> result<buffer, rasmcore-error>;`;
    return `${doc}\n${sig}\n`;
  }).join('');

  return records + '\n' + funcs;
}

/**
 * Pipeline WIT mirrors the filters interface — config records are passed as-is.
 * ConfigParams = WIT record. Same records, same types.
 */
function generatePipelineWit(filters, configStructs) {
  return filters.map(f => {
    const configParam = f.params.find(p => p.type.startsWith('&') && p.type.endsWith('Params'));
    const extraParams = f.params.filter(p => !(p.type.startsWith('&') && p.type.endsWith('Params')));

    const parts = ['source: node-id'];
    for (const ep of extraParams) {
      const witType = resolveWitType(ep.type, configStructs, `${f.name}.${ep.name}`);
      if (witType) parts.push(`${toWitName(ep.name)}: ${witType}`);
    }
    if (configParam) {
      const structName = configParam.type.slice(1);
      const info = configStructs.get(structName);
      if (info) parts.push(`config: ${info.witRecordName}`);
    }

    return `        ${toWitName(f.name)}: func(${parts.join(', ')}) -> result<node-id, rasmcore-error>;`;
  }).join('\n');
}

// ─── Main ────────────────────────────────────────────────────────────────

const filtersSource = fs.readFileSync(FILTERS_RS, 'utf8');
const paramTypesSource = fs.existsSync(PARAM_TYPES_RS) ? fs.readFileSync(PARAM_TYPES_RS, 'utf8') : '';

// Parse all ConfigParams struct definitions
const configStructs = parseConfigParamsStructs([filtersSource, paramTypesSource]);
console.log(`Parsed ${configStructs.size} ConfigParams structs`);

// Parse all registered filters
const filters = parseRegisteredFilters(filtersSource);
console.log(`Parsed ${filters.length} registered filters from filters.rs`);

if (fs.existsSync(FILTERS_TMPL)) {
  const tmpl = fs.readFileSync(FILTERS_TMPL, 'utf8');
  const content = generateFiltersWit(filters, configStructs);
  fs.writeFileSync(FILTERS_WIT, tmpl.replace('{{GENERATED_FILTERS}}', content));
  console.log(`Generated: wit/image/filters.wit (${filters.length} filter functions)`);
}

// pipeline.wit is now generated by build.rs (handles transforms + graph description).
// Run `cargo build -p rasmcore-image` to regenerate pipeline.wit from the template.
console.log(`Skipped: wit/image/pipeline.wit (generated by build.rs, not this script)`);
