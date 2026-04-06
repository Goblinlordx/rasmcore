#!/usr/bin/env node
/**
 * Generate fluent TypeScript SDK from V2 WASM operation registry.
 *
 * Instantiates the V2 WASM component, calls listOperations(), and generates
 * a typed Pipeline class with chainable methods for each filter and typed
 * write methods for each encoder.
 *
 * Usage: node scripts/generate-v2-fluent-sdk.mjs
 *
 * Requires: ./scripts/generate-v2-sdk.sh to have been run first
 *           npm install @bytecodealliance/preview2-shim
 */

import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');
const outDir = join(projectRoot, 'sdk', 'typescript', 'v2-fluent');

// Get operation metadata from native Rust binary (not WASM — inventory links natively)
const registryJson = join('/tmp', 'v2_registry.json');
console.log('Building registry dump...');
execSync('cargo run --bin dump_registry -p rasmcore-v2-wasm 2>/dev/null > ' + registryJson, { cwd: projectRoot });
const registry = JSON.parse(readFileSync(registryJson, 'utf8'));

const filters = registry.filters;
const encoders = registry.encoders;
const decoders = registry.decoders;

console.log(`Discovered: ${filters.length} filters, ${encoders.length} encoders, ${decoders.length} decoders`);

// ─── Generate TypeScript ────────────────────────────────────────────────────

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

function snakeToPascal(s) {
  const camel = snakeToCamel(s);
  return camel.charAt(0).toUpperCase() + camel.slice(1);
}

function paramTypeToTs(paramType) {
  switch (paramType) {
    case 'f32': case 'f64': case 'f32-val': case 'f64-val': return 'number';
    case 'u32': case 'i32': case 'u32-val': case 'i32-val': return 'number';
    case 'bool': case 'bool-val': return 'boolean';
    case 'string': case 'string-val': return 'string';
    default: return 'number';
  }
}

function generateFilterInterface(op) {
  if (op.params.length === 0) return null;
  const name = `${snakeToPascal(op.name)}Config`;
  const fields = op.params.map(p => {
    const vt = p.valueType || p.type;
    const def = p.defaultVal ?? p.default;
    const optional = def != null ? '?' : '';
    return `  ${snakeToCamel(p.name)}${optional}: ${paramTypeToTs(vt)};`;
  });
  return `export interface ${name} {\n${fields.join('\n')}\n}`;
}

function generateSerializeParams(op) {
  if (op.params.length === 0) return '  return new Uint8Array(0);';
  const lines = op.params.map(p => {
    const camel = snakeToCamel(p.name);
    const name = p.name;
    const vt = p.valueType || p.type;
    const def = p.defaultVal ?? p.default;
    const defaultVal = def != null ? ` ?? ${def}` : '';
    switch (vt) {
      case 'f32': case 'f64': case 'f32-val': case 'f64-val':
        return `  pushF32(buf, '${name}', config.${camel}${defaultVal});`;
      case 'u32': case 'i32': case 'u32-val': case 'i32-val':
        return `  pushU32(buf, '${name}', config.${camel}${defaultVal});`;
      case 'bool': case 'bool-val':
        return `  pushBool(buf, '${name}', config.${camel}${defaultVal ? ` ?? ${def !== 0}` : ''});`;
      default:
        return `  pushF32(buf, '${name}', config.${camel}${defaultVal});`;
    }
  });
  return `  const buf: number[] = [];\n${lines.join('\n')}\n  return new Uint8Array(buf);`;
}

// ─── Build output ───────────────────────────────────────────────────────────

let ts = `// Auto-generated V2 fluent SDK — do not edit.
// Generated from ${filters.length} filters, ${encoders.length} encoders, ${decoders.length} decoders.
// Regenerate: node scripts/generate-v2-fluent-sdk.mjs

import type { ImagePipelineV2 as RawPipeline } from '../v2-generated/interfaces/rasmcore-v2-image-pipeline-v2.js';

// ─── Param serialization helpers ────────────────────────────────────────────

function pushF32(buf: number[], name: string, value: number) {
  buf.push(name.length);
  for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
  buf.push(0); // f32 type
  const ab = new ArrayBuffer(4);
  new DataView(ab).setFloat32(0, value, true);
  buf.push(...new Uint8Array(ab));
}

function pushU32(buf: number[], name: string, value: number) {
  buf.push(name.length);
  for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
  buf.push(1); // u32 type
  const ab = new ArrayBuffer(4);
  new DataView(ab).setUint32(0, value, true);
  buf.push(...new Uint8Array(ab));
}

function pushBool(buf: number[], name: string, value: boolean) {
  buf.push(name.length);
  for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
  buf.push(2); // bool type
  buf.push(value ? 1 : 0);
}

// ─── Config interfaces ──────────────────────────────────────────────────────

`;

// Generate config interfaces for filters
for (const op of filters) {
  const iface = generateFilterInterface(op);
  if (iface) ts += iface + '\n\n';
}

// ─── Pipeline class ─────────────────────────────────────────────────────────

ts += `// ─── Pipeline class ──────────────────────────────────────────────────────────

export interface ReadConfig {
  hint?: string;
}

/**
 * V2 image processing pipeline with fluent API.
 *
 * Usage:
 *   const result = Pipeline.open(imageBytes)
 *     .brightness({ amount: 0.5 })
 *     .blur({ radius: 3 })
 *     .writePng();
 */
export class Pipeline {
  private _pipe: RawPipeline;
  private _node: number;

  private constructor(pipe: RawPipeline, node: number) {
    this._pipe = pipe;
    this._node = node;
  }

  /** Create a pipeline from raw image bytes (auto-loads WASM module). */
  static open(data: Uint8Array, config?: ReadConfig): Pipeline {
    const { pipelineV2 } = require('../v2-generated/rasmcore-v2-image.js');
    const pipe = new pipelineV2.ImagePipelineV2();
    const readConfig = config ? { formatHint: config.hint } : undefined;
    const node = pipe.read(data, readConfig);
    return new Pipeline(pipe, node);
  }

  /** Create a pipeline from a pre-loaded pipeline class (for web workers). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static fromRaw(PipelineClass: any, data: Uint8Array, config?: ReadConfig): Pipeline {
    const pipe = new PipelineClass();
    const readConfig = config ? { formatHint: config.hint } : undefined;
    const node = pipe.read(data, readConfig);
    return new Pipeline(pipe, node);
  }

  /** Get image dimensions and color space. */
  get info() {
    return this._pipe.nodeInfo(this._node);
  }

  /** Get raw f32 RGBA pixel data. */
  render(): Float32Array {
    return this._pipe.render(this._node);
  }

  /** Get the current sink node ID (for GPU plan extraction). */
  get sinkNode(): number {
    return this._node;
  }

  // ─── Refs (DAG branch points) ─────────────────────────────────────────

  /** Mark the current node as a named branch point.
   *  Downstream consumers can fork from this ref via branch(). */
  ref(name: string): Pipeline {
    if (typeof this._pipe.setRef === 'function') {
      this._pipe.setRef(name, this._node);
    }
    return this;
  }

  /** Fork from a named ref — returns a new Pipeline cursor at that node.
   *  Both this pipeline and the branched one share the same underlying graph. */
  branch(name: string): Pipeline {
    let nodeId: number | undefined;
    if (typeof this._pipe.getRef === 'function') {
      nodeId = this._pipe.getRef(name);
    }
    if (nodeId == null) {
      throw new Error(\`Unknown ref: \${name}\`);
    }
    return new Pipeline(this._pipe, nodeId);
  }

  // ─── Multi-output display targets ─────────────────────────────────────

  /** Internal map of display target name → node ID. */
  private _displays: Map<string, number> | null = null;

  /** Register a display target at the current node.
   *  Multiple displays can be attached to the same or different nodes.
   *  Call execute() to render all targets in one GPU submit. */
  addDisplay(name: string): Pipeline {
    if (!this._displays) this._displays = new Map();
    this._displays.set(name, this._node);
    return this;
  }

  /** Extract a multi-output GPU plan for all registered display targets.
   *  Returns the plan for host-side execution via GpuHandler.executeMulti(). */
  renderMultiGpuPlan(): any {
    if (!this._displays || this._displays.size === 0) return null;
    if (typeof this._pipe.renderMultiGpuPlan !== 'function') return null;
    const targets: [string, number][] = [];
    for (const [name, nodeId] of this._displays) {
      targets.push([name, nodeId]);
    }
    return this._pipe.renderMultiGpuPlan(targets);
  }

  // ─── Filter methods (generated) ─────────────────────────────────────────

`;

// Generate filter methods
for (const op of filters) {
  const method = snakeToCamel(op.name);
  const configType = op.params.length > 0 ? `${snakeToPascal(op.name)}Config` : null;
  const paramSig = configType ? `config: ${configType}` : '';
  const serializeBody = generateSerializeParams(op);

  ts += `  /** ${op.displayName} */\n`;
  ts += `  ${method}(${paramSig}): Pipeline {\n`;
  if (configType) {
    ts += `    const serialize = (config: ${configType}): Uint8Array => {\n`;
    ts += `  ${serializeBody}\n`;
    ts += `    };\n`;
    ts += `    const node = this._pipe.applyFilter(this._node, '${op.name}', serialize(config));\n`;
  } else {
    ts += `    const node = this._pipe.applyFilter(this._node, '${op.name}', new Uint8Array(0));\n`;
  }
  ts += `    return new Pipeline(this._pipe, node);\n`;
  ts += `  }\n\n`;
}

// LMT method — load .cube/.clf files as pipeline nodes
ts += `  // ─── LMT ────────────────────────────────────────────────────────────────

  /** Apply a Look Modification Transform from raw file data (.cube or .clf).
   *  Format is auto-detected from content. */
  useLmt(data: Uint8Array): Pipeline {
    const node = this._pipe.useLmt(this._node, data);
    return new Pipeline(this._pipe, node);
  }

`;

// Generate write methods
ts += `  // ─── Write methods ──────────────────────────────────────────────────────

  /** Encode to format by name (generic). */
  write(format: string, quality?: number): Uint8Array {
    return this._pipe.write(this._node, format, quality);
  }

`;

// Generate typed write methods for known encoders
const encoderFormats = [
  { name: 'png', method: 'writePng', display: 'PNG' },
  { name: 'jpeg', method: 'writeJpeg', display: 'JPEG' },
  { name: 'webp', method: 'writeWebp', display: 'WebP' },
  { name: 'gif', method: 'writeGif', display: 'GIF' },
  { name: 'bmp', method: 'writeBmp', display: 'BMP' },
  { name: 'tiff', method: 'writeTiff', display: 'TIFF' },
  { name: 'qoi', method: 'writeQoi', display: 'QOI' },
  { name: 'ico', method: 'writeIco', display: 'ICO' },
  { name: 'tga', method: 'writeTga', display: 'TGA' },
  { name: 'pnm', method: 'writePnm', display: 'PNM' },
  { name: 'exr', method: 'writeExr', display: 'EXR' },
  { name: 'hdr', method: 'writeHdr', display: 'HDR' },
];

for (const enc of encoderFormats) {
  ts += `  /** Encode as ${enc.display}. */\n`;
  ts += `  ${enc.method}(quality?: number): Uint8Array {\n`;
  ts += `    return this._pipe.write(this._node, '${enc.name}', quality);\n`;
  ts += `  }\n\n`;
}

ts += `  // ─── Discovery ──────────────────────────────────────────────────────────

  /** List all available operations (auto-loads WASM module). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static listOperations(): any[] {
    const { pipelineV2 } = require('../v2-generated/rasmcore-v2-image.js');
    const pipe = new pipelineV2.ImagePipelineV2();
    return pipe.listOperations();
  }

  /** List all available operations from a pre-loaded pipeline class. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static listOperationsFromRaw(PipelineClass: any): any[] {
    const pipe = new PipelineClass();
    return pipe.listOperations();
  }
}
`;

// ─── Extension Mechanism ─────────────────────────────────────────────────────
// Scan sdk/v2/extensions/ for hand-written .ts files.
// Each file is appended after the generated Pipeline class, giving it access
// to Pipeline.prototype for adding methods like writeRenderTarget().

const extensionsDir = join(projectRoot, 'sdk', 'v2', 'extensions');
if (existsSync(extensionsDir)) {
  const { readdirSync } = await import('fs');
  const extFiles = readdirSync(extensionsDir)
    .filter(f => f.endsWith('.ts'))
    .sort();
  for (const extFile of extFiles) {
    const extContent = readFileSync(join(extensionsDir, extFile), 'utf8');
    ts += `\n// ─── Extension: ${extFile} ────────────────────────────────────────\n`;
    ts += extContent;
    ts += '\n';
    console.log(`  Extension merged: ${extFile}`);
  }
}

// ─── Write output ───────────────────────────────────────────────────────────

mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'index.ts'), ts);

console.log(`\nGenerated: ${join(outDir, 'index.ts')}`);
console.log(`  ${filters.length} filter methods`);
console.log(`  ${encoderFormats.length} typed write methods`);
console.log(`  ${filters.filter(f => f.params.length > 0).length} config interfaces`);
