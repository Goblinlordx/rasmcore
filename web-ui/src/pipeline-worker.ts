// @ts-nocheck
// eslint-disable
/**
 * rasmcore pipeline worker — runs all WASM processing off the main thread.
 *
 * Protocol:
 *   Main → Worker: { type: 'init' | 'load' | 'process' | 'export' }
 *   Worker → Main: { type: 'ready' | 'loaded' | 'result' | 'exported' | 'error' }
 *
 * Supports both full-res and thumbnail modes via the `mode` field.
 * Uses Transferable ArrayBuffer for zero-copy image data transfer.
 */

let Pipeline = null;
let imageBytes = null;
let thumbBytes = null;
const THUMB_MAX = 256;
let formatMimeMap = {}; // Populated from SDK: { jpeg: "image/jpeg", ... }

// ─── SDK Loading ────────────────────────────────────────────────────────────

async function initSDK() {
  try {
    const sdk = await import('../sdk/rasmcore-image.js');
    Pipeline = sdk.pipeline.ImagePipeline;

    // Build MIME map from backend format metadata
    try {
      const infos = sdk.encoder.allFormatInfo();
      for (const fi of infos) {
        formatMimeMap[fi.name] = fi.mimeType;
      }
    } catch (_) {
      // Fallback: SDK may not have allFormatInfo yet
    }

    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `SDK init failed: ${e.message}` });
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function expandArgs(params, paramValues) {
  const args = [];
  for (const p of params) {
    if (p.type === 'color') {
      const [r, g, b] = hexToRgb(paramValues[p.name] || '#808080');
      args.push(r, g, b);
    } else {
      args.push(paramValues[p.name]);
    }
  }
  return args;
}

// Pipeline methods that take positional args (not a config record).
// Everything else uses a config record: pipe.blur(node, { radius: 5.0 })
const POSITIONAL_ARG_OPS = new Set([
  'resize',
  'crop',
  'rotate',
  'flip',
  'grayscale',
  'convertFormat',
  'convolve',
  'displacementMap',
  'curvesRed',
  'curvesGreen',
  'curvesBlue',
  'curvesMaster',
  'hueVsSat',
  'hueVsLum',
  'lumVsSat',
  'satVsSat',
  'applyCubeLut',
  'applyHaldLut',
  'gradientMap',
]);

// Build a WIT config record from param metadata + values.
// Color params expand to { r, g, b } fields in the config.
// Seed params (rc.seed hint, u64 in WIT) must be converted to BigInt for jco.
function buildConfig(params, paramValues) {
  const config = {};
  for (const p of params) {
    if (p.type === 'color') {
      const [r, g, b] = hexToRgb(paramValues[p.name] || '#808080');
      config[p.name] = { r, g, b };
    } else if (p.hint === 'rc.seed') {
      // u64 in WIT → BigInt in JS (jco requirement)
      config[p.name] = BigInt(Math.floor(paramValues[p.name] || 0));
    } else {
      config[p.name] = paramValues[p.name];
    }
  }
  return config;
}

function applyStep(pipe, node, step, mode) {
  const name = step.name;

  // Special thumbnail handling for resize/crop
  if (name === 'resize' && mode === 'thumb') {
    const args = expandArgs(step.params, step.paramValues);
    const scale = Math.min(THUMB_MAX / args[0], THUMB_MAX / args[1], 1);
    return pipe.resize(
      node,
      Math.max(1, Math.round(args[0] * scale)),
      Math.max(1, Math.round(args[1] * scale)),
      args[2],
    );
  }
  if (name === 'crop' && mode === 'thumb') {
    const args = expandArgs(step.params, step.paramValues);
    const info = pipe.nodeInfo(node);
    return pipe.crop(node, 0, 0, Math.min(args[2], info.width), Math.min(args[3], info.height));
  }

  // Positional-arg ops: spread flat args
  if (POSITIONAL_ARG_OPS.has(name)) {
    const args = expandArgs(step.params, step.paramValues);
    return pipe[name](node, ...args);
  }

  // No-param ops (equalize, invert, etc.)
  if (!step.params || step.params.length === 0) {
    return pipe[name](node);
  }

  // All other ops: build config record
  const config = buildConfig(step.params, step.paramValues);
  return pipe[name](node, config);
}

// ─── Image Loading ──────────────────────────────────────────────────────────

function loadImage(bytes) {
  imageBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = new Pipeline();
    const src = pipe.read(imageBytes);
    info = pipe.nodeInfo(src);

    // Create thumbnail
    const scale = Math.min(THUMB_MAX / info.width, THUMB_MAX / info.height, 1);
    if (scale < 1) {
      const tw = Math.round(info.width * scale);
      const th = Math.round(info.height * scale);
      const resized = pipe.resize(src, tw, th, 'bilinear');
      thumbBytes = pipe.writePng(resized, {}, undefined);
    } else {
      thumbBytes = imageBytes;
    }
  } catch (e) {
    thumbBytes = imageBytes;
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Pipeline Processing ────────────────────────────────────────────────────

function processChain(chain, mode) {
  const source = mode === 'thumb' ? thumbBytes : imageBytes;
  if (!source) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  const timings = [];

  try {
    const pipe = new Pipeline();
    let node = pipe.read(source);

    for (const step of chain) {
      const t = performance.now();
      node = applyStep(pipe, node, step, mode);
      timings.push({ name: step.name, ms: Math.round(performance.now() - t) });
    }

    const output = pipe.writePng(node, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);

    self.postMessage({ type: 'result', png: buf, timings, totalMs, mode }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
  }
}

// ─── Export ─────────────────────────────────────────────────────────────────

function exportImage(chain, format, quality) {
  if (!imageBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  try {
    const pipe = new Pipeline();
    let node = pipe.read(imageBytes);

    for (const step of chain) {
      node = applyStep(pipe, node, step, 'full');
    }

    // Use generic write() — MIME resolved from backend metadata (no hardcoded map)
    const output = pipe.write(node, format, quality > 0 ? quality : undefined, undefined);
    const mime = formatMimeMap[format] || 'application/octet-stream';

    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'exported', data: buf, mime, format }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
  }
}

// ─── Multi-Layer Composite ───────────────────────────────────────────────────

function compositeLayers(layerDefs) {
  if (!layerDefs || layerDefs.length === 0) {
    self.postMessage({ type: 'error', message: 'No layers to composite' });
    return;
  }

  const t0 = performance.now();
  try {
    const pipe = new Pipeline();

    // Process each layer: load → apply chain → get node
    let resultNode = null;
    for (let i = 0; i < layerDefs.length; i++) {
      const layer = layerDefs[i];
      const bytes = new Uint8Array(layer.imageBytes);
      let node = pipe.read(bytes);

      // Apply per-layer chain
      for (const step of layer.chain) {
        node = applyStep(pipe, node, step, 'full');
      }

      if (i === 0) {
        resultNode = node;
      } else {
        // Composite this layer onto the result
        // Map blend mode string to WIT enum value
        const mode = layer.blendMode || undefined;
        resultNode = pipe.composite(node, resultNode, layer.x || 0, layer.y || 0, mode);
      }
    }

    const output = pipe.writePng(resultNode, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, timings: [], totalMs, mode: 'full' }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: `Composite error: ${e.message}` });
  }
}

// ─── Message Handler ────────────────────────────────────────────────────────

self.onmessage = (e) => {
  const { type } = e.data;
  switch (type) {
    case 'init':
      initSDK();
      break;
    case 'load':
      loadImage(e.data.imageBytes);
      break;
    case 'process':
      processChain(e.data.chain, e.data.mode);
      break;
    case 'export':
      exportImage(e.data.chain, e.data.format, e.data.quality);
      break;
    case 'composite':
      compositeLayers(e.data.layers);
      break;
  }
};
