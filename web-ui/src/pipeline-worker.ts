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

import { GpuHandler, createGpuExecuteImport } from './gpu-handler';
import type { GpuOp, GpuResult } from './gpu-handler';

let Pipeline = null;
let LayerCache = null;
let layerCache = null; // Shared cross-pipeline content-addressed cache
let imageBytes = null;
let thumbBytes = null;
let cachedPipe = null; // Reused pipeline instance for graph caching
let cachedSourceNode = null; // Source node from last load
const THUMB_MAX = 256;
let formatMimeMap = {}; // Populated from SDK: { jpeg: "image/jpeg", ... }

// GPU handler — initialized during SDK init, null if WebGPU unavailable
let gpuExecute: ((ops: GpuOp[], input: Uint8Array, width: number, height: number) => Promise<GpuResult>) | null = null;
let gpuAvailable = false;

// ─── SDK Loading ────────────────────────────────────────────────────────────

async function initSDK() {
  try {
    const sdk = await import('../sdk/rasmcore-image.js');
    Pipeline = sdk.pipeline.ImagePipeline;
    LayerCache = sdk.pipeline.LayerCache;

    // Create shared layer cache for cross-pipeline content-addressed caching
    if (LayerCache) {
      layerCache = new LayerCache(256); // 256 MB capacity
    }

    // Build MIME map from backend format metadata
    try {
      const infos = sdk.encoder.allFormatInfo();
      for (const fi of infos) {
        formatMimeMap[fi.name] = fi.mimeType;
      }
    } catch (_) {
      // Fallback: SDK may not have allFormatInfo yet
    }

    // Initialize GPU handler if WebGPU is available
    gpuAvailable = GpuHandler.isAvailable();
    if (gpuAvailable) {
      gpuExecute = createGpuExecuteImport();
    }

    self.postMessage({ type: 'ready', gpu: gpuAvailable });
  } catch (e) {
    self.postMessage({ type: 'error', message: `SDK init failed: ${e.message}` });
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function attachCache(pipe) {
  if (layerCache && pipe.setLayerCache) {
    pipe.setLayerCache(layerCache);
  }
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// Build a WIT config record from param metadata + values.
// Color params expand to { r, g, b } fields in the config.
// u64/i64 params must be converted to BigInt for jco.
function buildConfig(params, paramValues) {
  const config = {};
  for (const p of params) {
    if (p.type === 'color') {
      const [r, g, b] = hexToRgb(paramValues[p.name] || '#808080');
      config[p.name] = { r, g, b };
    } else if (p.witType === 'u64' || p.witType === 'i64') {
      config[p.name] = BigInt(Math.floor(paramValues[p.name] || 0));
    } else {
      config[p.name] = paramValues[p.name];
    }
  }
  return config;
}

function applyStep(pipe, node, step, mode) {
  const name = step.name;

  // Special thumbnail handling for resize/crop — scale down for preview
  if (name === 'resize' && mode === 'thumb') {
    const pv = step.paramValues;
    const w = pv.width || 256,
      h = pv.height || 256;
    const filter = pv.filter || 'lanczos3';
    const scale = Math.min(THUMB_MAX / w, THUMB_MAX / h, 1);
    return pipe.resize(node, {
      width: Math.max(1, Math.round(w * scale)),
      height: Math.max(1, Math.round(h * scale)),
      filter,
    });
  }
  if (name === 'crop' && mode === 'thumb') {
    const pv = step.paramValues;
    const info = pipe.nodeInfo(node);
    return pipe.crop(node, {
      x: 0,
      y: 0,
      width: Math.min(pv.width || 256, info.width),
      height: Math.min(pv.height || 256, info.height),
    });
  }

  // No-param ops (equalize, invert, etc.)
  if (!step.params || step.params.length === 0) {
    return pipe[name](node);
  }

  // All ops: build config record
  const config = buildConfig(step.params, step.paramValues);
  return pipe[name](node, config);
}

// ─── Image Loading ──────────────────────────────────────────────────────────

function loadImage(bytes) {
  imageBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    // Create a persistent pipeline for caching across process calls
    cachedPipe = new Pipeline();
    attachCache(cachedPipe);
    cachedSourceNode = cachedPipe.read(imageBytes);
    info = cachedPipe.nodeInfo(cachedSourceNode);

    const pipe = cachedPipe;
    const src = cachedSourceNode;

    // Create thumbnail
    const scale = Math.min(THUMB_MAX / info.width, THUMB_MAX / info.height, 1);
    if (scale < 1) {
      const tw = Math.round(info.width * scale);
      const th = Math.round(info.height * scale);
      const resized = pipe.resize(src, { width: tw, height: th, filter: 'bilinear' });
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
  if (!imageBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  const timings = [];

  try {
    // Reuse cached pipeline for graph caching, or create fresh one
    const pipe = cachedPipe || new Pipeline();
    const node = cachedSourceNode || pipe.read(imageBytes);
    let current = node;

    for (const step of chain) {
      const t = performance.now();
      current = applyStep(pipe, current, step, mode);
      timings.push({ name: step.name, ms: Math.round(performance.now() - t) });
    }

    const output = pipe.writePng(current, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);

    if (layerCache) {
      const s = layerCache.stats();
      console.log(`[pipeline-worker] ${totalMs}ms | cache: ${s.hits} hits, ${s.misses} misses, ${s.entries} entries`);
    }

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
    const pipe = cachedPipe || new Pipeline();
    let node = cachedSourceNode || pipe.read(imageBytes);

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
    attachCache(pipe);

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
