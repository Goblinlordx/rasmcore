/**
 * Playground Web Worker — runs WASM pipeline off the main thread.
 *
 * CPU-only: renders f32 pixels in WASM, quantizes to u8, posts ImageData
 * back to main thread. No GPU display — keeps the worker simple and avoids
 * OffscreenCanvas/WebGPU complexity in the docs site.
 */

// @ts-nocheck

let PipelineClass = null;
let sourceClass = null;
const sourceCache = new Map();

let busy = false;
let queued = null;

async function initWasm() {
  try {
    const sdk = await import(/* webpackIgnore: true */ '/sdk/wasm/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    sourceClass = sdk.pipelineV2.Source;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `WASM init failed: ${e.message}` });
  }
}

function getOrCreateSource(imageBytes, cacheKey) {
  if (sourceCache.has(cacheKey)) return sourceCache.get(cacheKey);
  if (sourceClass) {
    const source = new sourceClass(imageBytes, undefined);
    sourceCache.set(cacheKey, source);
    return source;
  }
  return null;
}

function serializeParams(params, paramTypes) {
  const entries = Object.entries(params);
  if (entries.length === 0) return new Uint8Array(0);
  const buf = [];
  for (const [name, value] of entries) {
    buf.push(name.length);
    for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
    const ptype = paramTypes?.[name];
    const isInt = ptype === 'U32' || ptype === 'I32' || ptype === 'u32' || ptype === 'i32';
    const isBool = typeof value === 'boolean' || ptype === 'Bool' || ptype === 'bool';
    if (isBool) {
      buf.push(2);
      buf.push(value ? 1 : 0);
    } else if (isInt) {
      buf.push(1);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setUint32(0, Math.round(value), true);
      buf.push(...new Uint8Array(ab));
    } else {
      buf.push(0);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, value, true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}

async function doRender(data) {
  if (!PipelineClass) {
    self.postMessage({ type: 'error', message: 'WASM not loaded' });
    return;
  }

  const t0 = performance.now();
  const { imageBytes, filterName, params, cacheKey, paramTypes } = data;

  try {
    const pipe = new PipelineClass();

    const source = getOrCreateSource(new Uint8Array(imageBytes), cacheKey);
    let nodeId;
    if (source && typeof pipe.readSource === 'function') {
      nodeId = pipe.readSource(source);
    } else {
      nodeId = pipe.read(new Uint8Array(imageBytes), undefined);
    }

    const paramBuf = serializeParams(params, paramTypes);
    const filterId = pipe.applyFilter(nodeId, filterName, paramBuf);
    const info = pipe.nodeInfo(filterId);

    const pixels = pipe.render(filterId);
    const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
    const len = f32.length;
    const u8 = new Uint8ClampedArray(len);
    // render() returns sRGB-space f32 (OT applied by pipeline) — quantize directly.
    for (let i = 0; i < len; i++) u8[i] = f32[i] * 255 + 0.5;

    const totalMs = Math.round(performance.now() - t0);
    console.log(`[playground-worker] ${filterName}: ${totalMs}ms ${info.width}x${info.height}`);

    self.postMessage(
      { type: 'result', imageData: u8.buffer, width: info.width, height: info.height, totalMs },
      [u8.buffer],
    );
  } catch (e) {
    const msg = e?.payload ? JSON.stringify(e.payload, null, 2) : (e?.message || String(e));
    self.postMessage({ type: 'error', message: msg });
  }
}

async function processRender(data) {
  busy = true;
  await doRender(data);
  busy = false;
  if (queued) {
    const next = queued;
    queued = null;
    processRender(next);
  }
}

self.onmessage = (e) => {
  switch (e.data.type) {
    case 'init': initWasm(); break;
    case 'render':
      if (busy) { queued = e.data; }
      else { processRender(e.data); }
      break;
  }
};
