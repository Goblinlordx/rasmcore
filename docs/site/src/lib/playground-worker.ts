/**
 * Playground Web Worker — runs WASM pipeline off the main thread.
 *
 * The jco-generated SDK uses bare specifiers (@bytecodealliance/preview2-shim/*).
 * Workers don't inherit the page's import map. We patch globalThis to provide
 * the shim modules before the SDK's static imports execute.
 *
 * Strategy: load the SDK via dynamic import with webpackIgnore so it bypasses
 * the bundler and loads from the public folder at runtime. Before that, we
 * pre-register the preview2-shim modules by importing them from absolute paths.
 */

// @ts-nocheck

let PipelineClass = null;
let sourceClass = null;
let gpuHandler = null;
const sourceCache = new Map();

let busy = false;
let queued = null;

async function initWasm() {
  try {
    // The SDK is in the public folder — load at runtime, not through bundler
    const sdk = await import(/* webpackIgnore: true */ '/sdk/v2/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    sourceClass = sdk.pipelineV2.Source;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `WASM init failed: ${e.message}` });
  }
}

async function initGpu() {
  if (gpuHandler) return true;
  try {
    const mod = await import(/* webpackIgnore: true */ '/sdk/v2/lib/gpu-handler.js');
    const GpuHandlerV2 = mod.GpuHandlerV2;
    if (GpuHandlerV2.isAvailable()) {
      gpuHandler = new GpuHandlerV2();
      return true;
    }
  } catch { /* GPU not available */ }
  return false;
}

async function setDisplay(canvas, hdr) {
  if (!gpuHandler) {
    const ok = await initGpu();
    if (!ok) return;
  }
  try {
    await gpuHandler.setDisplayCanvas(canvas, hdr);
  } catch (e) {
    console.warn('[playground-worker] GPU display failed:', e?.message);
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

    let gpuRendered = false;
    if (gpuHandler?.hasDisplay && typeof pipe.renderGpuPlan === 'function') {
      try {
        const plan = pipe.renderGpuPlan(filterId);
        if (plan) {
          gpuHandler.updateViewport(0, 0, 1.0, info.width, info.height, info.width, info.height, 0);
          const ops = plan.shaders.map(s => ({
            source: s.source, entryPoint: s.entryPoint,
            workgroupX: s.workgroupX, workgroupY: s.workgroupY, workgroupZ: s.workgroupZ,
            params: new Uint8Array(s.params),
            extraBuffers: s.extraBuffers.map(b => new Uint8Array(b)),
          }));
          await gpuHandler.prepare(ops);
          const err = await gpuHandler.executeAndDisplay(ops, new Float32Array(plan.inputPixels), plan.width, plan.height);
          if (!err) gpuRendered = true;
        }
      } catch { /* GPU failed */ }
    }

    if (!gpuRendered) {
      if (gpuHandler?.hasDisplay) {
        const pixels = pipe.render(filterId);
        const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
        gpuHandler.updateViewport(0, 0, 1.0, info.width, info.height, info.width, info.height, 0);
        gpuHandler.displayFromCpu(f32, info.width, info.height);
        gpuRendered = true;
      } else {
        const pixels = pipe.render(filterId);
        const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
        const u8 = new Uint8ClampedArray(f32.length);
        for (let i = 0; i < f32.length; i++) u8[i] = f32[i] * 255;
        self.postMessage(
          { type: 'result', imageData: u8.buffer, width: info.width, height: info.height, totalMs: Math.round(performance.now() - t0) },
          [u8.buffer],
        );
        return;
      }
    }

    const totalMs = Math.round(performance.now() - t0);
    self.postMessage({ type: 'displayed', width: info.width, height: info.height, totalMs });
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
    case 'set-display': setDisplay(e.data.canvas, e.data.hdr ?? false); break;
    case 'render':
      if (busy) { queued = e.data; }
      else { processRender(e.data); }
      break;
  }
};
