// @ts-nocheck
/**
 * V2 pipeline worker — full-resolution processing using V2 pipeline.
 *
 * Uses V2 WASM component exclusively. Analytic fusion, f32 precision.
 */

let Pipeline = null;
let imageBytes = null;

async function initSDK() {
  try {
    const sdk = await import('../sdk/v2/rasmcore-v2-image.js');
    Pipeline = sdk.pipelineV2.ImagePipelineV2;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `V2 SDK init failed: ${e.message}` });
  }
}

// ─── Param serialization ────────────────────────────────────────────────────

function serializeParams(params, paramValues) {
  if (!params || params.length === 0) return new Uint8Array(0);
  const buf = [];
  for (const p of params) {
    const name = p.name;
    const value = paramValues[p.name];
    if (value === undefined || value === null) continue;

    buf.push(name.length);
    for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));

    if (p.type === 'bool' || typeof value === 'boolean') {
      buf.push(2);
      buf.push(value ? 1 : 0);
    } else if (Number.isInteger(value) && (p.witType === 'u32' || p.witType === 'i32' || p.type === 'u32')) {
      buf.push(1);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setUint32(0, value, true);
      buf.push(...new Uint8Array(ab));
    } else {
      buf.push(0);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, Number(value), true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}

// ─── Image Loading ──────────────────────────────────────────────────────────

function loadImage(bytes) {
  imageBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = new Pipeline();
    const src = pipe.read(imageBytes, undefined);
    const nodeInfo = pipe.nodeInfo(src);
    info = { width: nodeInfo.width, height: nodeInfo.height };
    console.log(`[v2-pipeline] Loaded: ${info.width}x${info.height}`);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-pipeline] Load failed:', detail);
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Process ────────────────────────────────────────────────────────────────

function processChain(chain, mode) {
  if (!imageBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  const timings = [];

  try {
    const pipe = new Pipeline();
    let current = pipe.read(imageBytes, undefined);

    for (const step of chain) {
      const t = performance.now();
      const params = serializeParams(step.params, step.paramValues);
      current = pipe.applyFilter(current, step.name, params);
      timings.push({ name: step.name, ms: Math.round(performance.now() - t) });
    }

    const output = pipe.write(current, 'png', undefined);
    const totalMs = Math.round(performance.now() - t0);
    console.log(`[v2-pipeline] ${totalMs}ms`);

    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, timings, totalMs, mode }, [buf]);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-pipeline] Error:', detail);
    self.postMessage({ type: 'error', message: detail });
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
    let current = pipe.read(imageBytes, undefined);

    for (const step of chain) {
      const params = serializeParams(step.params, step.paramValues);
      current = pipe.applyFilter(current, step.name, params);
    }

    const output = pipe.write(current, format, quality);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'export-result', data: buf, format }, [buf]);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    self.postMessage({ type: 'error', message: detail });
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
      processChain(e.data.chain, e.data.mode || 'full');
      break;
    case 'export':
      exportImage(e.data.chain, e.data.format || 'png', e.data.quality);
      break;
  }
};
