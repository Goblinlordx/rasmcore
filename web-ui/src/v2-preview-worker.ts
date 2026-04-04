// @ts-nocheck
/**
 * V2 preview worker — processes chains using the V2 pipeline (f32-native, fusion).
 *
 * Uses the V2 WASM component exclusively. No V1 code in the execution path.
 * Analytic operations (brightness, contrast, etc.) are fused at graph setup.
 */

const PREVIEW_MAX = 400;

let Pipeline = null;
let previewBytes = null;

async function initSDK() {
  try {
    const sdk = await import('../sdk/v2/rasmcore-v2-image.js');
    Pipeline = sdk.pipelineV2.ImagePipelineV2;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `V2 Preview SDK init failed: ${e.message}` });
  }
}

// ─── Param serialization (V2 binary format) ─────────────────────────────────

function serializeParams(params, paramValues) {
  if (!params || params.length === 0) return new Uint8Array(0);
  const buf = [];
  for (const p of params) {
    const name = p.name;
    const value = paramValues[p.name];
    if (value === undefined || value === null) continue;

    // Push name
    buf.push(name.length);
    for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));

    // Push typed value
    if (p.type === 'bool' || typeof value === 'boolean') {
      buf.push(2); // bool
      buf.push(value ? 1 : 0);
    } else if (Number.isInteger(value) && (p.witType === 'u32' || p.witType === 'i32' || p.type === 'u32')) {
      buf.push(1); // u32
      const ab = new ArrayBuffer(4);
      new DataView(ab).setUint32(0, value, true);
      buf.push(...new Uint8Array(ab));
    } else {
      buf.push(0); // f32
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, Number(value), true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}

// ─── Load ───────────────────────────────────────────────────────────────────

function loadImage(bytes) {
  previewBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = new Pipeline();
    const src = pipe.read(previewBytes, undefined);
    const nodeInfo = pipe.nodeInfo(src);
    info = { width: nodeInfo.width, height: nodeInfo.height };
    console.log(`[v2-preview] Loaded: ${info.width}x${info.height}`);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Load failed:', detail);
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Process ────────────────────────────────────────────────────────────────

function processChain(chain) {
  if (!previewBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  try {
    const pipe = new Pipeline();
    let current = pipe.read(previewBytes, undefined);

    for (const step of chain) {
      const params = serializeParams(step.params, step.paramValues);
      current = pipe.applyFilter(current, step.name, params);
    }

    const output = pipe.write(current, 'png', undefined);
    const totalMs = Math.round(performance.now() - t0);
    console.log(`[v2-preview] ${totalMs}ms`);

    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, totalMs }, [buf]);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Process failed:', detail);
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
      processChain(e.data.chain);
      break;
  }
};
