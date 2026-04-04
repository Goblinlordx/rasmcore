// @ts-nocheck
/**
 * V2 pipeline worker — full-resolution processing using V2 fluent SDK.
 *
 * Uses the V2 fluent Pipeline class for named method dispatch:
 *   pipe.brightness({amount: 0.5}) instead of pipe.applyFilter(node, 'brightness', bytes)
 */

import { Pipeline } from '../sdk/v2/fluent/index';

let PipelineClass = null;
let imageBytes = null;

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

async function initSDK() {
  try {
    const sdk = await import('../sdk/v2/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `V2 SDK init failed: ${e.message}` });
  }
}

// ─── Config building (same pattern as V1 workers) ──────────────────────────

function buildConfig(params, paramValues) {
  const config = {};
  for (const p of params) {
    const value = paramValues[p.name];
    if (value === undefined || value === null) continue;
    config[snakeToCamel(p.name)] = value;
  }
  return config;
}

// ─── Image Loading ──────────────────────────────────────────────────────────

function loadImage(bytes) {
  imageBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = Pipeline.fromRaw(PipelineClass, imageBytes);
    info = { width: pipe.info.width, height: pipe.info.height };
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
    let pipe = Pipeline.fromRaw(PipelineClass, imageBytes);

    for (const step of chain) {
      const t = performance.now();
      const method = snakeToCamel(step.name);
      if (!step.params || step.params.length === 0) {
        pipe = pipe[method]();
      } else {
        const config = buildConfig(step.params, step.paramValues);
        pipe = pipe[method](config);
      }
      timings.push({ name: step.name, ms: Math.round(performance.now() - t) });
    }

    const output = pipe.write('png');
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
    let pipe = Pipeline.fromRaw(PipelineClass, imageBytes);

    for (const step of chain) {
      const method = snakeToCamel(step.name);
      if (!step.params || step.params.length === 0) {
        pipe = pipe[method]();
      } else {
        const config = buildConfig(step.params, step.paramValues);
        pipe = pipe[method](config);
      }
    }

    const output = pipe.write(format, quality);
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
