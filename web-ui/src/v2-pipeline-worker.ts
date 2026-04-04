// @ts-nocheck
/**
 * V2 pipeline worker — full-resolution processing using V2 fluent SDK.
 *
 * Uses the V2 fluent Pipeline class for named method dispatch:
 *   pipe.brightness({amount: 0.5}) instead of pipe.applyFilter(node, 'brightness', bytes)
 *
 * GPU dispatch: after building the filter chain, checks for a GPU plan
 * (fused shader chain). If WebGPU is available, dispatches shaders via
 * GpuHandlerV2 and injects the result back into the pipeline cache.
 */

import { Pipeline } from '../sdk/v2/fluent/index';
import { GpuHandlerV2, type GpuShader } from './gpu-handler-v2';

let PipelineClass = null;
let LayerCacheClass = null;
let layerCache = null; // Shared cross-pipeline content-addressed cache
let imageBytes = null;
let gpuHandler: GpuHandlerV2 | null = null;

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

async function initSDK() {
  try {
    const sdk = await import('../sdk/v2/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    LayerCacheClass = sdk.pipelineV2.LayerCache;

    // Create shared layer cache for cross-pipeline content-addressed caching
    if (LayerCacheClass) {
      layerCache = new LayerCacheClass(256); // 256 MB capacity
      console.log('[v2-pipeline] Layer cache created (256 MB)');
    }

    // Initialize WebGPU handler if available
    if (GpuHandlerV2.isAvailable()) {
      gpuHandler = new GpuHandlerV2();
      console.log('[v2-pipeline] WebGPU available');
    } else {
      console.log('[v2-pipeline] WebGPU not available — CPU only');
    }

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
    const pipe = Pipeline.fromRaw(PipelineClass, imageBytes, undefined, layerCache);
    info = { width: pipe.info.width, height: pipe.info.height };
    console.log(`[v2-pipeline] Loaded: ${info.width}x${info.height}`);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-pipeline] Load failed:', detail);
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Process ────────────────────────────────────────────────────────────────

async function processChain(chain, mode) {
  if (!imageBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  const timings = [];

  try {
    let pipe = Pipeline.fromRaw(PipelineClass, imageBytes, undefined, layerCache);

    // Enable tracing if the fluent SDK exposes it
    if (typeof pipe.setTracing === 'function') {
      pipe.setTracing(true);
    }

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

    // Attempt GPU dispatch if WebGPU is available and fluent SDK exposes GPU plan
    let gpuUsed = false;
    if (gpuHandler && typeof pipe.renderGpuPlan === 'function') {
      try {
        const gpuPlan = pipe.renderGpuPlan(pipe.sinkNode);
        if (gpuPlan) {
          const tGpu = performance.now();
          const ops: GpuShader[] = gpuPlan.shaders.map(s => ({
            source: s.source,
            entryPoint: s.entryPoint,
            workgroupX: s.workgroupX,
            workgroupY: s.workgroupY,
            workgroupZ: s.workgroupZ,
            params: new Uint8Array(s.params),
            extraBuffers: s.extraBuffers.map(b => new Uint8Array(b)),
          }));
          const result = await gpuHandler.execute(
            ops,
            new Float32Array(gpuPlan.inputPixels),
            gpuPlan.width,
            gpuPlan.height,
          );
          if ('ok' in result) {
            pipe.injectGpuResult(pipe.sinkNode, Array.from(result.ok));
            gpuUsed = true;
            timings.push({ name: 'gpu_dispatch', ms: Math.round(performance.now() - tGpu) });
          } else {
            console.warn('[v2-pipeline] GPU failed, falling back to CPU:', result.err);
          }
        }
      } catch (gpuErr) {
        console.warn('[v2-pipeline] GPU plan extraction failed:', gpuErr);
      }
    }

    const output = pipe.write('png');
    const totalMs = Math.round(performance.now() - t0);

    // Collect trace events
    const traceEvents = (typeof pipe.takeTrace === 'function') ? pipe.takeTrace() : [];

    // Log cache stats
    if (layerCache) {
      const s = layerCache.stats();
      console.log(`[v2-pipeline] ${totalMs}ms (GPU: ${gpuUsed}) | cache: ${s.hits} hits, ${s.misses} misses, ${s.entries} entries`);
    } else {
      console.log(`[v2-pipeline] ${totalMs}ms (GPU: ${gpuUsed})`);
    }

    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage(
      { type: 'result', png: buf, timings, totalMs, mode, gpuUsed, traceEvents },
      [buf],
    );
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
    let pipe = Pipeline.fromRaw(PipelineClass, imageBytes, undefined, layerCache);

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
