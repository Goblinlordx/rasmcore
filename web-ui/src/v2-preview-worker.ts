// @ts-nocheck
/**
 * V2 preview worker — processes chains using the V2 fluent SDK (f32-native, fusion).
 *
 * Uses the V2 fluent Pipeline class for named method dispatch:
 *   pipe.brightness({amount: 0.5}) instead of pipe.applyFilter(node, 'brightness', bytes)
 */

import { Pipeline } from '../sdk/v2/fluent/index';
import { GpuHandlerV2, type GpuShader } from './gpu-handler-v2';

const PREVIEW_MAX = 400;

let PipelineClass = null;
let previewBytes = null;
let gpuHandler: GpuHandlerV2 | null = null;

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

async function initSDK() {
  try {
    const sdk = await import('../sdk/v2/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;

    if (GpuHandlerV2.isAvailable()) {
      gpuHandler = new GpuHandlerV2();
    }

    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `V2 Preview SDK init failed: ${e.message}` });
  }
}

// ─── Config building ────────────────────────────────────────────────────────

function buildConfig(params, paramValues) {
  const config = {};
  for (const p of params) {
    const value = paramValues[p.name];
    if (value === undefined || value === null) continue;
    config[snakeToCamel(p.name)] = value;
  }
  return config;
}

// ─── Load ───────────────────────────────────────────────────────────────────

function loadImage(bytes) {
  previewBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = Pipeline.fromRaw(PipelineClass, previewBytes);
    info = { width: pipe.info.width, height: pipe.info.height };
    console.log(`[v2-preview] Loaded: ${info.width}x${info.height}`);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Load failed:', detail);
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Process ────────────────────────────────────────────────────────────────

async function processChain(chain) {
  if (!previewBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  try {
    let pipe = Pipeline.fromRaw(PipelineClass, previewBytes);

    for (const step of chain) {
      const method = snakeToCamel(step.name);
      if (!step.params || step.params.length === 0) {
        pipe = pipe[method]();
      } else {
        const config = buildConfig(step.params, step.paramValues);
        pipe = pipe[method](config);
      }
    }

    // Attempt GPU dispatch
    if (gpuHandler) {
      try {
        const gpuPlan = pipe.renderGpuPlan(pipe.sinkNode);
        if (gpuPlan) {
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
          }
        }
      } catch (_) {
        // GPU failed — CPU fallback via write() below
      }
    }

    const output = pipe.write('png');
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
