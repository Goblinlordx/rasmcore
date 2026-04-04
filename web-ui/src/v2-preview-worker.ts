// @ts-nocheck
/**
 * V2 preview worker — processes chains using the V2 fluent SDK (f32-native, fusion).
 *
 * Uses the V2 fluent Pipeline class for named method dispatch:
 *   pipe.brightness({amount: 0.5}) instead of pipe.applyFilter(node, 'brightness', bytes)
 */

import { Pipeline } from '../sdk/v2/fluent/index';
import { GpuHandlerV2, type GpuShader } from './gpu-handler-v2';

const PREVIEW_MAX = 720;

/** Detect display-p3 OffscreenCanvas support (worker-safe, no document). */
let _workerP3: boolean | null = null;
function workerSupportsP3(): boolean {
  if (_workerP3 !== null) return _workerP3;
  try {
    const oc = new OffscreenCanvas(1, 1);
    const ctx = oc.getContext('2d', { colorSpace: 'display-p3' });
    _workerP3 = ctx !== null;
  } catch {
    _workerP3 = false;
  }
  return _workerP3;
}

function workerPreferredColorSpace(): PredefinedColorSpace {
  return workerSupportsP3() ? 'display-p3' : 'srgb';
}

let PipelineClass = null;
let LayerCacheClass = null;
let layerCache = null; // Shared cross-pipeline content-addressed cache
let previewBytes = null;
let gpuHandler: GpuHandlerV2 | null = null;

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

/** Serialize params to WIT binary format: [name_len, name_bytes, type_tag, value_bytes] */
function buildParamBuf(params, paramValues) {
  const buf = [];
  if (!params) return new Uint8Array(0);
  for (const p of params) {
    const val = paramValues[p.name];
    if (val === undefined || val === null) continue;
    const name = p.name;
    buf.push(name.length);
    for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
    if (p.type === 'toggle' || typeof val === 'boolean') {
      buf.push(2); // bool
      buf.push(val ? 1 : 0);
    } else {
      buf.push(0); // f32
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, Number(val), true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}

/** Create a fluent Pipeline with layerCache + proxyScale wired via raw WIT resource. */
function createPipeline(bytes, proxyScale?: number) {
  const rawPipe = new PipelineClass();
  if (layerCache && typeof rawPipe.setLayerCache === 'function') rawPipe.setLayerCache(layerCache);
  if (proxyScale && proxyScale < 1.0 && typeof rawPipe.setProxyScale === 'function') rawPipe.setProxyScale(proxyScale);
  const node = rawPipe.read(bytes, undefined);
  const pipe = Object.create(Pipeline.prototype);
  pipe._pipe = rawPipe;
  pipe._node = node;
  return pipe;
}

async function initSDK() {
  try {
    const sdk = await import('../sdk/v2/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    LayerCacheClass = sdk.pipelineV2.LayerCache;

    // Create shared layer cache for cross-pipeline content-addressed caching
    if (LayerCacheClass) {
      layerCache = new LayerCacheClass(64); // 64 MB — preview images are small
      console.log('[v2-preview] Layer cache created (64 MB)');
    }

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

let fullWidth = 0;
let fullHeight = 0;

async function loadImage(bytes) {
  const fullBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = createPipeline(fullBytes);
    info = { width: pipe.info.width, height: pipe.info.height };
    fullWidth = info.width;
    fullHeight = info.height;

    // Downscale source image for fast preview
    const scale = computeProxyScale();
    if (scale < 1.0) {
      previewBytes = await downscaleBytes(fullBytes, scale);
      const pw = Math.round(fullWidth * scale);
      const ph = Math.round(fullHeight * scale);
      console.log(`[v2-preview] Loaded: ${fullWidth}x${fullHeight} → preview ${pw}x${ph}`);
    } else {
      previewBytes = fullBytes;
      console.log(`[v2-preview] Loaded: ${fullWidth}x${fullHeight} (no downscale needed)`);
    }
  } catch (e: any) {
    previewBytes = fullBytes;
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Load failed:', detail);
  }

  self.postMessage({ type: 'loaded', info });
}

/** Compute proxy scale factor based on full image dimensions and PREVIEW_MAX */
function computeProxyScale(): number {
  const maxDim = Math.max(fullWidth, fullHeight);
  if (maxDim <= PREVIEW_MAX) return 1.0;
  return PREVIEW_MAX / maxDim;
}

/** Downscale image bytes via OffscreenCanvas for preview. Returns PNG bytes at proxy resolution. */
async function downscaleBytes(bytes: Uint8Array, scale: number): Promise<Uint8Array> {
  const blob = new Blob([bytes]);
  const bmp = await createImageBitmap(blob);
  const w = Math.round(bmp.width * scale);
  const h = Math.round(bmp.height * scale);
  const oc = new OffscreenCanvas(w, h);
  const ctx = oc.getContext('2d', { colorSpace: workerPreferredColorSpace() })!;
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const outBlob = await oc.convertToBlob({ type: 'image/png' });
  return new Uint8Array(await outBlob.arrayBuffer());
}

// ─── Process ────────────────────────────────────────────────────────────────

async function processChain(chain) {
  if (!previewBytes) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  try {
    // previewBytes are already downscaled at load time — no proxyScale needed
    let pipe = createPipeline(previewBytes);

    for (const step of chain) {
      const method = snakeToCamel(step.name);
      if (typeof pipe[method] !== 'function') {
        // Fluent SDK missing method — fall back to raw applyFilter
        const raw = pipe._pipe;
        if (raw && typeof raw.applyFilter === 'function') {
          const paramBuf = buildParamBuf(step.params, step.paramValues);
          const node = raw.applyFilter(pipe._node, step.name, paramBuf);
          pipe = Object.create(Pipeline.prototype);
          pipe._pipe = raw;
          pipe._node = node;
        } else {
          console.warn(`[v2-preview] Unknown filter: ${step.name} (${method})`);
        }
        continue;
      }
      if (!step.params || step.params.length === 0) {
        pipe = pipe[method]();
      } else {
        const config = buildConfig(step.params, step.paramValues);
        pipe = pipe[method](config);
      }
    }

    // GPU dispatch — access raw WIT resource via pipe._pipe (private but accessible at runtime)
    const raw = pipe._pipe;
    const sinkNode = pipe._node;
    if (gpuHandler && raw && typeof raw.renderGpuPlan === 'function') {
      try {
        const gpuPlan = raw.renderGpuPlan(sinkNode);
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
            raw.injectGpuResult(sinkNode, Array.from(result.ok));
          }
        }
      } catch (_) {
        // GPU failed — CPU fallback via write() below
      }
    }

    const output = pipe.write('png');
    const totalMs = Math.round(performance.now() - t0);

    if (layerCache) {
      const s = layerCache.stats();
      console.log(`[v2-preview] ${totalMs}ms | cache: ${s.hits} hits, ${s.misses} misses, ${s.entries} entries`);
    } else {
      console.log(`[v2-preview] ${totalMs}ms`);
    }

    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, totalMs, proxyMax: PREVIEW_MAX }, [buf]);
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
