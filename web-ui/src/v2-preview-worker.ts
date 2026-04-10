// @ts-nocheck
/**
 * V2 preview worker — processes chains using the V2 fluent SDK.
 *
 * Uses Source (decode once) and RenderTarget (GPU canvas blit) from the SDK.
 * The worker is a thin message handler around the SDK API.
 */

import { Pipeline } from '../sdk/pipeline';
import { GpuHandlerV2, type GpuShader } from './gpu-handler-v2';
import { RenderTarget } from '../sdk/lib/render-target';

const PREVIEW_MAX = 720;

let PipelineClass = null;
let SourceClass = null;
let LayerCacheClass = null;
let layerCache = null;
let gpuHandler: GpuHandlerV2 | null = null;
let displayMode = false;
let tracingEnabled = false;

// Source-based decode caching — decode once, reuse across processChain calls
let currentSource = null; // WIT Source resource (holds decoded pixels)
let previewBytes: Uint8Array | null = null; // raw bytes for fallback
let fullWidth = 0;
let fullHeight = 0;

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

/** Serialize params to WIT binary format. */
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
      buf.push(2);
      buf.push(val ? 1 : 0);
    } else {
      buf.push(0);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, Number(val), true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}

function buildConfig(params, paramValues) {
  const config = {};
  for (const p of params) {
    const value = paramValues[p.name];
    if (value === undefined || value === null) continue;
    config[snakeToCamel(p.name)] = value;
  }
  return config;
}

/** Create a fluent Pipeline from a Source (decode cached) or raw bytes. */
function createPipelineFromSource() {
  const rawPipe = new PipelineClass();
  if (layerCache && typeof rawPipe.setLayerCache === 'function') rawPipe.setLayerCache(layerCache);
  if (tracingEnabled && typeof rawPipe.setTracing === 'function') rawPipe.setTracing(true);

  let node;
  if (currentSource && typeof rawPipe.readSource === 'function') {
    // Source path — no re-decode, pixels cached in Source resource
    node = rawPipe.readSource(currentSource);
  } else if (previewBytes) {
    // Fallback — decode from raw bytes
    node = rawPipe.read(previewBytes, undefined);
  } else {
    return null;
  }

  const pipe = Object.create(Pipeline.prototype);
  pipe._pipe = rawPipe;
  pipe._node = node;
  return pipe;
}

/** Collect trace events. */
function collectTrace(rawPipe) {
  if (!tracingEnabled || !rawPipe || typeof rawPipe.takeTrace !== 'function') return null;
  try {
    const events = rawPipe.takeTrace();
    if (!events || events.length === 0) return null;
    return events.map(e => ({
      name: `${e.kind}:${e.name}`,
      ms: e.durationUs / 1000,
      detail: e.detail || undefined,
    }));
  } catch { return null; }
}

function logTrace(trace, totalMs: number) {
  if (!trace) return;
  console.group(`[v2-preview] Trace (${totalMs}ms total)`);
  for (const e of trace) {
    const detail = e.detail ? ` — ${e.detail}` : '';
    console.log(`  ${e.name}: ${e.ms.toFixed(1)}ms${detail}`);
  }
  console.groupEnd();
}

// ─── Init ──────────────────────────────────────────────────────────────────

async function initSDK() {
  try {
    const sdk = await import('../sdk/wasm/rasmcore-v2-image.js');
    PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    SourceClass = sdk.pipelineV2.Source;
    LayerCacheClass = sdk.pipelineV2.LayerCache;

    if (LayerCacheClass) {
      layerCache = new LayerCacheClass(64);
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

// ─── Load ──────────────────────────────────────────────────────────────────

async function loadImage(bytes) {
  const fullBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    // Create Source — decodes once, cached for all future processChain calls
    if (SourceClass) {
      currentSource = new SourceClass(fullBytes, undefined);
      const srcInfo = currentSource.info();
      info = { width: srcInfo.width, height: srcInfo.height };
    } else {
      // Fallback — no Source class available
      const pipe = createPipelineFromSource();
      if (pipe) info = { width: pipe.info.width, height: pipe.info.height };
    }

    fullWidth = info.width;
    fullHeight = info.height;
    previewBytes = fullBytes; // keep for fallback
    console.log(`[v2-preview] Loaded: ${fullWidth}x${fullHeight} (Source cached)`);
  } catch (e: any) {
    previewBytes = fullBytes;
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Load failed:', detail);
  }

  self.postMessage({ type: 'loaded', info, previewWidth: fullWidth, previewHeight: fullHeight });
  renderOriginalSource();
}

// ─── Process ───────────────────────────────────────────────────────────────

async function processChain(chain) {
  if (!previewBytes && !currentSource) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  let pipeRaw: any = null; // track raw pipeline for cleanup in finally
  try {
    let pipe = createPipelineFromSource();
    if (!pipe) {
      self.postMessage({ type: 'error', message: 'Failed to create pipeline' });
      return;
    }
    pipeRaw = pipe._pipe;

    // Apply filter chain
    for (const step of chain) {
      const method = snakeToCamel(step.name);
      if (typeof pipe[method] !== 'function') {
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

    // GPU dispatch
    const raw = pipe._pipe;
    const sinkNode = pipe._node;
    let gpuDisplayed = false;

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

          await gpuHandler.prepare(ops);

          if (displayMode && gpuHandler.hasDisplay) {
            const err = await gpuHandler.executeAndDisplay(
              ops, new Float32Array(gpuPlan.inputPixels), gpuPlan.width, gpuPlan.height,
            );
            if (!err) gpuDisplayed = true;
          } else {
            const result = await gpuHandler.execute(
              ops, new Float32Array(gpuPlan.inputPixels), gpuPlan.width, gpuPlan.height,
            );
            if ('ok' in result) {
              raw.injectGpuResult(sinkNode, Array.from(result.ok));
            }
          }
        }
      } catch (_) { /* GPU failed — CPU fallback */ }
    }

    if (gpuDisplayed) {
      const totalMs = Math.round(performance.now() - t0);
      const trace = collectTrace(raw);
      logTrace(trace, totalMs);
      if (layerCache) {
        const s = layerCache.stats();
        console.log(`[v2-preview] ${totalMs}ms (display) | cache: ${s.hits} hits, ${s.misses} misses, ${s.entries} entries`);
      }
      self.postMessage({ type: 'displayed', totalMs, proxyMax: PREVIEW_MAX, trace });
      return;
    }

    // CPU fallback — display mode
    if (displayMode && gpuHandler?.hasDisplay) {
      try {
        if (raw && typeof raw.render === 'function') {
          const pixels = raw.render(sinkNode);
          if (pixels && pixels.length > 0) {
            const info2 = raw.nodeInfo(sinkNode);
            const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
            gpuHandler.displayFromCpu('viewport', f32, info2.width, info2.height);
            const totalMs = Math.round(performance.now() - t0);
            self.postMessage({ type: 'displayed', totalMs, proxyMax: PREVIEW_MAX });
            return;
          }
        }
      } catch (_) { /* fall through to PNG */ }
    }

    // PNG fallback
    const output = pipe.write('png');
    const totalMs = Math.round(performance.now() - t0);
    const trace = collectTrace(pipe._pipe);
    logTrace(trace, totalMs);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, totalMs, proxyMax: PREVIEW_MAX, trace }, [buf]);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Process failed:', detail);
    self.postMessage({ type: 'error', message: detail });
  } finally {
    // Finalize layer cache — evict unreferenced entries from previous render cycles.
    // Without this, old image/filter results accumulate indefinitely.
    if (pipeRaw && typeof pipeRaw.finalizeLayerCache === 'function') {
      try { pipeRaw.finalizeLayerCache(); } catch { /* ignore */ }
    }
  }
}

// ─── Display Surface ───────────────────────────────────────────────────────

async function initDisplay(canvas: OffscreenCanvas, hdr: boolean) {
  if (!gpuHandler) return;
  try {
    await gpuHandler.setDisplayCanvas(canvas, hdr);
    displayMode = true;
    console.log(`[v2-preview] Display mode enabled (HDR: ${hdr})`);
  } catch (e: any) {
    console.warn('[v2-preview] Display init failed, staying in PNG mode:', e?.message);
  }
}

function handleResizeCanvas(width: number, height: number) {
  if (!gpuHandler || !displayMode) return;
  gpuHandler.resizeDisplay('viewport', width, height);
}

async function initOriginalDisplay(canvas: OffscreenCanvas, hdr: boolean) {
  if (!gpuHandler) return;
  try {
    await gpuHandler.setOriginalCanvas(canvas, hdr);
    console.log(`[v2-preview] Original display enabled (HDR: ${hdr})`);
    renderOriginalSource();
  } catch (e: any) {
    console.warn('[v2-preview] Original display init failed:', e?.message);
  }
}

function renderOriginalSource() {
  if (!gpuHandler || !gpuHandler.hasOriginalDisplay) return;
  if (!PipelineClass) return;
  if (!currentSource && !previewBytes) return;
  try {
    const pipe = createPipelineFromSource();
    if (!pipe) return;
    const raw = pipe._pipe;
    if (!raw || typeof raw.render !== 'function') return;
    const pixels = raw.render(pipe._node);
    gpuHandler.storeSourcePixels(new Float32Array(pixels), pipe.info.width, pipe.info.height);
  } catch (e: any) {
    console.warn('[v2-preview] Original source render failed:', e?.message);
  }
}

function handleViewport(data: any) {
  if (!gpuHandler || !displayMode) return;
  gpuHandler.resizeDisplay('viewport', data.canvasWidth, data.canvasHeight);
  gpuHandler.updateViewport(
    data.panX, data.panY, data.zoom,
    data.canvasWidth, data.canvasHeight,
    gpuHandler.imageWidth, gpuHandler.imageHeight,
    data.toneMode ?? 0,
  );
  gpuHandler.displayOnly('viewport');

  if (gpuHandler.hasOriginalDisplay) {
    gpuHandler.resizeOriginalDisplay(data.canvasWidth, data.canvasHeight);
    gpuHandler.updateOriginalViewport(
      data.panX, data.panY, data.zoom,
      data.canvasWidth, data.canvasHeight,
      data.toneMode ?? 0,
    );
    gpuHandler.blitOriginal();
  }
}

// ─── Multi-output rendering (ref/branch/display) ─────────────────────────

async function processMulti(chain: any[], scopes: string[]) {
  if (!previewBytes && !currentSource) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  let pipeRaw: any = null;
  try {
    let pipe = createPipelineFromSource();
    if (!pipe) {
      self.postMessage({ type: 'error', message: 'Failed to create pipeline' });
      return;
    }
    pipeRaw = pipe._pipe;

    // Apply filter chain
    for (const step of chain) {
      const method = snakeToCamel(step.name);
      if (typeof pipe[method] !== 'function') {
        const raw = pipe._pipe;
        if (raw && typeof raw.applyFilter === 'function') {
          const paramBuf = buildParamBuf(step.params, step.paramValues);
          const node = raw.applyFilter(pipe._node, step.name, paramBuf);
          pipe = Object.create(Pipeline.prototype);
          pipe._pipe = raw;
          pipe._node = node;
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

    // Set ref at the end of the chain for scope branches
    const raw = pipe._pipe;
    const viewportNode = pipe._node;
    if (typeof raw.setRef === 'function') {
      raw.setRef('main', viewportNode);
    }

    // Build display targets: viewport + each scope
    const targets: [string, number][] = [['viewport', viewportNode]];

    for (const scopeName of scopes) {
      // Branch from ref, append scope filter
      if (typeof raw.applyFilter === 'function') {
        const scopeNode = raw.applyFilter(viewportNode, scopeName, new Uint8Array(0));
        targets.push([scopeName, scopeNode]);
      }
    }

    // Try GPU multi-plan dispatch
    let multiDisplayed = false;
    if (gpuHandler && displayMode && gpuHandler.hasDisplay('viewport') && typeof raw.renderMultiGpuPlan === 'function') {
      try {
        const plan = raw.renderMultiGpuPlan(targets);
        if (plan && plan.stages && plan.stages.length > 0) {
          const multiStages = plan.stages.map(s => ({
            name: s.targetName,
            shaders: s.shaders.map(sh => ({
              source: sh.source,
              entryPoint: sh.entryPoint,
              workgroupX: sh.workgroupX,
              workgroupY: sh.workgroupY,
              workgroupZ: sh.workgroupZ,
              params: new Uint8Array(sh.params),
              extraBuffers: sh.extraBuffers.map(b => new Uint8Array(b)),
            })),
            input: s.input.tag === 'pixels'
              ? { tag: 'pixels' as const, data: new Float32Array(s.input.val) }
              : { tag: 'prior' as const, name: s.input.val },
            width: s.width,
            height: s.height,
          }));

          const err = await gpuHandler.executeMulti(multiStages);
          if (!err) multiDisplayed = true;
        }
      } catch (e: any) {
        console.warn('[v2-preview] Multi GPU dispatch failed, falling back:', e?.message);
      }
    }

    if (multiDisplayed) {
      const totalMs = Math.round(performance.now() - t0);
      const trace = collectTrace(raw);
      logTrace(trace, totalMs);
      self.postMessage({ type: 'displayed', totalMs, proxyMax: PREVIEW_MAX, trace, multi: true });
      return;
    }

    // CPU fallback: render viewport via existing single-output path
    // Viewport
    if (displayMode && gpuHandler?.hasDisplay('viewport')) {
      try {
        const pixels = raw.render(viewportNode);
        if (pixels && pixels.length > 0) {
          const info = raw.nodeInfo(viewportNode);
          const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
          gpuHandler.displayFromCpu('viewport', f32, info.width, info.height);
        }
      } catch (_) { /* fall through */ }
    }

    // Scopes — render to PNG (CPU path, but chain is already cached from viewport render)
    for (const scopeName of scopes) {
      try {
        const scopeNode = raw.applyFilter(viewportNode, scopeName, new Uint8Array(0));
        const scopePixels = raw.render(scopeNode);
        if (scopePixels && scopePixels.length > 0) {
          const scopeInfo = raw.nodeInfo(scopeNode);
          // If scope has a display target, use GPU blit
          if (gpuHandler?.hasDisplay(scopeName)) {
            const f32 = scopePixels instanceof Float32Array ? scopePixels : new Float32Array(scopePixels);
            gpuHandler.displayFromCpu(scopeName, f32, scopeInfo.width, scopeInfo.height);
          } else {
            // PNG fallback for scope
            const pngPipe = Object.create(Pipeline.prototype);
            pngPipe._pipe = raw;
            pngPipe._node = scopeNode;
            const output = pngPipe.write('png');
            const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
            self.postMessage({ type: 'scope-result', png: buf, scopeName, totalMs: 0 }, [buf]);
          }
        }
      } catch (_) { /* scope failure is non-critical */ }
    }

    const totalMs = Math.round(performance.now() - t0);
    self.postMessage({ type: 'displayed', totalMs, proxyMax: PREVIEW_MAX, multi: true });
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.error('[v2-preview] Multi process failed:', detail);
    self.postMessage({ type: 'error', message: detail });
  } finally {
    if (pipeRaw && typeof pipeRaw.finalizeLayerCache === 'function') {
      try { pipeRaw.finalizeLayerCache(); } catch { /* ignore */ }
    }
  }
}

// ─── Scope Rendering (legacy single-scope path) ─────────────────────────

async function processScope(chain: any[], scopeName: string) {
  if (!previewBytes && !currentSource) {
    return; // no image loaded
  }

  const t0 = performance.now();
  try {
    let pipe = createPipelineFromSource();
    if (!pipe) return;

    // Apply the user's filter chain first
    for (const step of chain) {
      const method = snakeToCamel(step.name);
      if (typeof pipe[method] === 'function') {
        if (!step.params || step.params.length === 0) {
          pipe = pipe[method]();
        } else {
          const config = buildConfig(step.params, step.paramValues);
          pipe = pipe[method](config);
        }
      } else {
        // Fallback to raw applyFilter
        const raw = pipe._pipe;
        if (raw && typeof raw.applyFilter === 'function') {
          const paramBuf = buildParamBuf(step.params, step.paramValues);
          const node = raw.applyFilter(pipe._node, step.name, paramBuf);
          pipe = Object.create(Pipeline.prototype);
          pipe._pipe = raw;
          pipe._node = node;
        }
      }
    }

    // Append scope filter (no params — uses defaults: 512x512)
    const raw = pipe._pipe;
    if (raw && typeof raw.applyFilter === 'function') {
      const scopeNode = raw.applyFilter(pipe._node, scopeName, new Uint8Array(0));
      pipe = Object.create(Pipeline.prototype);
      pipe._pipe = raw;
      pipe._node = scopeNode;
    }

    // Render scope to PNG (always CPU — scopes are small images)
    const output = pipe.write('png');
    const totalMs = Math.round(performance.now() - t0);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'scope-result', png: buf, scopeName, totalMs }, [buf]);
  } catch (e: any) {
    const detail = e?.payload ? JSON.stringify(e.payload, null, 2) : e?.message || String(e);
    console.warn(`[v2-preview] Scope ${scopeName} failed:`, detail);
    // Don't send error — scope failure is non-critical
  }
}

// ─── Message Handler ────────────────────────────────────────────────────────

self.onmessage = (e) => {
  const { type } = e.data;
  switch (type) {
    case 'init':
      initSDK();
      break;
    case 'init-display':
      initDisplay(e.data.canvas, e.data.hdr ?? false);
      break;
    case 'init-original-display':
      initOriginalDisplay(e.data.canvas, e.data.hdr ?? false);
      break;
    case 'load':
      loadImage(e.data.imageBytes);
      break;
    case 'process':
      processChain(e.data.chain);
      break;
    case 'resize-canvas':
      handleResizeCanvas(e.data.width, e.data.height);
      break;
    case 'viewport':
      handleViewport(e.data);
      break;
    case 'process-multi':
      processMulti(e.data.chain, e.data.scopes || []);
      break;
    case 'process-scope':
      processScope(e.data.chain, e.data.scopeName);
      break;
    case 'init-scope-display':
      if (gpuHandler) {
        gpuHandler.addDisplay(e.data.name, e.data.canvas, e.data.hdr ?? false);
        console.log(`[v2-preview] Scope display registered: ${e.data.name}`);
      }
      break;
    case 'set-tracing':
      tracingEnabled = !!e.data.enabled;
      break;
  }
};
