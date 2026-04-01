// @ts-nocheck
// eslint-disable
/**
 * rasmcore preview worker — processes chains on a downscaled copy for fast feedback.
 *
 * On load: resizes image to max PREVIEW_MAX px (longest side), stores as previewBytes.
 * On process: applies chain to previewBytes and returns PNG.
 * Separate Pipeline instance from the full-res worker — independent cache.
 */

const PREVIEW_MAX = 400;

let Pipeline = null;
let previewBytes = null; // Downscaled image bytes
let cachedPipe = null;
let cachedSourceNode = null;

async function initSDK() {
  try {
    const sdk = await import('../sdk/rasmcore-image.js');
    Pipeline = sdk.pipeline.ImagePipeline;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `Preview SDK init failed: ${e.message}` });
  }
}

// ─── Helpers (same as pipeline-worker) ──────────────────────────────────────

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function expandArgs(params, paramValues) {
  const args = [];
  for (const p of params) {
    if (p.type === 'color') {
      const [r, g, b] = hexToRgb(paramValues[p.name] || '#808080');
      args.push(r, g, b);
    } else {
      args.push(paramValues[p.name]);
    }
  }
  return args;
}

const EXTRA_POSITIONAL_OPS = new Set([
  'convolve',
  'displacementMap',
  'curvesRed',
  'curvesGreen',
  'curvesBlue',
  'curvesMaster',
  'hueVsSat',
  'hueVsLum',
  'lumVsSat',
  'satVsSat',
  'applyCubeLut',
  'gradientMap',
]);

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

function applyStep(pipe, node, step) {
  const name = step.name;
  if (EXTRA_POSITIONAL_OPS.has(name)) {
    const args = expandArgs(step.params, step.paramValues);
    return pipe[name](node, ...args);
  }
  if (!step.params || step.params.length === 0) {
    return pipe[name](node);
  }
  const config = buildConfig(step.params, step.paramValues);
  return pipe[name](node, config);
}

// ─── Load (with downscale) ──────────────────────────────────────────────────

function loadImage(bytes) {
  const raw = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    cachedPipe = new Pipeline();
    const src = cachedPipe.read(raw);
    info = cachedPipe.nodeInfo(src);

    const origW = info.width,
      origH = info.height;
    // Downscale if larger than PREVIEW_MAX
    const scale = Math.min(PREVIEW_MAX / info.width, PREVIEW_MAX / info.height, 1);
    if (scale < 1) {
      const tw = Math.max(1, Math.round(info.width * scale));
      const th = Math.max(1, Math.round(info.height * scale));
      const resized = cachedPipe.resize(src, { width: tw, height: th, filter: 'bilinear' });
      const png = cachedPipe.writePng(resized, {}, undefined);
      previewBytes = png;
      // Re-create pipeline with downscaled source for caching
      cachedPipe = new Pipeline();
      cachedSourceNode = cachedPipe.read(previewBytes);
      const pInfo = cachedPipe.nodeInfo(cachedSourceNode);
      info = { width: pInfo.width, height: pInfo.height };
      console.log(`[preview-worker] Resized ${origW}x${origH} → ${info.width}x${info.height}`);
    } else {
      previewBytes = raw;
      cachedSourceNode = src;
      console.log(`[preview-worker] No resize needed: ${origW}x${origH} <= ${PREVIEW_MAX}px`);
    }
  } catch (e) {
    console.error('[preview-worker] Load failed:', e);
    previewBytes = raw;
    cachedPipe = null;
    cachedSourceNode = null;
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Process ────────────────────────────────────────────────────────────────

function processChain(chain) {
  if (!previewBytes) {
    self.postMessage({ type: 'error', message: 'No preview image loaded' });
    return;
  }

  const t0 = performance.now();
  try {
    const pipe = cachedPipe || new Pipeline();
    let current = cachedSourceNode || pipe.read(previewBytes);

    for (const step of chain) {
      current = applyStep(pipe, current, step);
    }

    const output = pipe.writePng(current, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, totalMs }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
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
