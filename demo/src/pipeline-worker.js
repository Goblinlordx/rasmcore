/**
 * rasmcore pipeline worker — runs all WASM processing off the main thread.
 *
 * Protocol:
 *   Main → Worker: { type: 'init' | 'load' | 'process' | 'export' }
 *   Worker → Main: { type: 'ready' | 'loaded' | 'result' | 'exported' | 'error' }
 *
 * Supports both full-res and thumbnail modes via the `mode` field.
 * Uses Transferable ArrayBuffer for zero-copy image data transfer.
 */

let Pipeline = null;
let imageBytes = null;
let thumbBytes = null;
const THUMB_MAX = 256;

// ─── SDK Loading ────────────────────────────────────────────────────────────

async function initSDK() {
  try {
    const sdk = await import('../sdk/rasmcore-image.js');
    Pipeline = sdk.pipeline.ImagePipeline;
    self.postMessage({ type: 'ready' });
  } catch (e) {
    self.postMessage({ type: 'error', message: `SDK init failed: ${e.message}` });
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

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

// ─── Image Loading ──────────────────────────────────────────────────────────

function loadImage(bytes) {
  imageBytes = new Uint8Array(bytes);
  let info = { width: 0, height: 0 };

  try {
    const pipe = new Pipeline();
    const src = pipe.read(imageBytes);
    info = pipe.nodeInfo(src);

    // Create thumbnail
    const scale = Math.min(THUMB_MAX / info.width, THUMB_MAX / info.height, 1);
    if (scale < 1) {
      const tw = Math.round(info.width * scale);
      const th = Math.round(info.height * scale);
      const resized = pipe.resize(src, tw, th, 'bilinear');
      thumbBytes = pipe.writePng(resized, {}, undefined);
    } else {
      thumbBytes = imageBytes;
    }
  } catch (e) {
    thumbBytes = imageBytes;
  }

  self.postMessage({ type: 'loaded', info });
}

// ─── Pipeline Processing ────────────────────────────────────────────────────

function processChain(chain, mode) {
  const source = mode === 'thumb' ? thumbBytes : imageBytes;
  if (!source) {
    self.postMessage({ type: 'error', message: 'No image loaded' });
    return;
  }

  const t0 = performance.now();
  const timings = [];

  try {
    const pipe = new Pipeline();
    let node = pipe.read(source);

    for (const step of chain) {
      const t = performance.now();
      const args = expandArgs(step.params, step.paramValues);

      if (step.name === 'resize' && mode === 'thumb') {
        const info = pipe.nodeInfo(node);
        const scale = Math.min(THUMB_MAX / args[0], THUMB_MAX / args[1], 1);
        node = pipe.resize(node, Math.max(1, Math.round(args[0] * scale)), Math.max(1, Math.round(args[1] * scale)), args[2]);
      } else if (step.name === 'crop' && mode === 'thumb') {
        const info = pipe.nodeInfo(node);
        node = pipe.crop(node, 0, 0, Math.min(args[2], info.width), Math.min(args[3], info.height));
      } else {
        node = pipe[step.name](node, ...args);
      }

      timings.push({ name: step.name, ms: Math.round(performance.now() - t) });
    }

    const output = pipe.writePng(node, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);

    self.postMessage({ type: 'result', png: buf, timings, totalMs, mode }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
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
    let node = pipe.read(imageBytes);

    for (const step of chain) {
      const args = expandArgs(step.params, step.paramValues);
      node = pipe[step.name](node, ...args);
    }

    let output, mime;
    if (format === 'jpeg') { output = pipe.writeJpeg(node, { quality }, undefined); mime = 'image/jpeg'; }
    else if (format === 'webp') { output = pipe.writeWebp(node, { quality }, undefined); mime = 'image/webp'; }
    else { output = pipe.writePng(node, {}, undefined); mime = 'image/png'; }

    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'exported', data: buf, mime, format }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
  }
}

// ─── Multi-Layer Composite ───────────────────────────────────────────────────

function compositeLayers(layerDefs) {
  if (!layerDefs || layerDefs.length === 0) {
    self.postMessage({ type: 'error', message: 'No layers to composite' });
    return;
  }

  const t0 = performance.now();
  try {
    const pipe = new Pipeline();

    // Process each layer: load → apply chain → get node
    let resultNode = null;
    for (let i = 0; i < layerDefs.length; i++) {
      const layer = layerDefs[i];
      const bytes = new Uint8Array(layer.imageBytes);
      let node = pipe.read(bytes);

      // Apply per-layer chain
      for (const step of layer.chain) {
        const args = expandArgs(step.params, step.paramValues);
        node = pipe[step.name](node, ...args);
      }

      if (i === 0) {
        resultNode = node;
      } else {
        // Composite this layer onto the result
        // Map blend mode string to WIT enum value
        const mode = layer.blendMode || undefined;
        resultNode = pipe.composite(node, resultNode, layer.x || 0, layer.y || 0, mode);
      }
    }

    const output = pipe.writePng(resultNode, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);
    const buf = output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
    self.postMessage({ type: 'result', png: buf, timings: [], totalMs, mode: 'full' }, [buf]);
  } catch (e) {
    self.postMessage({ type: 'error', message: `Composite error: ${e.message}` });
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
      processChain(e.data.chain, e.data.mode);
      break;
    case 'export':
      exportImage(e.data.chain, e.data.format, e.data.quality);
      break;
    case 'composite':
      compositeLayers(e.data.layers);
      break;
  }
};
