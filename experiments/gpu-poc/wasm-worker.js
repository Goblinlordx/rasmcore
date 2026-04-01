/**
 * WASM benchmark worker — runs rasmcore pipeline ops and reports timing.
 *
 * Messages:
 *   init → { type: 'ready' } | { type: 'error', message }
 *   bench { op, config, pixels (ArrayBuffer), width, height } → { type: 'result', ms }
 */

let Pipeline = null;

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      const sdk = await import('./sdk/rasmcore-image.js');
      Pipeline = sdk.pipeline.ImagePipeline;
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: `SDK init: ${err.message}` });
    }
    return;
  }

  if (type === 'bench') {
    if (!Pipeline) {
      self.postMessage({ type: 'error', message: 'SDK not loaded' });
      return;
    }

    const { op, config, pixels, width, height } = e.data;
    const imageBytes = new Uint8Array(pixels);

    // Warm up (3 runs)
    for (let i = 0; i < 3; i++) {
      try {
        runOp(op, config, imageBytes);
      } catch (err) {
        self.postMessage({ type: 'error', message: `${op}: ${err.message}` });
        return;
      }
    }

    // Measure (10 runs)
    const times = [];
    for (let i = 0; i < 10; i++) {
      const t0 = performance.now();
      try {
        runOp(op, config, imageBytes);
      } catch (err) {
        self.postMessage({ type: 'error', message: `${op}: ${err.message}` });
        return;
      }
      times.push(performance.now() - t0);
    }

    times.sort((a, b) => a - b);
    self.postMessage({
      type: 'result',
      op,
      median: times[Math.floor(times.length / 2)],
      p95: times[Math.floor(times.length * 0.95)],
    });
    return;
  }
};

function runOp(op, config, imageBytes) {
  const pipe = new Pipeline();
  const src = pipe.read(imageBytes);
  let node;

  switch (op) {
    case 'blur':
      node = pipe.blur(src, { radius: config.radius });
      break;
    case 'spinBlur':
      node = pipe.spinBlur(src, { angle: config.angle, samples: config.samples });
      break;
    case 'spherize':
      node = pipe.spherize(src, { strength: config.strength });
      break;
    case 'bilateral':
      node = pipe.bilateral(src, {
        spatialSigma: config.spatialSigma,
        rangeSigma: config.rangeSigma,
        radius: config.radius,
      });
      break;
    default:
      throw new Error(`Unknown op: ${op}`);
  }

  // Force execution by writing output
  pipe.writePng(node, {}, undefined);
}
