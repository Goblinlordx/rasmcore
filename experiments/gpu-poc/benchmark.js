/**
 * GPU Offload PoC — Benchmark
 *
 * Compares WebGPU compute shaders vs WASM SIMD for expensive image operations.
 * Tests: gaussian blur (radius 20), spin blur, spherize, bilateral filter.
 * Reports: transfer overhead, per-op GPU vs WASM time, speedup ratios.
 */

const log = (msg) => {
  const el = document.getElementById('log');
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
  console.log(msg);
};

const status = (msg, isError) => {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = isError ? 'status error' : 'status';
};

// ─── WebGPU Init ────────────────────────────────────────────────────────────

let device, queue;
const shaderModules = {};
const pipelines = {};

async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found');

  // Request timestamp query feature if available
  const features = [];
  if (adapter.features.has('timestamp-query')) features.push('timestamp-query');

  device = await adapter.requestDevice({ requiredFeatures: features });
  queue = device.queue;
  const adapterInfo = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  log(`GPU: ${adapterInfo.description || adapterInfo.device || 'unknown'}`);
  log(`Timestamp queries: ${device.features.has('timestamp-query') ? 'YES' : 'NO'}`);
}

async function loadShader(name, entryPoint = 'main') {
  const resp = await fetch(`shaders/${name}.wgsl`);
  const code = await resp.text();
  const module = device.createShaderModule({ code });
  shaderModules[name] = module;
  // Pipelines created per-entrypoint when needed
  return module;
}

function getOrCreatePipeline(shaderName, entryPoint) {
  const key = `${shaderName}:${entryPoint}`;
  if (!pipelines[key]) {
    pipelines[key] = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModules[shaderName], entryPoint },
    });
  }
  return pipelines[key];
}

// ─── WASM Baseline (main thread — importmap resolves bare specifiers) ───────

let Pipeline;
let Filters; // stateless filter API (no tiling, full-image)

async function initWASM() {
  try {
    const sdk = await import('./sdk/rasmcore-image.js');
    Pipeline = sdk.pipeline.ImagePipeline;
    Filters = sdk.filters;
    // Log available methods
    const pipe = new Pipeline();
    const pipeMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(pipe))
      .filter(n => typeof pipe[n] === 'function' && !n.startsWith('_'))
      .filter(n => ['blur','spinBlur','spherize','bilateral','read','writePng'].includes(n));
    const filterMethods = Filters ? Object.keys(Filters).filter(n => ['blur','spinBlur','spherize','bilateral'].includes(n)) : [];
    log(`WASM SDK loaded — pipeline: ${pipeMethods.join(', ')}`);
    log(`WASM stateless filters: ${filterMethods.join(', ') || 'none found'}`);
  } catch (e) {
    log(`WASM SDK not available: ${e.message}`);
    log('(Run demo/build.sh first to generate SDK)');
  }
}

// ─── Test Data ──────────────────────────────────────────────────────────────

function generateTestImage(size) {
  // Random RGBA pixels (for GPU — raw buffer)
  const data = new Uint8Array(size * size * 4);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random() * 255;
  }
  return data;
}

// Generate a PNG-encoded test image for WASM pipeline (which needs encoded input)
let _wasmTestImageCache = {};
function generateWASMTestImage(size) {
  if (_wasmTestImageCache[size]) return _wasmTestImageCache[size];
  // Create a canvas, fill with gradient, export as PNG
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, size, size);
  grad.addColorStop(0, '#ff6600');
  grad.addColorStop(0.5, '#0066ff');
  grad.addColorStop(1, '#00ff66');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  // Add some noise for realism
  const imgData = ctx.getImageData(0, 0, size, size);
  for (let i = 0; i < imgData.data.length; i += 4) {
    imgData.data[i] += (Math.random() - 0.5) * 20;
    imgData.data[i+1] += (Math.random() - 0.5) * 20;
    imgData.data[i+2] += (Math.random() - 0.5) * 20;
  }
  ctx.putImageData(imgData, 0, 0);
  // Convert to PNG bytes
  const dataUrl = canvas.toDataURL('image/png');
  const binary = atob(dataUrl.split(',')[1]);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  _wasmTestImageCache[size] = bytes;
  log(`Generated ${size}x${size} test PNG: ${(bytes.length/1024).toFixed(0)}KB`);
  return bytes;
}

function gaussianKernel(radius, sigma) {
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  for (let i = 0; i < size; i++) kernel[i] /= sum;
  return kernel;
}

// ─── GPU Benchmark Helpers ──────────────────────────────────────────────────

async function benchTransferOnly(size) {
  const pixels = generateTestImage(size);
  const byteLen = size * size * 4;

  const times = [];
  for (let i = 0; i < 13; i++) {
    const t0 = performance.now();

    // Upload
    const inputBuf = device.createBuffer({
      size: byteLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    queue.writeBuffer(inputBuf, 0, pixels);

    // Readback
    const outputBuf = device.createBuffer({
      size: byteLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stagingBuf = device.createBuffer({
      size: byteLen,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(inputBuf, 0, stagingBuf, 0, byteLen);
    queue.submit([enc.finish()]);
    await stagingBuf.mapAsync(GPUMapMode.READ);
    const _result = new Uint8Array(stagingBuf.getMappedRange()).slice();
    stagingBuf.unmap();

    const t1 = performance.now();
    inputBuf.destroy();
    outputBuf.destroy();
    stagingBuf.destroy();

    if (i >= 3) times.push(t1 - t0); // skip warmup
  }
  times.sort((a, b) => a - b);
  return { median: times[Math.floor(times.length / 2)], p95: times[Math.floor(times.length * 0.95)] };
}

async function benchGPUOp(shaderName, entryPoint, size, makeParams, extraBuffers = []) {
  const pixels = generateTestImage(size);
  const byteLen = size * size * 4;

  // Precompile pipeline
  const pipeline = getOrCreatePipeline(shaderName, entryPoint);
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  const times = [];
  for (let i = 0; i < 13; i++) {
    const t0 = performance.now();

    const inputBuf = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outputBuf = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    queue.writeBuffer(inputBuf, 0, pixels);

    const paramData = makeParams(size);
    const paramBuf = device.createBuffer({ size: paramData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    queue.writeBuffer(paramBuf, 0, paramData);

    const entries = [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: paramBuf } },
    ];

    // Extra buffers (e.g., kernel for blur)
    const extraGPUBufs = [];
    for (let e = 0; e < extraBuffers.length; e++) {
      const data = extraBuffers[e](size);
      const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      queue.writeBuffer(buf, 0, data);
      entries.push({ binding: 3 + e, resource: { buffer: buf } });
      extraGPUBufs.push(buf);
    }

    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    if (entryPoint === 'blur_h' || entryPoint === 'blur_v') {
      pass.dispatchWorkgroups(Math.ceil(size / 256), size, 1);
    } else {
      pass.dispatchWorkgroups(Math.ceil(size / 16), Math.ceil(size / 16), 1);
    }
    pass.end();

    // Readback
    const stagingBuf = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, byteLen);
    queue.submit([enc.finish()]);
    await stagingBuf.mapAsync(GPUMapMode.READ);
    const _result = new Uint8Array(stagingBuf.getMappedRange()).slice();
    stagingBuf.unmap();

    const t1 = performance.now();

    inputBuf.destroy();
    outputBuf.destroy();
    paramBuf.destroy();
    stagingBuf.destroy();
    extraGPUBufs.forEach((b) => b.destroy());

    if (i >= 3) times.push(t1 - t0);
  }
  times.sort((a, b) => a - b);
  return { median: times[Math.floor(times.length / 2)], p95: times[Math.floor(times.length * 0.95)] };
}

async function benchGPUBlur(size) {
  const radius = 20;
  const sigma = radius / 3;
  const kernel = gaussianKernel(radius, sigma);

  // Two-pass separable blur: horizontal then vertical
  // For simplicity, benchmark single pass (horizontal)
  return benchGPUOp('gaussian_blur', 'blur_h', size, (s) => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, s, true);
    view.setUint32(4, s, true);
    view.setUint32(8, radius, true);
    view.setFloat32(12, sigma, true);
    return new Uint8Array(buf);
  }, [(_s) => kernel]);
}

async function benchGPUSpinBlur(size) {
  return benchGPUOp('spin_blur', 'main', size, (s) => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, s, true);
    view.setUint32(4, s, true);
    view.setFloat32(8, 0.5, true); // angle (radians)
    view.setUint32(12, 32, true);  // samples
    return new Uint8Array(buf);
  });
}

async function benchGPUSpherize(size) {
  return benchGPUOp('spherize', 'main', size, (s) => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, s, true);
    view.setUint32(4, s, true);
    view.setFloat32(8, 0.8, true); // strength
    view.setUint32(12, 0, true);   // padding
    return new Uint8Array(buf);
  });
}

async function benchGPUBilateral(size) {
  return benchGPUOp('bilateral', 'main', size, (s) => {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, s, true);
    view.setUint32(4, s, true);
    view.setUint32(8, 5, true);     // radius
    view.setFloat32(12, 10.0, true); // sigma_spatial
    view.setFloat32(16, 25.0, true); // sigma_range
    view.setUint32(20, 0, true);     // pad
    view.setUint32(24, 0, true);     // pad
    view.setUint32(28, 0, true);     // pad
    return new Uint8Array(buf);
  });
}

// ─── WASM Benchmarks (main thread — importmap provides bare specifier resolution) ──

// Call pipeline op — tries multiple calling conventions
function callWASMOp(pipe, src, opName, config) {
  const errors = [];

  // Try 1: config record (new SDK after codegen unification)
  try { return pipe[opName](src, config); }
  catch (e) { errors.push(`config-record: ${errMsg(e)}`); }

  // Try 2: positional args (old SDK)
  try {
    switch (opName) {
      case 'blur': return pipe.blur(src, config.radius);
      case 'spinBlur': return pipe.spinBlur(src, config.centerX || 0.5, config.centerY || 0.5, config.angle);
      case 'spherize': return pipe.spherize(src, config.strength);
      case 'bilateral': return pipe.bilateral(src, config.spatialSigma, config.rangeSigma, config.radius);
    }
  } catch (e) { errors.push(`positional: ${errMsg(e)}`); }

  throw new Error(`All calling conventions failed for ${opName}:\n  ${errors.join('\n  ')}`);
}

function errMsg(e) {
  if (typeof e === 'string') return e;
  // jco ComponentError: .payload has the actual error
  if (e && e.payload !== undefined) {
    const p = e.payload;
    if (typeof p === 'string') return p;
    if (p && p.message) return p.message;
    if (p && typeof p === 'object') {
      // WIT rasmcore-error variant
      const keys = Object.keys(p);
      return keys.map(k => `${k}: ${p[k]}`).join(', ') || String(p);
    }
    return String(p);
  }
  if (e && e.message) return e.message;
  try { return JSON.stringify(e, null, 2); } catch { return String(e); }
}

async function benchWASMOp(opName, size, config) {
  if (!Pipeline) return { median: -1, p95: -1 };

  const pixels = generateWASMTestImage(size);

  // Warm up
  for (let i = 0; i < 3; i++) {
    try {
      const pipe = new Pipeline();
      const src = pipe.read(pixels);
      callWASMOp(pipe, src, opName, config);
    } catch (e) {
      log(`WASM ${opName} warmup error: ${errMsg(e)}`);
      return { median: -1, p95: -1 };
    }
  }

  // Measure
  const times = [];
  for (let i = 0; i < 10; i++) {
    const pipe = new Pipeline();
    const src = pipe.read(pixels);
    const t0 = performance.now();
    try {
      const node = callWASMOp(pipe, src, opName, config);
      pipe.writePng(node, {}, undefined);
    } catch (e) {
      log(`WASM ${opName} error: ${errMsg(e)}`);
      return { median: -1, p95: -1 };
    }
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return { median: times[Math.floor(times.length / 2)], p95: times[Math.floor(times.length * 0.95)] };
}

// ─── Main ───────────────────────────────────────────────────────────────────

// Stateless filter benchmark — no pipeline, no tiling, full-image processing
async function benchWASMStateless(opName, size, config) {
  if (!Filters || !Filters[opName]) return { median: -1, p95: -1 };

  // Decode the test PNG to get raw pixels + info
  const pngBytes = generateWASMTestImage(size);
  let rawPixels, info;
  try {
    const pipe = new Pipeline();
    const src = pipe.read(pngBytes);
    info = pipe.nodeInfo(src);
    // Get raw pixels by writing as raw (or just use the pipeline to decode)
    // We'll use the decoded PNG via the pipeline nodeInfo
  } catch (e) {
    log(`Stateless ${opName} decode error: ${errMsg(e)}`);
    return { median: -1, p95: -1 };
  }

  // Call the stateless filter API directly
  const times = [];
  for (let i = 0; i < 13; i++) {
    const t0 = performance.now();
    try {
      Filters[opName](pngBytes, info, config);
    } catch (e) {
      if (i === 0) log(`Stateless ${opName} error: ${errMsg(e)}`);
      return { median: -1, p95: -1 };
    }
    if (i >= 3) times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return { median: times[Math.floor(times.length / 2)], p95: times[Math.floor(times.length * 0.95)] };
}

function renderResults(results) {
  const el = document.getElementById('results');
  let html = '<table><tr><th>Operation</th><th>Size</th><th>GPU (ms)</th><th>WASM Tiled (ms)</th><th>WASM Full (ms)</th><th>GPU vs Tiled</th><th>GPU vs Full</th></tr>';
  for (const r of results) {
    const speedupTiled = r.wasmTiled > 0 && r.gpu > 0 ? (r.wasmTiled / r.gpu).toFixed(1) + 'x' : 'N/A';
    const speedupFull = r.wasmFull > 0 && r.gpu > 0 ? (r.wasmFull / r.gpu).toFixed(1) + 'x' : 'N/A';
    const clsTiled = r.wasmTiled > 0 && r.gpu > 0 && r.gpu < r.wasmTiled ? 'faster' : 'slower';
    const clsFull = r.wasmFull > 0 && r.gpu > 0 && r.gpu < r.wasmFull ? 'faster' : 'slower';
    html += `<tr><td>${r.op}</td><td>${r.size}</td><td>${r.gpu.toFixed(1)}</td>`;
    html += `<td>${r.wasmTiled >= 0 ? r.wasmTiled.toFixed(1) : 'N/A'}</td>`;
    html += `<td>${r.wasmFull >= 0 ? r.wasmFull.toFixed(1) : 'N/A'}</td>`;
    html += `<td class="${clsTiled}">${speedupTiled}</td>`;
    html += `<td class="${clsFull}">${speedupFull}</td></tr>`;
  }
  html += '</table>';
  el.innerHTML = html;
}

async function runBenchmark() {
  const size = parseInt(document.getElementById('size-select').value);
  const results = [];

  log(`\n═══ Benchmark @ ${size}x${size} ═══`);

  // Transfer overhead
  status('Measuring transfer overhead...');
  const transfer = await benchTransferOnly(size);
  log(`Transfer only: ${transfer.median.toFixed(1)}ms (p95: ${transfer.p95.toFixed(1)}ms)`);
  results.push({ op: 'Transfer (upload+readback)', size: `${size}x${size}`, gpu: transfer.median, wasmTiled: 0, wasmFull: 0 });

  // Gaussian blur
  status('Benchmarking gaussian blur...');
  const gpuBlur = await benchGPUBlur(size);
  const wasmBlurTiled = await benchWASMOp('blur', size, { radius: 20.0 });
  const wasmBlurFull = await benchWASMStateless('blur', size, { radius: 20.0 });
  log(`Blur r=20: GPU=${gpuBlur.median.toFixed(1)}ms WASM-tiled=${wasmBlurTiled.median.toFixed(1)}ms WASM-full=${wasmBlurFull.median.toFixed(1)}ms`);
  results.push({ op: 'Gaussian Blur (r=20)', size: `${size}x${size}`, gpu: gpuBlur.median, wasmTiled: wasmBlurTiled.median, wasmFull: wasmBlurFull.median });

  // Spin blur
  status('Benchmarking spin blur...');
  const gpuSpin = await benchGPUSpinBlur(size);
  const wasmSpinTiled = await benchWASMOp('spinBlur', size, { angle: 0.5, samples: 32 });
  const wasmSpinFull = await benchWASMStateless('spinBlur', size, { angle: 0.5, samples: 32 });
  log(`Spin blur: GPU=${gpuSpin.median.toFixed(1)}ms WASM-tiled=${wasmSpinTiled.median.toFixed(1)}ms WASM-full=${wasmSpinFull.median.toFixed(1)}ms`);
  results.push({ op: 'Spin Blur (32 samples)', size: `${size}x${size}`, gpu: gpuSpin.median, wasmTiled: wasmSpinTiled.median, wasmFull: wasmSpinFull.median });

  // Spherize
  status('Benchmarking spherize...');
  const gpuSphere = await benchGPUSpherize(size);
  const wasmSphereTiled = await benchWASMOp('spherize', size, { strength: 0.8 });
  const wasmSphereFull = await benchWASMStateless('spherize', size, { strength: 0.8 });
  log(`Spherize: GPU=${gpuSphere.median.toFixed(1)}ms WASM-tiled=${wasmSphereTiled.median.toFixed(1)}ms WASM-full=${wasmSphereFull.median.toFixed(1)}ms`);
  results.push({ op: 'Spherize (0.8)', size: `${size}x${size}`, gpu: gpuSphere.median, wasmTiled: wasmSphereTiled.median, wasmFull: wasmSphereFull.median });

  // Bilateral
  status('Benchmarking bilateral filter...');
  const gpuBilateral = await benchGPUBilateral(size);
  const wasmBilateralTiled = await benchWASMOp('bilateral', size, { spatialSigma: 10.0, rangeSigma: 25.0, radius: 5 });
  const wasmBilateralFull = await benchWASMStateless('bilateral', size, { spatialSigma: 10.0, rangeSigma: 25.0, radius: 5 });
  log(`Bilateral r=5: GPU=${gpuBilateral.median.toFixed(1)}ms WASM-tiled=${wasmBilateralTiled.median.toFixed(1)}ms WASM-full=${wasmBilateralFull.median.toFixed(1)}ms`);
  results.push({ op: 'Bilateral (r=5)', size: `${size}x${size}`, gpu: gpuBilateral.median, wasmTiled: wasmBilateralTiled.median, wasmFull: wasmBilateralFull.median });

  renderResults(results);
  status(`Benchmark complete @ ${size}x${size}`);
}

async function runTransferOnly() {
  const size = parseInt(document.getElementById('size-select').value);
  status('Measuring transfer overhead...');
  const result = await benchTransferOnly(size);
  log(`Transfer ${size}x${size}: median=${result.median.toFixed(1)}ms p95=${result.p95.toFixed(1)}ms`);
  status('Done');
}

// ─── Init ───────────────────────────────────────────────────────────────────

(async () => {
  try {
    await initGPU();
    log('WebGPU initialized');

    // Load and compile shaders
    await loadShader('gaussian_blur');
    await loadShader('spin_blur');
    await loadShader('spherize');
    await loadShader('bilateral');
    log('Shaders compiled');

    // Pre-create pipelines
    getOrCreatePipeline('gaussian_blur', 'blur_h');
    getOrCreatePipeline('gaussian_blur', 'blur_v');
    getOrCreatePipeline('spin_blur', 'main');
    getOrCreatePipeline('spherize', 'main');
    getOrCreatePipeline('bilateral', 'main');
    log('Pipelines created');

    await initWASM();

    status('Ready — click Run Full Benchmark');
    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-transfer').disabled = false;
  } catch (e) {
    status(`Init failed: ${e.message}`, true);
    log(`ERROR: ${e.stack || e.message}`);
  }
})();

document.getElementById('btn-run').addEventListener('click', () => {
  document.getElementById('btn-run').disabled = true;
  runBenchmark().finally(() => {
    document.getElementById('btn-run').disabled = false;
  });
});

document.getElementById('btn-transfer').addEventListener('click', () => {
  runTransferOnly();
});
