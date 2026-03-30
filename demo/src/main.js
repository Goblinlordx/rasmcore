/**
 * rasmcore demo — v1 simple filter panel with before/after.
 *
 * Loads the jco-generated SDK, lets user drop an image, adjust filter sliders,
 * and see the result in real-time via the pipeline API.
 */

// ─── SDK Loading ────────────────────────────────────────────────────────────

const statusEl = document.getElementById('status');
const imageInfoEl = document.getElementById('image-info');
const timingEl = document.getElementById('timing');

let sdk = null;
let Pipeline = null;

try {
  sdk = await import('../sdk/rasmcore-image.js');
  Pipeline = sdk.pipeline.ImagePipeline;
  statusEl.textContent = 'SDK ready';
  statusEl.style.color = '#4ade80';

  // Auto-discover export formats from the WASM backend
  try {
    const writeFormats = sdk.pipeline.supportedWriteFormats();
    const formatSelect = document.getElementById('export-format');
    if (writeFormats && writeFormats.length > 0) {
      formatSelect.innerHTML = '';
      for (const fmt of writeFormats) {
        const opt = document.createElement('option');
        opt.value = fmt;
        opt.textContent = fmt.toUpperCase();
        if (fmt === 'jpeg') opt.selected = true;
        formatSelect.appendChild(opt);
      }
    }
  } catch (e) {
    console.warn('Could not auto-discover write formats:', e.message);
  }
} catch (e) {
  statusEl.textContent = `SDK failed: ${e.message}`;
  statusEl.style.color = '#f87171';
  console.error('SDK load error:', e);
}

// ─── State ──────────────────────────────────────────────────────────────────

let imageBytes = null;    // Original image as Uint8Array
let imageWidth = 0;
let imageHeight = 0;

// ─── Image Loading ──────────────────────────────────────────────────────────

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const controls = document.getElementById('controls');
const originalCanvas = document.getElementById('original');
const processedCanvas = document.getElementById('processed');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

async function loadFile(file) {
  const buffer = await file.arrayBuffer();
  imageBytes = new Uint8Array(buffer);

  // Show original on canvas
  const blob = new Blob([imageBytes], { type: file.type || 'image/png' });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    imageWidth = img.width;
    imageHeight = img.height;
    drawOnCanvas(originalCanvas, img);

    // Update info
    imageInfoEl.textContent = `${img.width}×${img.height} | ${(imageBytes.length / 1024).toFixed(0)} KB | ${file.name}`;
    dropZone.style.display = 'none';
    controls.style.display = 'block';

    // Set resize slider max
    document.getElementById('resize').max = img.width * 2;

    // Process with current settings
    processImage();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function drawOnCanvas(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.getContext('2d').drawImage(img, 0, 0);
}

// ─── Processing ─────────────────────────────────────────────────────────────

let debounceTimer = null;

function scheduleProcess() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(processImage, 100);
}

function processImage() {
  if (!imageBytes || !Pipeline) return;

  const blur = parseFloat(document.getElementById('blur').value);
  const sharpen = parseFloat(document.getElementById('sharpen').value);
  const brightness = parseFloat(document.getElementById('brightness').value);
  const contrast = parseFloat(document.getElementById('contrast').value);
  const resizeWidth = parseInt(document.getElementById('resize').value);

  const t0 = performance.now();

  try {
    const pipe = new Pipeline();
    let node = pipe.read(imageBytes);

    const timings = [];

    // Apply filters (skip if at default/zero)
    if (blur > 0) {
      const t = performance.now();
      node = pipe.blur(node, blur);
      timings.push(`blur ${(performance.now() - t).toFixed(0)}ms`);
    }
    if (sharpen > 0) {
      const t = performance.now();
      node = pipe.sharpen(node, sharpen);
      timings.push(`sharpen ${(performance.now() - t).toFixed(0)}ms`);
    }
    if (brightness !== 0) {
      const t = performance.now();
      node = pipe.brightness(node, brightness);
      timings.push(`brightness ${(performance.now() - t).toFixed(0)}ms`);
    }
    if (contrast !== 0) {
      const t = performance.now();
      node = pipe.contrast(node, contrast);
      timings.push(`contrast ${(performance.now() - t).toFixed(0)}ms`);
    }
    if (resizeWidth > 0 && resizeWidth !== imageWidth) {
      const t = performance.now();
      const ratio = resizeWidth / imageWidth;
      const resizeHeight = Math.round(imageHeight * ratio);
      node = pipe.resize(node, resizeWidth, resizeHeight, 'lanczos3');
      timings.push(`resize ${(performance.now() - t).toFixed(0)}ms`);
    }

    // Encode as PNG for preview
    const t = performance.now();
    const outputBytes = pipe.writePng(node, {}, undefined);
    timings.push(`encode ${(performance.now() - t).toFixed(0)}ms`);

    const totalMs = performance.now() - t0;
    timingEl.textContent = `${timings.join(' | ')} | total ${totalMs.toFixed(0)}ms`;

    // Display result
    const blob = new Blob([outputBytes], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      drawOnCanvas(processedCanvas, img);
      URL.revokeObjectURL(url);
    };
    img.src = url;

  } catch (e) {
    timingEl.textContent = `Error: ${e.message}`;
    timingEl.style.color = '#f87171';
    console.error('Processing error:', e);
  }
}

// ─── Slider Controls ────────────────────────────────────────────────────────

function bindSlider(id) {
  const input = document.getElementById(id);
  const valEl = document.getElementById(`${id}-val`);
  input.addEventListener('input', () => {
    valEl.textContent = input.value;
    scheduleProcess();
  });
}

bindSlider('blur');
bindSlider('sharpen');
bindSlider('brightness');
bindSlider('contrast');

document.getElementById('resize').addEventListener('input', (e) => {
  const val = parseInt(e.target.value);
  document.getElementById('resize-val').textContent = val > 0 ? `${val}px` : 'auto';
  scheduleProcess();
});

document.getElementById('quality').addEventListener('input', (e) => {
  document.getElementById('quality-val').textContent = e.target.value;
});

// ─── Download ───────────────────────────────────────────────────────────────

document.getElementById('download-btn').addEventListener('click', () => {
  if (!imageBytes || !Pipeline) return;

  const format = document.getElementById('export-format').value;
  const quality = parseInt(document.getElementById('quality').value);

  const pipe = new Pipeline();
  let node = pipe.read(imageBytes);

  // Apply current filters
  const blur = parseFloat(document.getElementById('blur').value);
  const sharpen = parseFloat(document.getElementById('sharpen').value);
  const brightness = parseFloat(document.getElementById('brightness').value);
  const contrast = parseFloat(document.getElementById('contrast').value);
  const resizeWidth = parseInt(document.getElementById('resize').value);

  if (blur > 0) node = pipe.blur(node, blur);
  if (sharpen > 0) node = pipe.sharpen(node, sharpen);
  if (brightness !== 0) node = pipe.brightness(node, brightness);
  if (contrast !== 0) node = pipe.contrast(node, contrast);
  if (resizeWidth > 0 && resizeWidth !== imageWidth) {
    const ratio = resizeWidth / imageWidth;
    node = pipe.resize(node, resizeWidth, Math.round(imageHeight * ratio), 'lanczos3');
  }

  // Encode in selected format using generic write() — supports all backend formats
  const MIME_MAP = {
    jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp', gif: 'image/gif',
    tiff: 'image/tiff', avif: 'image/avif', bmp: 'image/bmp', ico: 'image/x-icon',
    qoi: 'application/octet-stream', heic: 'image/heic', tga: 'application/octet-stream',
    hdr: 'application/octet-stream', pnm: 'application/octet-stream',
    exr: 'application/octet-stream', dds: 'application/octet-stream',
    jp2: 'image/jp2', fits: 'application/octet-stream',
  };
  const outputBytes = pipe.write(node, format, quality > 0 ? quality : undefined, undefined);
  const mimeType = MIME_MAP[format] || 'application/octet-stream';

  // Trigger download
  const blob = new Blob([outputBytes], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `rasmcore-output.${format}`;
  a.click();
  URL.revokeObjectURL(url);
});

// ─── Reset ──────────────────────────────────────────────────────────────────

document.getElementById('reset-btn').addEventListener('click', () => {
  document.getElementById('blur').value = 0;
  document.getElementById('sharpen').value = 0;
  document.getElementById('brightness').value = 0;
  document.getElementById('contrast').value = 0;
  document.getElementById('resize').value = 0;
  document.getElementById('blur-val').textContent = '0';
  document.getElementById('sharpen-val').textContent = '0';
  document.getElementById('brightness-val').textContent = '0';
  document.getElementById('contrast-val').textContent = '0';
  document.getElementById('resize-val').textContent = 'auto';
  processImage();
});
