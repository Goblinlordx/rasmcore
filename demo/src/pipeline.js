/**
 * rasmcore pipeline builder — visual chain editor with auto-discovered operations.
 *
 * Operations are discovered from the ImagePipeline class methods at runtime.
 * No hard-coded filter list — new #[register_filter] operations appear automatically.
 */

// ─── SDK + Operation Discovery ──────────────────────────────────────────────

let sdk, Pipeline;
try {
  sdk = await import('../sdk/rasmcore-image.js');
  Pipeline = sdk.pipeline.ImagePipeline;
} catch (e) {
  document.getElementById('palette-items').textContent = `SDK failed: ${e.message}`;
  throw e;
}

// Auto-discover operations from ImagePipeline prototype
const SKIP = new Set(['constructor', 'read', 'nodeInfo', 'composite']);
const WRITE_PREFIX = 'write';

// Operation metadata — auto-derived from method signatures
const operations = [];
const pipe = new Pipeline();
const proto = Object.getPrototypeOf(pipe);
const methodNames = Object.getOwnPropertyNames(proto).filter(n => typeof proto[n] === 'function' && !SKIP.has(n) && !n.startsWith(WRITE_PREFIX));

// Define param metadata for known operations (extensible via JSON manifest in future)
const PARAM_META = {
  blur:        [{ name: 'radius', type: 'number', min: 0, max: 50, step: 0.5, default: 3 }],
  sharpen:     [{ name: 'amount', type: 'number', min: 0, max: 10, step: 0.1, default: 1 }],
  brightness:  [{ name: 'amount', type: 'number', min: -1, max: 1, step: 0.05, default: 0 }],
  contrast:    [{ name: 'amount', type: 'number', min: -1, max: 1, step: 0.05, default: 0 }],
  grayscale:   [],
  sobel:       [],
  canny:       [{ name: 'lowThreshold', type: 'number', min: 0, max: 255, step: 1, default: 50 },
                { name: 'highThreshold', type: 'number', min: 0, max: 255, step: 1, default: 150 }],
  median:      [{ name: 'radius', type: 'number', min: 1, max: 20, step: 1, default: 3 }],
  resize:      [{ name: 'width', type: 'number', min: 1, max: 4000, step: 1, default: 800 },
                { name: 'height', type: 'number', min: 1, max: 4000, step: 1, default: 600 },
                { name: 'filter', type: 'enum', options: ['nearest', 'bilinear', 'bicubic', 'lanczos3'], default: 'lanczos3' }],
  crop:        [{ name: 'x', type: 'number', min: 0, max: 4000, step: 1, default: 0 },
                { name: 'y', type: 'number', min: 0, max: 4000, step: 1, default: 0 },
                { name: 'width', type: 'number', min: 1, max: 4000, step: 1, default: 256 },
                { name: 'height', type: 'number', min: 1, max: 4000, step: 1, default: 256 }],
  rotate:      [{ name: 'angle', type: 'enum', options: ['r90', 'r180', 'r270'], default: 'r90' }],
  flip:        [{ name: 'direction', type: 'enum', options: ['horizontal', 'vertical'], default: 'horizontal' }],
};

// Categorize operations
const CATEGORIES = {
  'Filters':    ['blur', 'sharpen', 'brightness', 'contrast', 'grayscale', 'median'],
  'Edge':       ['sobel', 'canny'],
  'Transform':  ['resize', 'crop', 'rotate', 'flip'],
  'Color':      ['convertFormat', 'autoOrient', 'iccToSrgb'],
};

// Build discovered operations list
for (const name of methodNames) {
  const params = PARAM_META[name] || [];
  let category = 'Other';
  for (const [cat, ops] of Object.entries(CATEGORIES)) {
    if (ops.includes(name)) { category = cat; break; }
  }
  operations.push({ name, category, params });
}

// ─── Palette UI ─────────────────────────────────────────────────────────────

const paletteEl = document.getElementById('palette-items');
paletteEl.innerHTML = '';

// Group by category
const grouped = {};
for (const op of operations) {
  if (!grouped[op.category]) grouped[op.category] = [];
  grouped[op.category].push(op);
}

for (const [cat, ops] of Object.entries(grouped)) {
  const h3 = document.createElement('h3');
  h3.textContent = cat;
  paletteEl.parentElement.appendChild(h3);
  for (const op of ops) {
    const div = document.createElement('div');
    div.className = 'palette-item';
    div.textContent = op.name;
    div.addEventListener('click', () => addNode(op));
    paletteEl.parentElement.appendChild(div);
  }
}
paletteEl.remove(); // remove the loading placeholder

// ─── Chain State ────────────────────────────────────────────────────────────

let imageBytes = null;
let chain = []; // Array of { op, paramValues, id, timingMs }
let nextId = 0;

const chainEl = document.getElementById('chain');
const dropHint = document.getElementById('drop-hint');
const previewCanvas = document.getElementById('preview-canvas');
const previewInfo = document.getElementById('preview-info');
const totalTimeEl = document.getElementById('total-time');

// ─── Image Loading ──────────────────────────────────────────────────────────

const fileInput = document.getElementById('file-input');

// Make preview area a drop zone
document.querySelector('.preview').addEventListener('click', () => {
  if (!imageBytes) fileInput.click();
});
document.querySelector('.preview').addEventListener('dragover', (e) => e.preventDefault());
document.querySelector('.preview').addEventListener('drop', (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

async function loadFile(file) {
  imageBytes = new Uint8Array(await file.arrayBuffer());
  previewInfo.textContent = `${file.name} | ${(imageBytes.length / 1024).toFixed(0)} KB`;
  dropHint.style.display = 'block';
  processChain();
}

// ─── Node Management ────────────────────────────────────────────────────────

function addNode(op) {
  if (!imageBytes) {
    fileInput.click();
    return;
  }
  const node = {
    id: nextId++,
    op,
    paramValues: Object.fromEntries(op.params.map(p => [p.name, p.default])),
    timingMs: 0,
  };
  chain.push(node);
  renderChain();
  processChain();
}

function removeNode(id) {
  chain = chain.filter(n => n.id !== id);
  renderChain();
  processChain();
}

function renderChain() {
  chainEl.innerHTML = '';
  for (const node of chain) {
    const card = document.createElement('div');
    card.className = 'node-card';
    card.dataset.id = node.id;

    // Header
    const header = document.createElement('div');
    header.className = 'node-header';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'node-name';
    nameSpan.textContent = node.op.name;
    const timingSpan = document.createElement('span');
    timingSpan.className = 'node-timing';
    timingSpan.textContent = node.timingMs > 0 ? `${node.timingMs}ms` : '';
    const removeBtn = document.createElement('span');
    removeBtn.className = 'node-remove';
    removeBtn.textContent = '✕';
    removeBtn.addEventListener('click', () => removeNode(node.id));
    header.appendChild(nameSpan);
    header.appendChild(timingSpan);
    header.appendChild(removeBtn);
    card.appendChild(header);

    // Params
    for (const p of node.op.params) {
      const paramDiv = document.createElement('div');
      paramDiv.className = 'node-param';

      if (p.type === 'number') {
        const label = document.createElement('label');
        const valSpan = document.createElement('span');
        valSpan.textContent = node.paramValues[p.name];
        label.textContent = p.name + ' ';
        label.appendChild(valSpan);
        paramDiv.appendChild(label);

        const input = document.createElement('input');
        input.type = 'range';
        input.min = p.min;
        input.max = p.max;
        input.step = p.step;
        input.value = node.paramValues[p.name];
        input.addEventListener('input', () => {
          node.paramValues[p.name] = parseFloat(input.value);
          valSpan.textContent = input.value;
          scheduleProcess();
        });
        paramDiv.appendChild(input);
      } else if (p.type === 'enum') {
        const label = document.createElement('label');
        label.textContent = p.name;
        paramDiv.appendChild(label);

        const select = document.createElement('select');
        for (const opt of p.options) {
          const option = document.createElement('option');
          option.value = opt;
          option.textContent = opt;
          if (opt === node.paramValues[p.name]) option.selected = true;
          select.appendChild(option);
        }
        select.addEventListener('change', () => {
          node.paramValues[p.name] = select.value;
          scheduleProcess();
        });
        paramDiv.appendChild(select);
      }

      card.appendChild(paramDiv);
    }

    chainEl.appendChild(card);
  }
}

// ─── Processing ─────────────────────────────────────────────────────────────

let debounceTimer = null;
function scheduleProcess() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(processChain, 150);
}

function processChain() {
  if (!imageBytes) return;

  const t0 = performance.now();

  try {
    const pipe = new Pipeline();
    let node = pipe.read(imageBytes);

    for (const chainNode of chain) {
      const t = performance.now();
      const args = chainNode.op.params.map(p => chainNode.paramValues[p.name]);
      node = pipe[chainNode.op.name](node, ...args);
      chainNode.timingMs = Math.round(performance.now() - t);
    }

    // Encode as PNG for preview
    const output = pipe.writePng(node, {}, undefined);
    const totalMs = Math.round(performance.now() - t0);
    totalTimeEl.textContent = `Total: ${totalMs}ms (${chain.length} operations)`;

    // Update per-node timing display
    for (const chainNode of chain) {
      const card = chainEl.querySelector(`[data-id="${chainNode.id}"]`);
      if (card) {
        const timing = card.querySelector('.node-timing');
        if (timing) timing.textContent = `${chainNode.timingMs}ms`;
      }
    }

    // Display
    const blob = new Blob([output], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      previewCanvas.width = img.width;
      previewCanvas.height = img.height;
      previewCanvas.getContext('2d').drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  } catch (e) {
    totalTimeEl.textContent = `Error: ${e.message}`;
    totalTimeEl.style.color = '#f87171';
    console.error(e);
  }
}

// ─── Export ──────────────────────────────────────────────────────────────────

document.getElementById('quality').addEventListener('input', (e) => {
  document.getElementById('quality-val').textContent = e.target.value;
});

document.getElementById('download-btn').addEventListener('click', () => {
  if (!imageBytes) return;

  const format = document.getElementById('export-format').value;
  const quality = parseInt(document.getElementById('quality').value);

  const pipe = new Pipeline();
  let node = pipe.read(imageBytes);
  for (const chainNode of chain) {
    const args = chainNode.op.params.map(p => chainNode.paramValues[p.name]);
    node = pipe[chainNode.op.name](node, ...args);
  }

  let output, mime;
  if (format === 'jpeg') {
    output = pipe.writeJpeg(node, { quality }, undefined);
    mime = 'image/jpeg';
  } else if (format === 'webp') {
    output = pipe.writeWebp(node, { quality }, undefined);
    mime = 'image/webp';
  } else {
    output = pipe.writePng(node, {}, undefined);
    mime = 'image/png';
  }

  const blob = new Blob([output], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `rasmcore-pipeline.${format}`;
  a.click();
});
