/**
 * Lazy singleton loader for V2 WASM pipeline module.
 *
 * Uses Source objects for decode caching and RenderTarget for GPU-direct
 * canvas rendering. No f32→u8 quantize loop — GPU blits directly to canvas.
 */

// @ts-expect-error — SDK lib import
import { GpuHandlerV2 } from '../../../../sdk/v2/lib/gpu-handler';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadPromise: Promise<any> | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pipelineClass: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let sourceClass: any = null;

// Source cache: keyed by reference image URL to avoid re-decode
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sourceCache = new Map<string, any>();

// Persistent GPU handler for canvas rendering (shared across all playgrounds)
let gpuHandler: InstanceType<typeof GpuHandlerV2> | null = null;
let gpuInitAttempted = false;

export function isLoaded(): boolean {
  return pipelineClass !== null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function loadWasm(): Promise<any> {
  if (pipelineClass) return pipelineClass;

  if (!loadPromise) {
    loadPromise = (async () => {
      try {
        // @ts-expect-error — runtime-only import from public directory
        const sdk = await import(/* webpackIgnore: true */ '/sdk/v2/rasmcore-v2-image.js');
        pipelineClass = sdk.pipelineV2.ImagePipelineV2;
        sourceClass = sdk.pipelineV2.Source;
        return pipelineClass;
      } catch (e) {
        loadPromise = null;
        throw e;
      }
    })();
  }

  return loadPromise;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getOrCreateSource(imageBytes: Uint8Array, cacheKey: string): any {
  if (sourceCache.has(cacheKey)) return sourceCache.get(cacheKey);
  if (sourceClass) {
    const source = new sourceClass(imageBytes, undefined);
    sourceCache.set(cacheKey, source);
    return source;
  }
  return null;
}

/** Ensure GPU handler is initialized (once). */
async function ensureGpu(): Promise<boolean> {
  if (gpuHandler) return true;
  if (gpuInitAttempted) return false;
  gpuInitAttempted = true;
  if (typeof GpuHandlerV2 !== 'undefined' && GpuHandlerV2.isAvailable()) {
    try {
      gpuHandler = new GpuHandlerV2();
      return true;
    } catch { /* GPU init failed */ }
  }
  return false;
}

/**
 * Render a filter to a canvas — GPU direct when available, 2D fallback otherwise.
 *
 * GPU path: WASM produces shader chain → GPU dispatches → blits to canvas.
 * No f32 crosses the boundary. No JS quantize loop.
 *
 * 2D fallback: WASM renders f32 → JS quantizes to u8 → putImageData.
 */
export async function renderFilterToCanvas(
  canvas: HTMLCanvasElement,
  imageBytes: Uint8Array,
  filterName: string,
  params: Record<string, number | boolean>,
  cacheKey: string,
): Promise<{ width: number; height: number }> {
  const t0 = performance.now();

  const PipelineClass = await loadWasm();
  const pipe = new PipelineClass();
  if (typeof pipe.setTracing === 'function') pipe.setTracing(true);

  // Source-cached read
  const source = getOrCreateSource(imageBytes, cacheKey);
  let nodeId: number;
  if (source && typeof pipe.readSource === 'function') {
    nodeId = pipe.readSource(source);
  } else {
    nodeId = pipe.read(imageBytes, undefined);
  }
  const tRead = performance.now();

  const paramBuf = serializeParams(filterName, params);
  const filterId = pipe.applyFilter(nodeId, filterName, paramBuf);
  const tFilter = performance.now();

  const info = pipe.nodeInfo(filterId);
  canvas.width = info.width;
  canvas.height = info.height;

  // Try GPU path: get shader plan, dispatch on canvas
  const hasGpu = await ensureGpu();
  let gpuRendered = false;

  if (hasGpu && gpuHandler && typeof pipe.renderGpuPlan === 'function') {
    try {
      const plan = pipe.renderGpuPlan(filterId);
      if (plan) {
        // Configure GPU canvas if not already done
        if (!gpuHandler.hasDisplay) {
          await gpuHandler.setDisplayCanvas(canvas, false);
        }
        gpuHandler.updateViewport(0, 0, 1.0, info.width, info.height, info.width, info.height, 0);

        const ops = plan.shaders.map((s: any) => ({
          source: s.source,
          entryPoint: s.entryPoint,
          workgroupX: s.workgroupX,
          workgroupY: s.workgroupY,
          workgroupZ: s.workgroupZ,
          params: new Uint8Array(s.params),
          extraBuffers: s.extraBuffers.map((b: any) => new Uint8Array(b)),
        }));

        await gpuHandler.prepare(ops);
        const err = await gpuHandler.executeAndDisplay(
          ops, new Float32Array(plan.inputPixels), plan.width, plan.height,
        );
        if (!err) gpuRendered = true;
      }
    } catch { /* GPU failed — fall through to CPU */ }
  }

  const tGpu = performance.now();

  // CPU fallback: render f32, quantize to u8, putImageData
  if (!gpuRendered) {
    const pixels = pipe.render(filterId);
    const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
    const len = f32.length;
    const u8 = new Uint8ClampedArray(len);
    for (let i = 0; i < len; i++) {
      u8[i] = f32[i] * 255;
    }
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.putImageData(new ImageData(u8, info.width, info.height), 0, 0);
  }

  const tTotal = performance.now();

  // Trace
  if (typeof pipe.takeTrace === 'function') {
    try {
      const trace = pipe.takeTrace();
      if (trace && trace.length > 0) {
        const parts = trace.map((e: any) =>
          `${e.kind}:${e.name}=${(e.durationUs / 1000).toFixed(1)}ms${e.detail ? ` (${e.detail})` : ''}`
        );
        console.log(`[playground] WASM trace: ${parts.join(' | ')}`);
      }
    } catch { /* ignore */ }
  }

  console.log(
    `[playground] ${filterName}: total=${(tTotal - t0).toFixed(1)}ms ` +
    `(read=${(tRead - t0).toFixed(1)} filter=${(tFilter - tRead).toFixed(1)} ` +
    `${gpuRendered ? 'gpu' : 'cpu'}=${(tGpu - tFilter).toFixed(1)} ` +
    `${gpuRendered ? '' : `quantize=${(tTotal - tGpu).toFixed(1)} `}` +
    `${info.width}x${info.height})`
  );

  return { width: info.width, height: info.height };
}

function serializeParams(
  _name: string,
  params: Record<string, number | boolean>,
): Uint8Array {
  const entries = Object.entries(params);
  if (entries.length === 0) return new Uint8Array(0);

  const buf: number[] = [];
  for (const [name, value] of entries) {
    buf.push(name.length);
    for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
    if (typeof value === 'boolean') {
      buf.push(2);
      buf.push(value ? 1 : 0);
    } else if (Number.isInteger(value)) {
      buf.push(1);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setUint32(0, value as number, true);
      buf.push(...new Uint8Array(ab));
    } else {
      buf.push(0);
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, value as number, true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}
