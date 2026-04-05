/**
 * Lazy singleton loader for V2 WASM pipeline module.
 *
 * Uses Source objects for decode caching — the reference image is decoded
 * once and reused across all subsequent renders. Returns f32 pixels for
 * canvas rendering (no PNG encode on slider changes).
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadPromise: Promise<any> | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pipelineClass: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let sourceClass: any = null;

// Source cache: keyed by reference image URL to avoid re-decode
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sourceCache = new Map<string, any>();

export function isLoaded(): boolean {
  return pipelineClass !== null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function loadWasm(): Promise<any> {
  if (pipelineClass) return pipelineClass;

  if (!loadPromise) {
    loadPromise = (async () => {
      try {
        // Dynamic import of the jco-generated SDK from public/sdk/v2/
        // @ts-expect-error — runtime-only import from public directory, not a TS module
        const sdk = await import(/* webpackIgnore: true */ '/sdk/v2/rasmcore-v2-image.js');
        pipelineClass = sdk.pipelineV2.ImagePipelineV2;
        sourceClass = sdk.pipelineV2.Source;
        return pipelineClass;
      } catch (e) {
        loadPromise = null; // Allow retry on failure
        throw e;
      }
    })();
  }

  return loadPromise;
}

/**
 * Get or create a cached Source for the given image bytes.
 * The Source decodes once; subsequent calls return the cached instance.
 */
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

/**
 * Render result — f32 RGBA pixels + dimensions for canvas display.
 */
export interface RenderResult {
  pixels: Float32Array;
  width: number;
  height: number;
}

/**
 * Apply a single filter to image bytes and return f32 pixels for canvas rendering.
 * Uses Source caching — the image is decoded once per unique cacheKey.
 */
export async function renderFilterToPixels(
  imageBytes: Uint8Array,
  filterName: string,
  params: Record<string, number | boolean>,
  cacheKey: string,
): Promise<RenderResult> {
  const t0 = performance.now();

  const PipelineClass = await loadWasm();
  const tWasm = performance.now();

  const pipe = new PipelineClass();
  const tPipe = performance.now();

  // Use Source (decode cached) or fall back to read(bytes)
  const source = getOrCreateSource(imageBytes, cacheKey);
  let nodeId: number;
  let usedSource = false;
  if (source && typeof pipe.readSource === 'function') {
    nodeId = pipe.readSource(source);
    usedSource = true;
  } else {
    nodeId = pipe.read(imageBytes, undefined);
  }
  const tRead = performance.now();

  const paramBuf = serializeParams(filterName, params);
  const filterId = pipe.applyFilter(nodeId, filterName, paramBuf);
  const tFilter = performance.now();

  // Get f32 pixels instead of PNG — faster for canvas display
  const pixels = pipe.render(filterId);
  const tRender = performance.now();

  const info = pipe.nodeInfo(filterId);
  const f32 = pixels instanceof Float32Array ? pixels : new Float32Array(pixels);
  const tTotal = performance.now();

  console.log(
    `[playground] ${filterName}: total=${(tTotal - t0).toFixed(1)}ms ` +
    `(wasm=${(tWasm - t0).toFixed(1)} pipe=${(tPipe - tWasm).toFixed(1)} ` +
    `read=${(tRead - tPipe).toFixed(1)}${usedSource ? '[Source]' : '[bytes]'} ` +
    `filter=${(tFilter - tRead).toFixed(1)} render=${(tRender - tFilter).toFixed(1)} ` +
    `copy=${(tTotal - tRender).toFixed(1)}) ${info.width}x${info.height}`
  );

  return { pixels: f32, width: info.width, height: info.height };
}

/**
 * Legacy: apply filter and return PNG bytes.
 */
export async function renderFilter(
  imageBytes: Uint8Array,
  filterName: string,
  params: Record<string, number | boolean>,
): Promise<ArrayBuffer> {
  const PipelineClass = await loadWasm();
  const pipe = new PipelineClass();
  const nodeId = pipe.read(imageBytes, undefined);
  const paramBuf = serializeParams(filterName, params);
  const filterId = pipe.applyFilter(nodeId, filterName, paramBuf);
  const output = pipe.write(filterId, 'png', undefined);
  return output.buffer.slice(output.byteOffset, output.byteOffset + output.byteLength);
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
