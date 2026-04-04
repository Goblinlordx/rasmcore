/**
 * Lazy singleton loader for V2 WASM pipeline module.
 *
 * Loads the 13MB WASM module only when first requested (not on page load).
 * Returns the PipelineClass constructor for use with Pipeline.fromRaw().
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadPromise: Promise<any> | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pipelineClass: any = null;

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
 * Apply a single filter to image bytes and return the result as a PNG ArrayBuffer.
 */
export async function renderFilter(
  imageBytes: Uint8Array,
  filterName: string,
  params: Record<string, number | boolean>,
): Promise<ArrayBuffer> {
  const PipelineClass = await loadWasm();
  const pipe = new PipelineClass();
  const nodeId = pipe.read(imageBytes, undefined);

  // Serialize params to binary format expected by applyFilter
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
    // Push name length + name bytes
    buf.push(name.length);
    for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));

    if (typeof value === 'boolean') {
      buf.push(2); // bool type
      buf.push(value ? 1 : 0);
    } else if (Number.isInteger(value)) {
      buf.push(1); // u32 type
      const ab = new ArrayBuffer(4);
      new DataView(ab).setUint32(0, value as number, true);
      buf.push(...new Uint8Array(ab));
    } else {
      buf.push(0); // f32 type
      const ab = new ArrayBuffer(4);
      new DataView(ab).setFloat32(0, value as number, true);
      buf.push(...new Uint8Array(ab));
    }
  }
  return new Uint8Array(buf);
}
