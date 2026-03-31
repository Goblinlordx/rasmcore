/**
 * @rasmcore/sdk — Dynamic runtime for any rasmcore-compatible WASM module.
 *
 * Loads a WASM module, reads its manifest via getFilterManifest(),
 * and provides dynamic dispatch for all operations the module supports.
 *
 * Usage:
 *   const img = RcImage.load(wasmModule, pngBytes);
 *   const jpeg = img.apply('blur', { radius: 3.0 }).encode('jpeg', { quality: 85 });
 *
 * The runtime is module-agnostic — it works with any WASM component
 * that exposes getFilterManifest() and getManifestHash().
 */

import type {
  FilterManifest,
  OperationMeta,
  ParamMeta,
  WasmModule,
  WasmPipeline,
} from './types.js';

/**
 * Convert a snake_case or kebab-case name to camelCase.
 * WASM WIT uses kebab-case, jco transpiles to camelCase.
 */
function toCamelCase(name: string): string {
  return name.replace(/[-_](\w)/g, (_, c) => c.toUpperCase());
}

/**
 * Validate a parameter value against its manifest metadata.
 */
function validateParam(meta: ParamMeta, name: string, value: unknown): void {
  if (value === undefined || value === null) return;

  const t = meta.type;
  if (
    (t === 'f32' || t === 'f64' || t === 'u32' || t === 'u16' || t === 'u8' || t === 'i32') &&
    typeof value !== 'number'
  ) {
    throw new TypeError(`${name}: expected number, got ${typeof value}`);
  }
  if (t === 'bool' && typeof value !== 'boolean') {
    throw new TypeError(`${name}: expected boolean, got ${typeof value}`);
  }
  if ((t === 'string' || t === 'String') && typeof value !== 'string') {
    throw new TypeError(`${name}: expected string, got ${typeof value}`);
  }

  if (typeof value === 'number') {
    if (meta.min !== null && value < meta.min) {
      throw new RangeError(`${name}: ${value} is below minimum ${meta.min}`);
    }
    if (meta.max !== null && value > meta.max) {
      throw new RangeError(`${name}: ${value} is above maximum ${meta.max}`);
    }
  }
}

/**
 * Resolve the pipeline constructor and introspection functions from a WASM module.
 *
 * Modules may have different namespace layouts depending on how they were
 * transpiled. This function searches common patterns.
 */
function resolvePipeline(
  module: WasmModule
): {
  createPipeline: () => WasmPipeline;
  getManifest: () => string;
  getManifestHash: () => string;
  getSupportedWriteFormats: () => string[];
} {
  // Search for pipeline namespace — common patterns from jco transpile
  const candidates = ['pipeline', 'rasmcoreImagePipeline', 'image'];
  let ns: Record<string, unknown> | undefined;

  for (const key of candidates) {
    const val = (module as Record<string, unknown>)[key];
    if (val && typeof val === 'object') {
      ns = val as Record<string, unknown>;
      break;
    }
  }

  // Fallback: search all top-level keys for one that has getFilterManifest
  if (!ns) {
    for (const key of Object.keys(module)) {
      const val = (module as Record<string, unknown>)[key];
      if (val && typeof val === 'object' && 'getFilterManifest' in (val as object)) {
        ns = val as Record<string, unknown>;
        break;
      }
    }
  }

  if (!ns) {
    throw new Error(
      'Could not find pipeline namespace in WASM module. ' +
      'Expected a namespace with getFilterManifest() method.'
    );
  }

  // Find the pipeline constructor (ImagePipeline or similar)
  let PipelineCtor: (new () => WasmPipeline) | undefined;
  for (const key of Object.keys(ns)) {
    const val = ns[key];
    if (typeof val === 'function' && /pipeline/i.test(key)) {
      PipelineCtor = val as new () => WasmPipeline;
      break;
    }
  }

  if (!PipelineCtor) {
    throw new Error('Could not find Pipeline constructor in WASM module namespace.');
  }

  const getManifest = ns['getFilterManifest'] as (() => string) | undefined;
  if (!getManifest || typeof getManifest !== 'function') {
    throw new Error('WASM module does not expose getFilterManifest().');
  }

  const getManifestHash = (ns['getManifestHash'] as (() => string) | undefined) ?? (() => '');
  const getSupportedWriteFormats =
    (ns['supportedWriteFormats'] as (() => string[]) | undefined) ?? (() => []);

  return {
    createPipeline: () => new PipelineCtor!(),
    getManifest: getManifest as () => string,
    getManifestHash: getManifestHash as () => string,
    getSupportedWriteFormats: getSupportedWriteFormats as () => string[],
  };
}

/**
 * Dynamic image processing chain.
 *
 * Works with any rasmcore-compatible WASM module. Discovers available
 * operations at load time via the module's embedded manifest.
 *
 * All operations are lazy — pixels are only computed when an encode
 * method is called.
 */
export class RcImage {
  private _pipeline: WasmPipeline;
  private _nodeId: number;
  private _manifest: FilterManifest;
  private _manifestHash: string;
  private _operationIndex: Map<string, OperationMeta>;
  private _writeFormats: string[];

  /** @internal Use RcImage.load() instead. */
  constructor(
    pipeline: WasmPipeline,
    nodeId: number,
    manifest: FilterManifest,
    manifestHash: string,
    writeFormats: string[],
  ) {
    this._pipeline = pipeline;
    this._nodeId = nodeId;
    this._manifest = manifest;
    this._manifestHash = manifestHash;
    this._writeFormats = writeFormats;

    // Build a lookup index for O(1) operation discovery
    this._operationIndex = new Map();
    for (const op of manifest.filters) {
      this._operationIndex.set(op.name, op);
    }
  }

  /**
   * Load image data from a WASM module.
   *
   * @param module - The imported WASM module (from jco transpile or similar)
   * @param data - Raw image file bytes (PNG, JPEG, WebP, etc.)
   */
  static load(module: WasmModule, data: Uint8Array): RcImage {
    const resolved = resolvePipeline(module);
    const pipeline = resolved.createPipeline();
    const manifest: FilterManifest = JSON.parse(resolved.getManifest());
    const manifestHash = resolved.getManifestHash();
    const writeFormats = resolved.getSupportedWriteFormats();
    const nodeId = (pipeline as any).read(data) as number;
    return new RcImage(pipeline, nodeId, manifest, manifestHash, writeFormats);
  }

  /** The full manifest describing all operations this module supports. */
  get manifest(): FilterManifest {
    return this._manifest;
  }

  /** The manifest content hash — for SDK version validation. */
  get manifestHash(): string {
    return this._manifestHash;
  }

  /** Available output formats (from registered encoders). */
  get writeFormats(): string[] {
    return this._writeFormats;
  }

  /** Image info (dimensions, format) without computing pixels. */
  get info(): Record<string, unknown> {
    return (this._pipeline as any).nodeInfo(this._nodeId);
  }

  /** List all available filter/operation names. */
  get availableOperations(): string[] {
    return Array.from(this._operationIndex.keys());
  }

  /** Get metadata for a specific operation (params, ranges, defaults). */
  operationMeta(name: string): OperationMeta | undefined {
    return this._operationIndex.get(name);
  }

  /** Branch the pipeline — returns a new RcImage sharing the same graph. */
  fork(): RcImage {
    return new RcImage(
      this._pipeline,
      this._nodeId,
      this._manifest,
      this._manifestHash,
      this._writeFormats,
    );
  }

  /**
   * Apply a named operation with parameters.
   *
   * @param name - Operation name (snake_case, as in the manifest)
   * @param params - Parameter values keyed by parameter name
   * @returns this (for chaining)
   *
   * @example
   *   img.apply('blur', { radius: 3.0 })
   *   img.apply('brightness', { amount: 0.1 })
   *   img.apply('resize', { width: 800, height: 600 })
   */
  apply(name: string, params: Record<string, unknown> = {}): this {
    const meta = this._operationIndex.get(name);
    if (!meta) {
      const available = this.availableOperations.join(', ');
      throw new Error(
        `Unknown operation "${name}". Available: ${available}`
      );
    }

    // Validate params
    for (const pmeta of meta.params) {
      const value = params[pmeta.name];
      if (value !== undefined) {
        validateParam(pmeta, `${name}.${pmeta.name}`, value);
      }
    }

    // Build ordered args: nodeId, then params in manifest order
    // Use default values for missing params
    const args: unknown[] = [this._nodeId];
    for (const pmeta of meta.params) {
      const value = params[pmeta.name] ?? pmeta.default ?? undefined;
      args.push(value);
    }

    // Dispatch to the pipeline method
    // jco transpiles WIT kebab-case to camelCase
    const methodName = toCamelCase(name);
    const method = this._pipeline[methodName];
    if (typeof method !== 'function') {
      throw new Error(
        `Pipeline method "${methodName}" not found. ` +
        `The WASM module may not implement this operation.`
      );
    }

    this._nodeId = method.call(this._pipeline, ...args) as number;
    return this;
  }

  /**
   * Encode the current image to a format.
   *
   * @param format - Output format name (e.g., 'jpeg', 'png', 'webp')
   * @param config - Format-specific configuration (e.g., { quality: 85 })
   * @returns Encoded bytes
   *
   * @example
   *   const jpeg = img.encode('jpeg', { quality: 85 });
   *   const png = img.encode('png', { compressionLevel: 6 });
   */
  encode(format: string, config: Record<string, unknown> = {}): Uint8Array {
    // Try format-specific write method first (e.g., writeJpeg)
    const writeMethod = toCamelCase(`write_${format}`);
    const method = this._pipeline[writeMethod];

    if (typeof method === 'function') {
      return method.call(this._pipeline, this._nodeId, config, null) as Uint8Array;
    }

    // Fall back to generic write(nodeId, format, quality, metadata)
    const genericWrite = this._pipeline['write'];
    if (typeof genericWrite === 'function') {
      const quality = (config.quality as number) ?? null;
      return genericWrite.call(this._pipeline, this._nodeId, format, quality, null) as Uint8Array;
    }

    throw new Error(`No encoder found for format "${format}".`);
  }
}

export default RcImage;
