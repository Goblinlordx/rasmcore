/**
 * ML Provider interface — the contract between the core SDK and ML packages.
 *
 * The core SDK exposes `registerMlProvider(provider)` which wires the provider
 * into the WASM adapter's ml-execute/ml-capabilities imports.
 *
 * ML packages (@rasmcore/ml, rasmcore-ml crate) implement this interface.
 * The core SDK has zero ML dependencies — it just defines the contract.
 */

// ─── Model Definition Types ────────────────────────────────────────────────

/** How the model handles input dimensions. */
export type MlTileMode = 'tileable' | 'full-image';

/** How to pad tiles at image edges. */
export type MlPaddingMode = 'mirror' | 'zero' | 'clamp';

/** Model input requirements — pipeline uses this for tiling strategy. */
export interface MlInputSpec {
  tileMode: MlTileMode;
  /** For tileable: optimal tile size. */
  preferredSize?: [number, number];
  /** For tileable: minimum usable tile size. */
  minSize?: [number, number];
  /** For full-image: resize input to this. */
  targetSize?: [number, number];
  /** Pixels of overlap between tiles (tileable only). */
  overlap: number;
  /** Edge padding mode. */
  padding: MlPaddingMode;
}

/** What kind of output the model produces. */
export type MlOutputKind = 'image' | 'mask';

/** Tensor element type. */
export type TensorDtype = 'float32' | 'float16' | 'uint8' | 'int8';

/** Tensor shape and type. */
export interface TensorDesc {
  shape: number[];
  dtype: TensorDtype;
}

/** Parameter descriptor — same schema as filter params for UI generation. */
export interface MlParamDescriptor {
  name: string;
  type: 'f32' | 'u32' | 'bool' | 'string';
  min?: number;
  max?: number;
  step?: number;
  default?: number;
  hint?: string;
}

/** Model capability info — what the host reports to the pipeline. */
export interface MlCapabilityInfo {
  model: { name: string; version: string };
  displayName: string;
  category: string;
  outputKind: MlOutputKind;
  outputScale?: number;
  inputSpec: MlInputSpec;
  inputDesc: TensorDesc;
  outputDesc: TensorDesc;
  params: MlParamDescriptor[];
  backend: string;
  estimatedMsPerTile: number;
}

/** ML inference request — single tile or full image. */
export interface MlOp {
  modelName: string;
  modelVersion: string;
  input: ArrayBuffer;
  inputDesc: TensorDesc;
  outputDesc: TensorDesc;
  params: ArrayBuffer;
}

/** ML inference error. */
export class MlError extends Error {
  constructor(
    public code: 'model-not-found' | 'model-loading' | 'inference-error' | 'shape-mismatch' | 'not-available',
    message: string,
  ) {
    super(message);
    this.name = 'MlError';
  }
}

// ─── Provider Interface ────────────────────────────────────────────────────

/**
 * ML Provider — implemented by @rasmcore/ml (browser) or rasmcore-ml (native).
 *
 * The core SDK calls these methods to:
 * 1. Discover available models (capabilities)
 * 2. Execute single-tile inference (execute)
 * 3. Clean up resources (dispose)
 */
export interface MlProvider {
  /** List available models and their capabilities. */
  capabilities(): MlCapabilityInfo[];

  /** Execute inference on a single tile. Returns output tensor bytes. */
  execute(op: MlOp): Promise<ArrayBuffer>;

  /** Release resources (model sessions, GPU contexts, etc.). */
  dispose(): void;
}

// ─── Registration API ──────────────────────────────────────────────────────

let _registeredProvider: MlProvider | null = null;

/**
 * Register an ML provider with the SDK.
 *
 * Call this before creating pipelines to enable ML operations.
 * Only one provider can be active at a time.
 *
 * @example
 * ```ts
 * import { registerMlProvider } from '@rasmcore/sdk';
 * import { createMlProvider } from '@rasmcore/ml';
 * import { realEsrgan, rmbg } from '@rasmcore/ml/models';
 *
 * registerMlProvider(createMlProvider({
 *   models: [realEsrgan, rmbg],
 * }));
 * ```
 */
export function registerMlProvider(provider: MlProvider): void {
  if (_registeredProvider) {
    _registeredProvider.dispose();
  }
  _registeredProvider = provider;
}

/** Get the currently registered ML provider (or null). */
export function getMlProvider(): MlProvider | null {
  return _registeredProvider;
}

/** Check if ML is available (a provider is registered with capabilities). */
export function isMlAvailable(): boolean {
  return _registeredProvider !== null && _registeredProvider.capabilities().length > 0;
}

/** List available ML models. Returns empty array if no provider registered. */
export function listMlModels(): MlCapabilityInfo[] {
  return _registeredProvider?.capabilities() ?? [];
}
