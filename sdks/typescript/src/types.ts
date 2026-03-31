/**
 * Filter manifest types — describes all operations a WASM module supports.
 * Loaded at runtime via getFilterManifest().
 *
 * These types are module-agnostic. Any WASM component that exposes
 * getFilterManifest() and getManifestHash() is a valid module,
 * regardless of what filters/encoders/operations it contains.
 */

export interface ParamMeta {
  name: string;
  type: string;
  min: number | null;
  max: number | null;
  step: number | null;
  default: unknown;
  label: string;
  hint: string;
}

export interface OperationMeta {
  name: string;
  category: string;
  group: string;
  variant: string;
  reference: string;
  params: ParamMeta[];
}

export interface FilterManifest {
  filters: OperationMeta[];
  [key: string]: unknown; // extensible — modules may add generators, encoders, etc.
}

/**
 * A WASM module that supports manifest introspection.
 *
 * This is the only contract. The module can have any operations —
 * the runtime discovers them via getFilterManifest().
 * Methods are dispatched dynamically by name.
 */
export interface WasmModule {
  [namespace: string]: Record<string, unknown> | unknown;
}

/**
 * A pipeline resource from a WASM module.
 *
 * All methods are accessed dynamically via string keys.
 * The runtime reads the manifest to know what methods exist
 * and what parameters they accept.
 */
export interface WasmPipeline {
  [method: string]: (...args: unknown[]) => unknown;
}
