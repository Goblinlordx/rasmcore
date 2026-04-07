/**
 * Model Pack — a collection of ML model definitions with their download sources.
 *
 * Model packs are the distribution unit for ML capabilities. They contain
 * model definitions (metadata only, ~1KB each) and know where to download
 * weights from (configurable: local path, CDN, or custom URL).
 *
 * A pack can be:
 * - The standard pack (bundled with the SDK)
 * - A third-party pack (e.g., anime-focused models)
 * - A custom pack (user's own models)
 *
 * @example
 * ```ts
 * import { ml } from '@rasmcore/sdk';
 *
 * // Standard pack — all built-in models, default CDN
 * await ml.setup();
 *
 * // Standard pack — custom model hosting
 * await ml.setup({ cdnBase: 'https://my-cdn.com/models/' });
 *
 * // Standard pack — local files (no download needed)
 * await ml.setup({ cdnBase: '/models/' });
 *
 * // Custom pack
 * await ml.setup({ pack: myCustomPack });
 * ```
 */

import type { MlCapabilityInfo, MlProvider } from '../ml-provider.js';

/** A model entry in a pack — definition + download source. */
export interface ModelPackEntry {
  /** Model capability info (the definition). */
  capability: MlCapabilityInfo;
  /** Default download URL for the ONNX file (overridable via cdnBase). */
  defaultUrl: string;
  /** File size in bytes (for progress reporting). */
  sizeBytes: number;
}

/** A collection of models that can be registered together. */
export interface ModelPack {
  /** Pack identifier. */
  name: string;
  /** Human-readable description. */
  description: string;
  /** Model entries in this pack. */
  models: ModelPackEntry[];
}

/** Options for ml.setup(). */
export interface MlSetupOptions {
  /** Base URL for model downloads. Appended with model filename.
   *  Examples: 'https://cdn.example.com/models/', '/local/models/', './models/'
   *  Default: uses each model's defaultUrl. */
  cdnBase?: string;
  /** Custom model pack. Default: standard pack (built-in models). */
  pack?: ModelPack;
  /** Progress callback for model downloads. */
  onProgress?: (modelName: string, loaded: number, total: number) => void;
}

/** Resolve model URLs from pack entries + optional cdnBase override. */
export function resolveModelUrls(pack: ModelPack, cdnBase?: string): Record<string, string> {
  const urls: Record<string, string> = {};
  for (const entry of pack.models) {
    if (cdnBase) {
      // cdnBase + model name + .onnx
      const base = cdnBase.endsWith('/') ? cdnBase : cdnBase + '/';
      urls[entry.capability.model.name] = base + entry.capability.model.name + '.onnx';
    } else {
      urls[entry.capability.model.name] = entry.defaultUrl;
    }
  }
  return urls;
}
