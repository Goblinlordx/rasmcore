/**
 * ML module — part of @rasmcore/sdk.
 *
 * Provides `ml.setup()` to enable AI/ML operations in the pipeline.
 * Lazy — imports nothing heavy until setup() is called.
 * Model weights are NOT bundled — downloaded on first use by the host.
 *
 * @example
 * ```ts
 * import { ml } from '@rasmcore/sdk';
 *
 * // Enable ML with standard models, local model files
 * await ml.setup({ cdnBase: './models/' });
 *
 * // Now apply-ml operations are available in the pipeline
 * ```
 */

import { registerMlProvider, isMlAvailable, listMlModels } from '../ml-provider.js';
import { createMlProvider } from './provider.js';
import { standardPack } from './standard-pack.js';
import { resolveModelUrls, type MlSetupOptions } from './model-pack.js';

/**
 * Set up ML capabilities for the pipeline.
 *
 * Creates an ML provider with the specified model pack (default: standard)
 * and registers it with the SDK. After this call, `applyMl()` operations
 * are available in the pipeline.
 *
 * @param options - Configuration: cdnBase for model hosting, custom pack, progress callback
 */
export async function setup(options: MlSetupOptions = {}): Promise<void> {
  const pack = options.pack ?? standardPack;
  const modelUrls = resolveModelUrls(pack, options.cdnBase);
  const capabilities = pack.models.map(m => m.capability);

  const provider = await createMlProvider({
    models: capabilities,
    modelUrls,
    onProgress: options.onProgress,
  });

  registerMlProvider(provider);
}

/** Check if ML has been set up and models are available. */
export { isMlAvailable as isAvailable };

/** List available ML models (empty if setup() hasn't been called). */
export { listMlModels as listModels };

// Re-export types and packs for advanced usage
export { standardPack } from './standard-pack.js';
export type { ModelPack, ModelPackEntry, MlSetupOptions } from './model-pack.js';
export { resolveModelUrls } from './model-pack.js';
export { detectBackends, selectOrtProvider } from './backends/detect.js';
