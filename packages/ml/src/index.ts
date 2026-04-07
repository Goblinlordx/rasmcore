/**
 * @rasmcore/ml — ML inference provider package.
 *
 * Separate from the core SDK. Users opt-in by importing this package
 * and registering the provider.
 *
 * @example
 * ```ts
 * import { registerMlProvider } from '@rasmcore/sdk';
 * import { createMlProvider } from '@rasmcore/ml';
 * import { realEsrganX4plus } from '@rasmcore/ml/models';
 *
 * const provider = await createMlProvider({
 *   models: [realEsrganX4plus],
 *   modelUrls: { 'real-esrgan-x4plus': './models/Real-ESRGAN-x4plus.onnx' },
 * });
 * registerMlProvider(provider);
 * ```
 */

export { createMlProvider, type MlProviderConfig } from './provider.js';
export { detectBackends, selectOrtProvider, type BackendInfo } from './backends/detect.js';
export type { MlCapabilityInfo, MlProvider, MlOp, MlError } from './types.js';
