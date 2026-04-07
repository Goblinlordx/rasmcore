/**
 * ML Provider implementation — the core of @rasmcore/ml.
 *
 * Creates an MlProvider that:
 * 1. Detects available backends (WebNN, ORT WebGPU, ORT WASM)
 * 2. Manages model sessions (load on demand, cache)
 * 3. Executes single-tile inference via the best available backend
 */

import type { MlCapabilityInfo, MlOp, MlProvider } from '../ml-provider.js';
import { MlError } from '../ml-provider.js';
import { detectBackends, selectOrtProvider, type BackendInfo } from './backends/detect.js';

export interface MlProviderConfig {
  /** Model definitions to register. */
  models: MlCapabilityInfo[];
  /** Optional: override model file URLs (keyed by model name). */
  modelUrls?: Record<string, string>;
  /** Optional: progress callback for model downloads. */
  onProgress?: (modelName: string, loaded: number, total: number) => void;
}

/** Create an ML provider with the given configuration. */
export async function createMlProvider(config: MlProviderConfig): Promise<MlProvider> {
  const backends = await detectBackends();
  const selectedProvider = selectOrtProvider(backends);
  const sessions = new Map<string, any>(); // model name -> ORT InferenceSession

  // Annotate each model capability with the detected backend
  const capabilities: MlCapabilityInfo[] = config.models.map(m => ({
    ...m,
    backend: selectedProvider,
  }));

  return {
    capabilities() {
      return capabilities;
    },

    async execute(op: MlOp): Promise<ArrayBuffer> {
      // Check if ORT is available
      const ort = (globalThis as any).ort;
      if (!ort) {
        throw new MlError('not-available', 'ONNX Runtime Web not loaded. Include onnxruntime-web via CDN or npm.');
      }

      // Get or create session for this model
      let session = sessions.get(op.modelName);
      if (!session) {
        const modelUrl = config.modelUrls?.[op.modelName];
        if (!modelUrl) {
          throw new MlError('model-not-found', `No URL configured for model "${op.modelName}". Set modelUrls in provider config.`);
        }

        // Map our backend name to ORT execution provider config
        const epMap: Record<string, any[]> = {
          'webnn-gpu': [{ name: 'webnn', deviceType: 'gpu', powerPreference: 'high-performance' }],
          'webnn-cpu': [{ name: 'webnn', deviceType: 'cpu' }],
          'webgpu': ['webgpu'],
          'wasm': ['wasm'],
        };
        const eps = epMap[selectedProvider] || ['wasm'];

        // Fetch model
        const response = await fetch(modelUrl);
        if (!response.ok) {
          throw new MlError('model-loading', `Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        const modelBuffer = await response.arrayBuffer();

        // Create session
        try {
          session = await ort.InferenceSession.create(modelBuffer, { executionProviders: eps });
          sessions.set(op.modelName, session);
        } catch (e: any) {
          throw new MlError('model-loading', `Failed to create session: ${e.message}`);
        }
      }

      // Create input tensor
      const inputTensor = new ort.Tensor(
        op.inputDesc.dtype,
        new Float32Array(op.input),
        op.inputDesc.shape.map(d => d < 0 ? 1 : d), // replace -1 with 1 for fixed dims
      );

      // Run inference
      const feeds: Record<string, any> = {};
      feeds[session.inputNames[0]] = inputTensor;

      let results;
      try {
        results = await session.run(feeds);
      } catch (e: any) {
        throw new MlError('inference-error', `Inference failed: ${e.message}`);
      }

      // Extract output
      const outputTensor = results[session.outputNames[0]];
      return outputTensor.data.buffer as ArrayBuffer;
    },

    dispose() {
      for (const [name, session] of sessions) {
        try { session.release?.(); } catch {}
      }
      sessions.clear();
    },
  };
}
