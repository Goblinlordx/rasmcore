/**
 * ML backend detection — probes available runtimes at startup.
 */

export interface BackendInfo {
  name: string;
  available: boolean;
  deviceType?: string;
  error?: string;
}

/** Detect all available ML backends. */
export async function detectBackends(): Promise<BackendInfo[]> {
  const results: BackendInfo[] = [];

  // WebNN
  if (typeof navigator !== 'undefined' && 'ml' in navigator) {
    for (const deviceType of ['gpu', 'npu', 'cpu'] as const) {
      try {
        await (navigator as any).ml.createContext({ deviceType });
        results.push({ name: 'webnn', available: true, deviceType });
      } catch (e: any) {
        results.push({ name: 'webnn', available: false, deviceType, error: e.message });
      }
    }
  } else {
    results.push({ name: 'webnn', available: false, error: 'navigator.ml not available' });
  }

  // WebGPU (for ORT WebGPU EP)
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) {
        results.push({ name: 'webgpu', available: true });
      } else {
        results.push({ name: 'webgpu', available: false, error: 'No adapter' });
      }
    } catch (e: any) {
      results.push({ name: 'webgpu', available: false, error: e.message });
    }
  } else {
    results.push({ name: 'webgpu', available: false, error: 'navigator.gpu not available' });
  }

  // WASM SIMD (always available in modern browsers)
  results.push({ name: 'wasm', available: true });

  return results;
}

/** Pick the best ORT execution provider based on detected backends. */
export function selectOrtProvider(backends: BackendInfo[]): string {
  // Priority: webnn-gpu > webgpu > webnn-cpu > wasm
  if (backends.some(b => b.name === 'webnn' && b.available && b.deviceType === 'gpu')) {
    return 'webnn-gpu';
  }
  if (backends.some(b => b.name === 'webgpu' && b.available)) {
    return 'webgpu';
  }
  if (backends.some(b => b.name === 'webnn' && b.available && b.deviceType === 'cpu')) {
    return 'webnn-cpu';
  }
  return 'wasm';
}
