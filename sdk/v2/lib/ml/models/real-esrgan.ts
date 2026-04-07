/**
 * Real-ESRGAN x4plus — 4x super-resolution model definition.
 *
 * BSD-3-Clause license. Original: xinntao/Real-ESRGAN.
 * Weights downloaded on first use (~64MB ONNX).
 */
import type { MlCapabilityInfo } from '../../ml-provider.js';

export const realEsrganX4plus: MlCapabilityInfo = {
  model: { name: 'real-esrgan-x4plus', version: '1.0.0' },
  displayName: 'AI Super Resolution (4x)',
  category: 'upscale',
  outputKind: 'image',
  outputScale: 4,
  inputSpec: {
    tileMode: 'tileable',
    preferredSize: [256, 256],
    minSize: [64, 64],
    overlap: 8,
    padding: 'mirror',
  },
  inputDesc: { shape: [1, 3, -1, -1], dtype: 'float32' },
  outputDesc: { shape: [1, 3, -1, -1], dtype: 'float32' },
  params: [
    { name: 'denoise_strength', type: 'f32', min: 0, max: 1, step: 0.1, default: 0.5, hint: 'Noise reduction strength' },
  ],
  backend: 'auto',
  estimatedMsPerTile: 300,
};

/** Model file metadata for download. */
export const realEsrganX4plusFiles = {
  onnx: {
    url: '', // User must configure CDN URL or local path
    sizeBytes: 67_108_864,
    sha256: '', // Set after hosting
  },
};
