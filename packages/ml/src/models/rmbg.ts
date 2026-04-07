/**
 * RMBG-1.4 — background removal / segmentation model definition.
 *
 * MIT license. Original: BRIA AI (briaai/RMBG-1.4).
 * Weights downloaded on first use (~176MB ONNX).
 */
import type { MlCapabilityInfo } from '../types.js';

export const rmbg14: MlCapabilityInfo = {
  model: { name: 'rmbg-1.4', version: '1.4.0' },
  displayName: 'AI Background Removal',
  category: 'segmentation',
  outputKind: 'mask',
  inputSpec: {
    tileMode: 'full-image',
    targetSize: [1024, 1024],
    overlap: 0,
    padding: 'zero',
  },
  inputDesc: { shape: [1, 3, 1024, 1024], dtype: 'float32' },
  outputDesc: { shape: [1, 1, 1024, 1024], dtype: 'float32' },
  params: [
    { name: 'threshold', type: 'f32', min: 0, max: 1, step: 0.05, default: 0.5, hint: 'Mask threshold' },
  ],
  backend: 'auto',
  estimatedMsPerTile: 500,
};

export const rmbg14Files = {
  onnx: {
    url: '',
    sizeBytes: 184_549_376,
    sha256: '',
  },
};
