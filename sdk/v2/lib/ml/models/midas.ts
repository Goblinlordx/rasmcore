/**
 * MiDaS v2.1 small — depth estimation model definition.
 *
 * MIT license. Original: Intel ISL (isl-org/MiDaS).
 * Weights downloaded on first use (~50MB ONNX).
 */
import type { MlCapabilityInfo } from '../../ml-provider.js';

export const midasV21Small: MlCapabilityInfo = {
  model: { name: 'midas-v2.1-small', version: '2.1.0' },
  displayName: 'AI Depth Estimation',
  category: 'depth',
  outputKind: 'mask',
  inputSpec: {
    tileMode: 'full-image',
    targetSize: [384, 384],
    overlap: 0,
    padding: 'zero',
  },
  inputDesc: { shape: [1, 3, 384, 384], dtype: 'float32' },
  outputDesc: { shape: [1, 1, 384, 384], dtype: 'float32' },
  params: [],
  backend: 'auto',
  estimatedMsPerTile: 200,
};

export const midasV21SmallFiles = {
  onnx: {
    url: '',
    sizeBytes: 52_428_800,
    sha256: '',
  },
};
