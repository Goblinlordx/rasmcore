/**
 * Standard model pack — ships with the SDK.
 *
 * Contains definitions (not weights) for:
 * - Real-ESRGAN x4plus (AI super-resolution, 4x upscale)
 * - RMBG-1.4 (AI background removal)
 * - MiDaS v2.1 small (AI depth estimation)
 */

import type { ModelPack } from './model-pack.js';
import { realEsrganX4plus } from './models/real-esrgan.js';
import { rmbg14 } from './models/rmbg.js';
import { midasV21Small } from './models/midas.js';

export const standardPack: ModelPack = {
  name: 'standard',
  description: 'Built-in AI models: super-resolution, background removal, depth estimation',
  models: [
    {
      capability: realEsrganX4plus,
      defaultUrl: '',  // Must be configured via cdnBase or modelUrls
      sizeBytes: 67_108_864,
    },
    {
      capability: rmbg14,
      defaultUrl: '',
      sizeBytes: 184_549_376,
    },
    {
      capability: midasV21Small,
      defaultUrl: '',
      sizeBytes: 52_428_800,
    },
  ],
};
