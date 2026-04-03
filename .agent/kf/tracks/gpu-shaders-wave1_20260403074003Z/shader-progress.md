# GPU Shaders Wave 1 — Per-Filter Progress

## Point Ops (COMPLETE)
| Filter | Shader | MAE | Status |
|--------|--------|-----|--------|
| brightness | brightness_f32.wgsl | 0.0000 | Done |
| contrast | contrast_f32.wgsl | 0.0000 | Done |
| gamma | gamma_f32.wgsl | 0.0000 | Done |
| exposure | exposure_f32.wgsl | 0.0000 | Done |
| levels | levels_f32.wgsl | 0.0000 | Done |
| posterize | posterize_f32.wgsl | 0.0000 | Done |
| sigmoidal_contrast | sigmoidal_contrast_f32.wgsl | 0.0000 | Done |
| solarize | solarize_f32.wgsl | 0.0000 | Done |
| invert | invert_f32.wgsl | — | Shader ready, no params struct |

## Color + Grading Ops (COMPLETE)
| Filter | Shader | MAE | Status |
|--------|--------|-----|--------|
| sepia | sepia_f32.wgsl | 0.0001 | Done |
| hue_rotate | hue_rotate_f32.wgsl | 0.0218 | Done |
| saturate | saturate_f32.wgsl | 0.0196 | Done |
| modulate | modulate_f32.wgsl | 0.0000 | Done |
| channel_mixer | channel_mixer_f32.wgsl | 0.0005 | Done |
| vibrance | vibrance_f32.wgsl | 0.0001 | Done |
| colorize | — | — | Pending (W3C blend + LAB) |
| color_balance | — | — | Pending (9 params + tonal weighting) |

## Effect Ops (IN PROGRESS)
| Filter | Shader | MAE | Status |
|--------|--------|-----|--------|
| gaussian_noise | gaussian_noise_f32.wgsl | N/A (PRNG) | Done (deterministic test) |
| uniform_noise | — | — | Pending |
| film_grain | — | — | Pending |

## Summary
- **15 shaders completed** with validated parity (21/21 GPU tests pass)
- **1 shader ready** (invert — no params struct yet)
- **~5 remaining** (colorize, color_balance, uniform_noise, film_grain + invert wiring)
