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

## Color + Grading Ops (IN PROGRESS)
| Filter | Shader | MAE | Status |
|--------|--------|-----|--------|
| sepia | sepia_f32.wgsl | 0.0001 | Done |
| hue_rotate | hue_rotate_f32.wgsl | 0.0218 | Done |
| saturate | saturate_f32.wgsl | 0.0196 | Done |
| modulate | modulate_f32.wgsl | 0.0000 | Done |
| channel_mixer | channel_mixer_f32.wgsl | 0.0005 | Done |
| vibrance | vibrance_f32.wgsl | — | Shader written, CPU formula mismatch |
| colorize | — | — | Complex (W3C blend mode + LAB), deferred |
| color_balance | — | — | Pending |

## Simple Effect Ops
| Filter | Shader | MAE | Status |
|--------|--------|-----|--------|
| film_grain | — | — | Pending |
| gaussian_noise | — | — | Pending |
| uniform_noise | — | — | Pending |

## Summary
- **13 shaders completed** with validated parity (19/19 GPU tests pass)
- **2 shaders written** but not wired (invert, vibrance)
- **~7 remaining** in this wave (color_balance, colorize, vibrance, 3 noise, film_grain)
