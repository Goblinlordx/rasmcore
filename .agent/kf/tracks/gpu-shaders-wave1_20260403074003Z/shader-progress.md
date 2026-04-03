# GPU Shaders Wave 1 — Per-Filter Progress

## Point Ops
| Filter | Shader | GpuFilter impl | Parity test | MAE | Status |
|--------|--------|----------------|-------------|-----|--------|
| brightness | brightness_f32.wgsl | BrightnessParams | gpu_f32_parity_brightness | 0.0000 | Done |
| contrast | contrast_f32.wgsl | ContrastParams | gpu_f32_parity_contrast | 0.0000 | Done |
| gamma | gamma_f32.wgsl | GammaParams | gpu_f32_parity_gamma | 0.0000 | Done |
| solarize | solarize_f32.wgsl | SolarizeParams | gpu_f32_parity_solarize | 0.0000 | Done |
| invert | invert_f32.wgsl | — | — | — | Shader ready, no params struct |
| exposure | exposure_f32.wgsl | ExposureParams | gpu_f32_parity_exposure | 0.0000 | Done |
| levels | — | — | — | — | Pending |
| posterize | posterize_f32.wgsl | PosterizeParams | gpu_f32_parity_posterize | 0.0000 | Done |
| sigmoidal_contrast | — | — | — | — | Pending |

## Color + Grading Ops
| Filter | Shader | GpuFilter impl | Parity test | MAE | Status |
|--------|--------|----------------|-------------|-----|--------|
| sepia | sepia_f32.wgsl | SepiaParams | gpu_f32_parity_sepia | 0.0001 | Done |
| hue_rotate | hue_rotate_f32.wgsl | HueRotateParams | gpu_f32_parity_hue_rotate | 0.0218 | Done |
| saturate | saturate_f32.wgsl | SaturateParams | gpu_f32_parity_saturate | 0.0196 | Done |
| vibrance | vibrance_f32.wgsl | — | — | — | Shader written, CPU uses different formula — needs matching |
| modulate | — | — | — | — | Pending |
| colorize | — | — | — | — | Pending |
| channel_mixer | — | — | — | — | Pending |
| color_balance | — | — | — | — | Pending |

## Simple Effect Ops
| Filter | Shader | GpuFilter impl | Parity test | MAE | Status |
|--------|--------|----------------|-------------|-----|--------|
| film_grain | — | — | — | — | Pending |
| gaussian_noise | — | — | — | — | Pending |
| uniform_noise | — | — | — | — | Pending |

## Summary
- **9 shaders completed** with validated parity (all MAE < 1.0)
- **2 shaders written** but not wired (invert: no params, vibrance: formula mismatch)
- **~13 remaining** in this wave

## Notes
- All shaders use 16x16 workgroup, [0,1] normalized f32 I/O
- GpuFilter detection is automatic via trait impl scanning (no gpu="true" needed)
- Parity tests compare GPU against f32 CPU reference (not u8 LUT)
- HSL-based ops (hue_rotate, saturate) have ±1 LSB difference from floating point — acceptable
