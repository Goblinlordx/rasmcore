# Clamp Audit — Scene-Referred Pipeline

The V2 pipeline is scene-referred linear f32. Pixel values are unbounded:
values > 1.0 (HDR highlights) and < 0.0 (difference mattes) are valid.
Clamping to [0,1] should only happen at the encode boundary.

## Disposition Summary

| Category | Count | Action |
|----------|-------|--------|
| REMOVE | 15 | Removed — pixel value clamping destroyed HDR data |
| FIX | 6 | Fixed — histogram binning now handles >1.0 gracefully |
| KEEP | 22 | Preserved — legitimate constraints (weights, alpha, HSL domain) |
| KEEP-OKLCH | 10 | Preserved — HSL domain constraints, will be obsoleted by OKLCH migration |

## REMOVED — Pixel value clamping (destroys HDR)

| Filter | File | What was clamped | Fix |
|--------|------|-----------------|-----|
| Sobel | edge/sobel.rs | Gradient magnitude | Removed `.clamp(0.0, 1.0)` |
| Scharr | edge/scharr.rs | Edge magnitude | Removed `.clamp(0.0, 1.0)` |
| Laplacian | edge/laplacian.rs | Laplacian magnitude | Removed `.clamp(0.0, 1.0)` |
| Drago tonemap | grading/tonemap.rs | Tone map output | Removed `.clamp(0.0, 1.0)` (compute + CLUT) |
| Filmic tonemap | grading/tonemap.rs | Tone map output | Removed `.clamp(0.0, 1.0)` (compute + CLUT) |
| Lift/Gamma/Gain | grading/lift_gamma_gain.rs | LGG output | Removed `.clamp(0.0, 1.0)` (kept `.max(0.0)` for pow safety) |
| Split Toning | grading/split_toning.rs | Final RGB | Removed `.clamp(0.0, 1.0)` (kept blend weight clamps) |
| ASC CDL | grading/asc_cdl.rs | Final RGB | Removed `.clamp(0.0, 1.0)` (kept `.max(0.0)` for pow safety) |
| Sepia | color/sepia.rs | Matrix output | Removed `.min(1.0)` |
| Clarity | enhancement/clarity.rs | Computed luminance | Removed `.clamp(0.0, 1.0)` |
| Shadow/Highlight | enhancement/shadow_highlight.rs | Computed + blurred luma | Removed `.clamp(0.0, 1.0)` (2 sites) |

## FIXED — Histogram binning (graceful >1.0 handling)

Changed from `pixel.clamp(0.0, 1.0) * 255.0` to `((pixel.max(0.0) * 255.0) as usize).min(255)`.
This maps >1.0 to bin 255 without clamping the pixel value.

| Filter | File |
|--------|------|
| Otsu threshold | edge/otsu_threshold.rs |
| Triangle threshold | edge/triangle_threshold.rs |
| Oil paint | effect/oil_paint.rs |
| Equalize | enhancement/equalize.rs (2 sites) |
| Normalize | enhancement/normalize.rs |

## KEPT — Legitimate constraints

- **HSL saturation/lightness** (saturate_hsl, modulate, vibrance, replace_color, selective_color): S and L are intrinsically [0,1] in the HSL model
- **Blend weights** (blend_if, split_toning shadow/highlight weights): Interpolation factors must be [0,1]
- **Math safety** (`.max(0.0)` before `powf()` or `ln()`): Prevents NaN from negative inputs
- **Alpha bounds**: Alpha channel is [0,1] by convention
- **Scope visualizations** (histogram, waveform, parade, vectorscope): Display requires [0,1] mapping

## KEPT-OKLCH — HSL domain constraints (pending OKLCH migration)

All hue curve filters (hue_vs_sat, hue_vs_lum, lum_vs_sat, sat_vs_sat) clamp
HSL saturation and lightness to [0,1]. These are correct for the HSL model but
will be obsoleted when these filters migrate to OKLCH (chroma is unbounded).
See track: perceptual-saturation_20260408000100Z for the migration pattern.
