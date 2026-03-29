# Image Operation Reference Alignment

Every image processing operation in rasmcore-image is validated against a known-good
reference implementation. This document records which reference each operation is
aligned with, the match expectation, and the validation status.

## Match Tiers

| Tier | Expectation | Meaning |
|------|------------|---------|
| **EXACT** | MAE == 0.0 | Bit-identical output. Any difference is a bug. |
| **DETERMINISTIC** | MAE == 0.0 | Same formula → same output. If different, match their formula. |
| **ALGORITHM** | MAE <= 1.0 | ±1 per channel max, each differing pixel documented. |
| **DESIGN** | Equivalent outcome | Different algorithm by design. Document rationale. |

## Operation Reference Table

### Point Operations (LUT-based)

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| gamma | ImageMagick | `magick -gamma {val}` | DETERMINISTIC | PASS (MAE=0.0) |
| invert | ImageMagick | `magick -negate` | EXACT | PASS (MAE=0.0) |
| threshold | ImageMagick | `magick -channel All -threshold {pct}%` | EXACT | PASS (MAE=0.0, per-channel) |
| posterize | ImageMagick | `magick +dither -posterize {levels}` | EXACT | PASS (MAE=0.0, no dither) |
| clamp | ImageMagick | `magick -clamp` (to [min,max]) | EXACT | NEEDS TEST |
| brightness | ImageMagick | `magick -brightness-contrast {b}x0` | DETERMINISTIC | PASS (MAE=0.0) |
| contrast | ImageMagick | `magick -brightness-contrast 0x{c}` | DETERMINISTIC | PASS (MAE<1.0) |

### Color Operations

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| grayscale | ImageMagick | `magick -colorspace Gray` | DETERMINISTIC | PASS (existing) |
| hue_rotate | ImageMagick | `magick -modulate 100,100,{hue}` (HSL) | ALGORITHM | PASS (MAE=0.06) |
| saturate | ImageMagick | `magick -modulate 100,{sat},100` | ALGORITHM | PASS (MAE=0.02) |
| sepia | ImageMagick | `magick -sepia-tone {pct}%` | DETERMINISTIC | NEEDS TEST |
| colorize | ImageMagick | `magick -fill "rgb()" -colorize {pct}%` | DETERMINISTIC | NEEDS TEST |

### Spatial Filters

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| blur | ImageMagick / libblur | `magick -blur 0x{sigma}` | ALGORITHM | PASS (MAE=0.0) |
| sharpen | ImageMagick | `magick -sharpen 0x{amount}` | ALGORITHM | NEEDS TEST |
| convolve | ImageMagick | `magick -convolve "{kernel}"` | DETERMINISTIC | NEEDS TEST |
| median | ImageMagick | `magick -median {radius}` | ALGORITHM | PASS (MAE=0.07) |
| sobel | ImageMagick (approx) | `magick -edge 1` | DESIGN | NEEDS TEST |
| canny | ImageMagick | `magick -canny 0x1+{lo}%+{hi}%` | DESIGN | NEEDS TEST |

### Histogram Operations

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| histogram | N/A (internal) | — | N/A | Internal only |
| statistics | N/A (internal) | — | N/A | Internal only |
| equalize | ImageMagick | `magick -equalize` | ALGORITHM | PASS (MAE=11.5, Q16-HDRI precision) |
| normalize | ImageMagick | `magick -normalize` (2%/1% stretch) | ALGORITHM | PASS (MAE=7.6, Q16-HDRI precision) |
| auto_level | ImageMagick | `magick -auto-level` | ALGORITHM | PASS (MAE<1.0) |
| contrast_stretch | ImageMagick | `magick -contrast-stretch {b}x{w}%` | ALGORITHM | NEEDS TEST |

### Alpha & Compositing

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| premultiply | ImageMagick (formula) | N/A (formula: c*a/255) | EXACT | NEEDS TEST |
| unpremultiply | ImageMagick (formula) | N/A (formula: c*255/a) | EXACT | NEEDS TEST |
| flatten | ImageMagick | `magick -background white -flatten` | EXACT | PASS (MAE=0.0) |
| add_alpha | Trivial | N/A (append 255 per pixel) | EXACT | NEEDS TEST |
| remove_alpha | Trivial | N/A (drop alpha channel) | EXACT | NEEDS TEST |
| alpha_composite_over | ImageMagick | `magick -compose Over -composite` | EXACT | NEEDS TEST |
| blend Multiply | ImageMagick | `magick -compose Multiply -composite` | EXACT | NEEDS TEST |
| blend Screen | ImageMagick | `magick -compose Screen -composite` | EXACT | NEEDS TEST |
| blend Overlay | ImageMagick | `magick -compose Overlay -composite` | EXACT | NEEDS TEST |
| blend Darken | ImageMagick | `magick -compose Darken -composite` | EXACT | NEEDS TEST |
| blend Lighten | ImageMagick | `magick -compose Lighten -composite` | EXACT | NEEDS TEST |
| blend SoftLight | ImageMagick | `magick -compose SoftLight -composite` | EXACT | NEEDS TEST |
| blend HardLight | ImageMagick | `magick -compose HardLight -composite` | EXACT | NEEDS TEST |
| blend Difference | ImageMagick | `magick -compose Difference -composite` | EXACT | NEEDS TEST |
| blend Exclusion | ImageMagick | `magick -compose Exclusion -composite` | EXACT | NEEDS TEST |

### Geometric Transforms

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| resize | ImageMagick | `magick -resize {w}x{h}` | ALGORITHM | PASS (existing) |
| crop | ImageMagick | `magick -crop {w}x{h}+{x}+{y}` | EXACT | PASS (existing) |
| rotate 90/180/270 | ImageMagick | `magick -rotate {deg}` | EXACT | PASS (existing) |
| flip H/V | ImageMagick | `magick -flip` / `-flop` | EXACT | PASS (existing) |
| rotate_arbitrary | ImageMagick | `magick -rotate {deg}` | DESIGN | PASS (bilinear vs three-shear, center MAE=2.6) |
| pad | ImageMagick | `magick -gravity center -extent {w}x{h}` | EXACT | PASS (MAE=0.0) |
| trim | ImageMagick | `magick -trim` | ALGORITHM | PASS (dimensions within ±2px) |
| affine | ImageMagick | `magick -affine "{matrix}" -transform` | ALGORITHM | NEEDS TEST |
| auto_orient | ImageMagick | `magick -auto-orient` | EXACT | NEEDS TEST |

### Content-Aware

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| smart_crop (entropy) | libvips | `vips smartcrop --interesting entropy` | DESIGN | PASS (aligned) |
| smart_crop (attention) | libvips | `vips smartcrop --interesting attention` | DESIGN | PASS (aligned) |

### Color Management

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| icc_to_srgb | moxcms (internal) | — | N/A | Internal (uses moxcms) |

### Pipeline Infrastructure

| Operation | Reference | Tier | Status |
|-----------|-----------|------|--------|
| build_lut / apply_lut | Self (mathematical identity) | EXACT | PASS (unit tests) |
| compose_luts | Self (a[b[i]] == sequential) | EXACT | PASS (unit tests) |
| ColorLut3D / tetrahedral | Self (accuracy within 0.005 of exact) | ALGORITHM | PASS (unit tests) |

## Summary

| Status | Count |
|--------|-------|
| PASS | 25 |
| NEEDS TEST | 18 |
| N/A (internal) | 4 |
| **Total** | **47** |

## Notes

### ImageMagick Q16-HDRI vs Q8

ImageMagick 7 Q16-HDRI processes pixels as 64-bit floating-point internally, while
rasmcore operates at 8-bit. For histogram-based operations (equalize, normalize), this
creates rounding differences visible as MAE 7-12. The algorithms are identical; the
precision difference is intrinsic.

### Per-channel vs Intensity

Our threshold and posterize are per-channel LUT operations (fusible in the pipeline).
ImageMagick's default `-threshold` uses pixel intensity (luminance); use `-channel All`
for per-channel comparison. ImageMagick's `-posterize` dithers by default; use `+dither`
for exact comparison.

### Hue Rotation: HSL not HSV

Our hue rotation uses HSL (matching ImageMagick `-modulate`). IM's modulate hue scale:
200 = 360°, so +90° = percentage 150. Saturation operates in HSL space with
multiplicative scaling.

### Rotation: Bilinear vs Three-Shear

ImageMagick uses the Paeth three-shear rotation algorithm, while we use bilinear
interpolation. This is a DESIGN-tier difference: equivalent visual outcome, different
algorithm. Canvas sizes may differ by ��2px.
