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
| gamma | ImageMagick | `magick -gamma {val}` | DETERMINISTIC | NEEDS TEST |
| invert | ImageMagick | `magick -negate` | EXACT | NEEDS TEST |
| threshold | ImageMagick | `magick -threshold {pct}%` | EXACT | NEEDS TEST |
| posterize | ImageMagick | `magick -posterize {levels}` | EXACT | NEEDS TEST |
| clamp | ImageMagick | `magick -clamp` (to [min,max]) | EXACT | NEEDS TEST |
| brightness | ImageMagick | `magick -brightness-contrast {b}x0` | DETERMINISTIC | NEEDS TEST |
| contrast | ImageMagick | `magick -brightness-contrast 0x{c}` | DETERMINISTIC | NEEDS TEST |

### Color Operations

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| grayscale | ImageMagick | `magick -colorspace Gray` | DETERMINISTIC | PASS (existing) |
| hue_rotate | ImageMagick | `magick -modulate 100,100,{hue}` | ALGORITHM | NEEDS TEST |
| saturate | ImageMagick | `magick -modulate 100,{sat},100` | ALGORITHM | NEEDS TEST |
| sepia | ImageMagick | `magick -sepia-tone {pct}%` | DETERMINISTIC | NEEDS TEST |
| colorize | ImageMagick | `magick -fill "rgb()" -colorize {pct}%` | DETERMINISTIC | NEEDS TEST |

### Spatial Filters

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| blur | ImageMagick / libblur | `magick -blur 0x{sigma}` | ALGORITHM | NEEDS TEST |
| sharpen | ImageMagick | `magick -sharpen 0x{amount}` | ALGORITHM | NEEDS TEST |
| convolve | ImageMagick | `magick -convolve "{kernel}"` | DETERMINISTIC | NEEDS TEST |
| median | ImageMagick | `magick -median {radius}` | ALGORITHM | NEEDS TEST |
| sobel | ImageMagick (approx) | `magick -edge 1` | DESIGN | NEEDS TEST |
| canny | ImageMagick | `magick -canny 0x1+{lo}%+{hi}%` | DESIGN | NEEDS TEST |

### Histogram Operations

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| histogram | N/A (internal) | — | N/A | Internal only |
| statistics | N/A (internal) | — | N/A | Internal only |
| equalize | ImageMagick | `magick -equalize` | ALGORITHM | NEEDS TEST |
| normalize | ImageMagick | `magick -normalize` | ALGORITHM | NEEDS TEST |
| auto_level | ImageMagick | `magick -auto-level` | ALGORITHM | NEEDS TEST |
| contrast_stretch | ImageMagick | `magick -contrast-stretch {b}x{w}%` | ALGORITHM | NEEDS TEST |

### Alpha & Compositing

| Operation | Reference | CLI Equivalent | Tier | Status |
|-----------|-----------|---------------|------|--------|
| premultiply | ImageMagick (formula) | N/A (formula: c*a/255) | EXACT | NEEDS TEST |
| unpremultiply | ImageMagick (formula) | N/A (formula: c*255/a) | EXACT | NEEDS TEST |
| flatten | ImageMagick | `magick -background white -flatten` | EXACT | NEEDS TEST |
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
| rotate_arbitrary | ImageMagick | `magick -rotate {deg}` | ALGORITHM | NEEDS TEST |
| pad | ImageMagick | `magick -gravity center -extent {w}x{h}` | EXACT | NEEDS TEST |
| trim | ImageMagick | `magick -trim` | ALGORITHM | NEEDS TEST |
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
| PASS | 11 |
| NEEDS TEST | 32 |
| N/A (internal) | 4 |
| **Total** | **47** |
