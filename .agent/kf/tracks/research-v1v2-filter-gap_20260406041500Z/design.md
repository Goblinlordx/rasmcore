# V1 → V2 Filter Gap Analysis

**Track:** research-v1v2-filter-gap_20260406041500Z
**Date:** 2026-04-06

---

## Summary

| Metric | Count |
|--------|-------|
| V1 filters | 179 |
| V2 filters | 81 |
| V2-only (new) | 12 |
| Duplicates/renamed | 6 |
| **Genuine gap** | **92** |

Architecture mandate: **GPU-first** (WGSL compute shaders primary), **SIMD CPU fallback** (production quality, not throwaway).

---

## Duplicates / Renamed (6) — No Action

| V1 Name | V2 Equivalent | Reason |
|---------|---------------|--------|
| blur | gaussian_blur | Renamed for clarity |
| gaussian_blur_cv | gaussian_blur | Consolidated (OpenCV naming dropped) |
| average_blur | box_blur | Equivalent operation |
| invert_v2 | invert | Consolidated (V2 version kept) |
| grayscale | saturate(factor=0) | Composable — not a separate filter |
| lab | lab_adjust | Renamed for clarity |

## V2-Only New (12) — Added During V2 Development

| Filter | Category | Notes |
|--------|----------|-------|
| aces_idt | color | ACES Input Device Transform |
| aces_odt | color | ACES Output Display Transform |
| aces_cct_to_cg | color | ACEScct → ACEScg conversion |
| aces_cg_to_cct | color | ACEScg → ACEScct conversion |
| film_grain_grading | grading | Professional grain with color option |
| gaussian_blur | spatial | Clean V2 reimplementation |
| lab_adjust | color | L*a*b* channel adjustment |
| lab_sharpen | color | L* channel sharpening |
| scope_histogram | analysis | Histogram visualization |
| scope_parade | analysis | RGB parade scope |
| scope_vectorscope | analysis | Vectorscope visualization |
| scope_waveform | analysis | Waveform monitor |

---

## Genuine Gap — 92 Filters by Priority

### P0 — Photoshop Parity (32 filters, 4 tracks)

These are core Photoshop/Lightroom operations expected by professional users.

#### Track: v2-grading-curves (12 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| curves_master | 1D LUT texture lookup | Spline interpolation → baked LUT |
| curves_red | 1D LUT per channel | Same as master, R channel only |
| curves_green | 1D LUT per channel | G channel |
| curves_blue | 1D LUT per channel | B channel |
| color_balance | Per-pixel matrix mul | Shadow/mid/highlight splits |
| hue_vs_lum | 2D LUT (hue → lum offset) | HSL domain, LUT-baked |
| hue_vs_sat | 2D LUT (hue → sat offset) | HSL domain |
| lum_vs_sat | 2D LUT (lum → sat offset) | HSL domain |
| sat_vs_sat | 2D LUT (sat → sat offset) | HSL domain |
| apply_cube_lut | 3D LUT tetrahedral interp | Existing CLUT infrastructure |
| apply_hald_lut | 3D LUT from Hald image | Decode Hald → 3D LUT |
| gradient_map | 1D LUT (luminance → color) | Per-pixel luminance lookup |

#### Track: v2-distortion-filters (10 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| barrel | Coordinate remap shader | Radial distortion k1/k2/k3 |
| spherize | Coordinate remap | Spherical bulge |
| swirl | Coordinate remap | Polar angle offset by radius |
| ripple | Coordinate remap | Sin/cos displacement |
| wave | Coordinate remap | Directional sine wave |
| polar | Cartesian → polar remap | Texture coordinate transform |
| depolar | Polar → Cartesian remap | Inverse of polar |
| liquify | Mesh displacement | GPU mesh warp via uniform grid |
| mesh_warp | Arbitrary mesh displacement | Control points → bilinear interp |
| displacement_map | Channel-driven offset | Second texture as displacement |

#### Track: v2-spatial-advanced (6 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| lens_blur | Disc kernel convolution | Bokeh shape via kernel texture |
| bokeh_blur | Shaped kernel blur | Hexagonal/circular kernels |
| tilt_shift | Depth-gradient blur | Gaussian with position-dependent radius |
| zoom_blur | Radial blur from center | Accumulate samples along radial lines |
| spin_blur | Circular motion blur | Accumulate along arc |
| smart_sharpen | Deconvolution sharpen | Wiener filter or Richardson-Lucy |

#### Track: v2-alpha-composite (4 filters from alpha + composite)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| blend | Porter-Duff compositing | Multiple blend modes in one shader |
| blend_if | Conditional blend by channel | Threshold-based mask generation |
| premultiply | Per-pixel alpha multiply | Trivial GPU shader |
| unpremultiply | Per-pixel alpha divide | Trivial GPU shader |

### P1 — Common Professional Use (28 filters, 4 tracks)

#### Track: v2-morphology (8 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| dilate | Max filter in kernel | Separable for rectangular SE |
| erode | Min filter in kernel | Separable for rectangular SE |
| morph_open | Erode then dilate | Two-pass |
| morph_close | Dilate then erode | Two-pass |
| morph_gradient | Dilate - erode | Single output pass |
| morph_tophat | Input - open | Differential |
| morph_blackhat | Close - input | Differential |
| skeletonize | Iterative thinning | Multi-pass GPU (Zhang-Suen) |

#### Track: v2-edge-threshold (8 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| canny | Gradient + NMS + hysteresis | Multi-pass: Sobel → NMS → double-threshold |
| sobel | 3×3 convolution | Standard Sobel kernels |
| scharr | 3×3 convolution | Higher-accuracy Scharr kernels |
| laplacian | 3×3 convolution | Discrete Laplacian kernel |
| adaptive_threshold | Local mean threshold | Box filter + comparison |
| otsu_threshold | Histogram-based threshold | GPU histogram → CPU threshold → GPU apply |
| threshold_binary | Simple threshold | Trivial GPU shader |
| triangle_threshold | Triangle algorithm | Similar to Otsu — histogram analysis |

#### Track: v2-mask-ops (7 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| color_range | HSL range → mask | Per-pixel HSL check |
| luminance_range | Lum range → mask | Per-pixel luminance check |
| feather | Gaussian blur on mask | Reuse gaussian_blur on single channel |
| from_path | Rasterize path → mask | GPU path rasterization |
| mask_apply | Multiply image by mask | Per-pixel multiply |
| masked_blend | Blend using mask alpha | Per-pixel lerp |
| combine | Boolean ops on masks | Per-pixel and/or/xor |

#### Track: v2-alpha-ops (5 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| add_alpha | Set A=1.0 on RGB image | Trivial |
| remove_alpha | Drop alpha channel | Trivial |
| flatten | Composite over background | Porter-Duff over |
| match_color | Histogram matching | GPU histogram → transfer function |
| channel_mixer | Matrix multiply RGB | 3×3 matrix per pixel |

### P2 — Nice to Have (22 filters, 3 tracks)

#### Track: v2-draw-ops (8 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| draw_line | GPU rasterization | Bresenham in compute shader |
| draw_rect | Fill rect region | Trivial bounds check |
| draw_circle | Distance field | `length(p - center) < radius` |
| draw_ellipse | Distance field | Scaled distance |
| draw_arc | Angle + distance field | Polar check |
| draw_polygon | Winding number test | Per-pixel inside test |
| draw_text | Glyph atlas sampling | Pre-rasterize glyphs, blit |
| draw_text_ttf | TTF → glyph atlas → sample | Font rasterization + atlas |

#### Track: v2-generators (6 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| checkerboard | Coordinate math | `(x/size + y/size) % 2` |
| gradient_linear | Coordinate interpolation | `dot(pos, direction)` |
| gradient_radial | Distance from center | `length(pos - center)` |
| perlin_noise | GPU noise function | Permutation table in storage buffer |
| simplex_noise | GPU noise function | Simplex grid in shader |
| plasma | Multi-octave noise | Sum of noise at different frequencies |

#### Track: v2-evaluate-math (9 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| evaluate_add | `pixel + constant` | Trivial point op |
| evaluate_subtract | `pixel - constant` | Trivial point op |
| evaluate_multiply | `pixel * constant` | Trivial point op |
| evaluate_divide | `pixel / constant` | Trivial point op |
| evaluate_abs | `abs(pixel)` | Trivial point op |
| evaluate_pow | `pow(pixel, exp)` | Point op |
| evaluate_log | `log(pixel)` | Point op |
| evaluate_max | `max(pixel, threshold)` | Point op |
| evaluate_min | `min(pixel, threshold)` | Point op |

### P3 — Specialized (10 filters, 2 tracks)

#### Track: v2-tools (7 filters)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| clone_stamp | Region copy with mask | GPU texture copy + blend |
| healing_brush | Seamless clone (Poisson) | Iterative GPU solver |
| smudge | Directional pixel push | Accumulation buffer |
| sponge | Saturation brush | Local saturation adjustment |
| flood_fill | Seed-based region fill | GPU connected component labeling |
| red_eye_remove | Detect + desaturate red | Hue check + desaturate |
| ca_remove | Lateral chromatic aberr. | Per-channel offset + correction |

#### Track: v2-analysis-transform (6 filters + 3 already in V2)
| Filter | GPU Approach | Notes |
|--------|-------------|-------|
| harris_corners | Gradient matrix → response | Structure tensor + GPU reduction |
| hough_lines | Accumulator space | GPU vote → CPU peak detection |
| connected_components | Label propagation | Multi-pass GPU labeling |
| template_match | Normalized cross-correlation | Sliding window GPU |
| seam_carve_width | Energy + DP | Hybrid GPU energy + CPU path |
| seam_carve_height | Transpose + width carve | Same as width, transposed |
| smart_crop | Saliency → crop | GPU saliency map + CPU rect |
| perspective_correct | Homography remap | 3×3 matrix coordinate transform |
| perspective_warp | Arbitrary quad remap | Bilinear interpolation in quad |
| sparse_color | Scattered point interpolation | GPU Voronoi or IDW interpolation |

---

## Recommended Implementation Tracks (13 total)

| Priority | Track | Filters | Est. Effort |
|----------|-------|---------|-------------|
| P0 | v2-grading-curves | 12 | Large (LUT infrastructure) |
| P0 | v2-distortion-filters | 10 | Medium (coordinate remap shaders) |
| P0 | v2-spatial-advanced | 6 | Large (complex kernels) |
| P0 | v2-alpha-composite | 4 | Small (trivial shaders) |
| P1 | v2-morphology | 8 | Medium (multi-pass) |
| P1 | v2-edge-threshold | 8 | Medium (convolution + analysis) |
| P1 | v2-mask-ops | 7 | Medium (mask infrastructure) |
| P1 | v2-alpha-ops | 5 | Small (trivial shaders) |
| P2 | v2-draw-ops | 8 | Medium (rasterization) |
| P2 | v2-generators | 6 | Small (pure compute) |
| P2 | v2-evaluate-math | 9 | Small (trivial point ops) |
| P3 | v2-tools | 7 | Large (interactive tools) |
| P3 | v2-analysis-transform | 10 | Large (complex algorithms) |

**Total: 92 filters across 13 tracks.**
Restoring these brings V2 to 173 filters (81 existing + 92 restored).
Combined with 12 V2-only additions = **185 total** (exceeding V1's 179).

---

## Implementation Notes

All new filters must follow the **GPU-first** mandate:
1. **Primary:** WGSL compute shader via `gpu_shader_body()`
2. **Fallback:** CPU with SIMD-optimized loops (`[f32; 4]` chunks)
3. **Exception:** Only if total GPU wall-clock (upload+dispatch+readback) measurably exceeds SIMD CPU end-to-end

Evaluate math (P2) filters are trivially implementable as `PointOpExpr` expressions — they get GPU shaders for free via the fusion optimizer's WGSL lowering. These could be done in a single session.
