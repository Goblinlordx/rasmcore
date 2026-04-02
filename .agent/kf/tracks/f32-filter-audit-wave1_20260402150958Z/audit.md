# f32 Filter Audit — All Filters by Category

**Date:** 2026-04-03
**Total filters:** 159 (excluding generators, tool)

## Legend

- **Pattern:** `point_op` = LUT-based, `color_op` = ColorOp/CLUT, `rgb_tx` = apply_rgb_transform, `custom` = hand-written per-pixel, `spatial` = needs neighbor pixels
- **f32 Ready:** Current state of f32 support
- **Migration:** `trivial` = add f32 branch to existing infra, `easy` = straightforward per-pixel f32 path, `moderate` = needs spatial/multi-pass rework, `hard` = external deps or complex state, `n/a` = not applicable
- **Wave:** 1 = this track, 2 = spatial/complex, 3 = hard/deferred

---

## Adjustment (10 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| brightness | point_op | No | trivial | — | 1 |
| contrast | point_op | No | trivial | — | 1 |
| gamma | point_op | No | trivial | — | 1 |
| exposure | point_op | No | trivial | — | 1 |
| levels | point_op | No | trivial | — | 1 |
| sigmoidal_contrast | point_op | No | trivial | — | 1 |
| posterize | point_op | No | trivial | — | 1 |
| invert | point_op | No | trivial | — | 1 |
| invert_v2 | custom | No | easy | — | 1 |
| color_balance | color_op | No | trivial | — | 1 |

## Color (19 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| hue_rotate | color_op | No | trivial | — | 1 |
| saturate | color_op | No | trivial | — | 1 |
| vibrance | color_op | No | trivial | — | 1 |
| modulate | color_op | No | trivial | — | 1 |
| sepia | color_op | No | trivial | — | 1 |
| channel_mixer | color_op | No | trivial | — | 1 |
| colorize | color_op | No | trivial | — | 1 |
| photo_filter | custom | No | easy | — | 1 |
| selective_color | custom | No | easy | — | 1 |
| white_balance_temperature | custom | No | easy | — | 1 |
| white_balance_gray_world | custom | No | moderate | global stats | 2 |
| grayscale | custom | No | easy | — | 2 |
| gradient_map | custom | No | easy | — | 2 |
| dither_floyd_steinberg | custom | No | moderate | error diffusion | 3 |
| dither_ordered | custom | No | moderate | bayer matrix | 3 |
| kmeans_quantize | custom | No | hard | iterative clustering | 3 |
| quantize | custom | No | moderate | color table | 3 |
| sparse_color | custom | No | moderate | spatial interp | 2 |
| lab | color_op | No | easy | — | 2 |

## Evaluate (9 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| evaluate_add | point_op | No | trivial | — | 1 |
| evaluate_subtract | point_op | No | trivial | — | 1 |
| evaluate_multiply | point_op | No | trivial | — | 1 |
| evaluate_divide | point_op | No | trivial | — | 1 |
| evaluate_min | point_op | No | trivial | — | 1 |
| evaluate_max | point_op | No | trivial | — | 1 |
| evaluate_pow | point_op | No | trivial | — | 1 |
| evaluate_log | point_op | No | trivial | — | 1 |
| evaluate_abs | custom | No | trivial | — | 1 |

## Grading (13 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| curves_master | rgb_tx | No | easy | — | 1 |
| curves_red | rgb_tx | No | easy | — | 1 |
| curves_green | rgb_tx | No | easy | — | 1 |
| curves_blue | rgb_tx | No | easy | — | 1 |
| lift_gamma_gain | color_op | No | trivial | — | 1 |
| asc_cdl | color_op | No | trivial | — | 1 |
| split_toning | color_op | No | trivial | — | 1 |
| hue_vs_sat | rgb_tx | No | easy | — | 1 |
| hue_vs_lum | rgb_tx | No | easy | — | 1 |
| lum_vs_sat | rgb_tx | No | easy | — | 1 |
| sat_vs_sat | rgb_tx | No | easy | — | 1 |
| apply_cube_lut | custom | No | moderate | CLUT apply | 2 |
| apply_hald_lut | custom | No | moderate | CLUT apply | 2 |

## Tonemapping (3 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| tonemap_reinhard | rgb_tx | No | easy | — | 1 |
| tonemap_filmic | rgb_tx | No | easy | — | 1 |
| tonemap_drago | rgb_tx | No | easy | — | 1 |

## Enhancement (18 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| dodge | custom | No | easy | — | 1 |
| burn | custom | No | easy | — | 1 |
| vignette | custom | No | easy | — | 1 |
| vignette_powerlaw | custom | No | easy | — | 1 |
| shadow_highlight | custom | No | moderate | multi-pass | 2 |
| auto_level | custom | No | moderate | global stats | 2 |
| equalize | custom | No | moderate | histogram | 2 |
| normalize | custom | No | moderate | global stats | 2 |
| clahe | custom | No | hard | tiled histogram | 3 |
| clarity | custom | No | moderate | multi-scale | 2 |
| dehaze | custom | No | moderate | dark channel | 2 |
| frequency_high | custom | No | moderate | FFT-based | 2 |
| frequency_low | custom | No | moderate | FFT-based | 2 |
| retinex_ssr | custom | No | moderate | Gaussian spatial | 2 |
| retinex_msr | custom | No | moderate | multi-scale spatial | 2 |
| retinex_msrcr | custom | No | moderate | multi-scale spatial | 2 |
| nlm_denoise | custom | No | hard | spatial search | 3 |
| pyramid_detail_remap | custom | No | hard | Laplacian pyramid | 3 |

## Effect (11 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| solarize | point_op | No | trivial | — | 1 |
| emboss | custom | No | moderate | 3x3 kernel | 2 |
| charcoal | custom | No | moderate | multi-pass spatial | 2 |
| film_grain | custom | No | easy | — | 2 |
| gaussian_noise | custom | No | easy | — | 2 |
| uniform_noise | custom | No | easy | — | 2 |
| poisson_noise | custom | No | easy | — | 2 |
| salt_pepper_noise | custom | No | easy | — | 2 |
| halftone | custom | No | moderate | pattern spatial | 2 |
| oil_paint | custom | No | hard | radius histogram | 3 |
| pixelate | custom | No | easy | — | 2 |

## Spatial (20 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| blur | spatial | No | moderate | Gaussian kernel | 2 |
| box_blur | spatial | No | moderate | box kernel | 2 |
| average_blur | spatial | No | moderate | spatial | 2 |
| gaussian_blur_cv | spatial | No | moderate | separable kernel | 2 |
| bilateral | spatial | No | hard | range+spatial kernel | 3 |
| bokeh_blur | spatial | No | hard | disc kernel FFT | 3 |
| lens_blur | spatial | No | hard | depth-dependent | 3 |
| median | spatial | No | moderate | histogram window | 2 |
| motion_blur | spatial | No | moderate | directional kernel | 2 |
| sharpen | spatial | No | moderate | USM kernel | 2 |
| smart_sharpen | spatial | No | moderate | multi-pass | 2 |
| high_pass | spatial | No | moderate | subtract blur | 2 |
| kuwahara | spatial | No | hard | local stats | 3 |
| rank_filter | spatial | No | moderate | histogram window | 2 |
| convolve | custom | No | moderate | arbitrary kernel | 2 |
| displacement_map | custom | No | moderate | secondary input | 2 |
| spin_blur | spatial | No | moderate | angular sampling | 2 |
| tilt_shift | spatial | No | moderate | gradient blur | 2 |
| zoom_blur | spatial | No | moderate | radial sampling | 2 |
| guided_filter | spatial | No | hard | box integral | 3 |

## Edge Detection (4 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| sobel | custom | No | moderate | 3x3 gradient | 2 |
| scharr | custom | No | moderate | 3x3 gradient | 2 |
| laplacian | custom | No | moderate | 3x3 kernel | 2 |
| canny | custom | No | hard | multi-stage pipeline | 3 |

## Morphology (8 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| dilate | custom | No | moderate | structuring element | 2 |
| erode | custom | No | moderate | structuring element | 2 |
| morph_open | custom | No | moderate | erode+dilate | 2 |
| morph_close | custom | No | moderate | dilate+erode | 2 |
| morph_gradient | custom | No | moderate | dilate-erode | 2 |
| morph_tophat | custom | No | moderate | open subtract | 2 |
| morph_blackhat | custom | No | moderate | close subtract | 2 |
| skeletonize | custom | No | hard | Zhang-Suen binary | 3 |

## Threshold (4 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| threshold_binary | custom | No | easy | — | 2 |
| otsu_threshold | custom | No | moderate | histogram | 2 |
| adaptive_threshold | custom | No | moderate | local mean | 2 |
| triangle_threshold | custom | No | moderate | histogram | 2 |

## Analysis (4 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| harris_corners | custom | No | hard | needs f32 buffer | 3 |
| connected_components | custom | No | hard | label propagation | 3 |
| hough_lines | custom | No | hard | accumulator array | 3 |
| template_match | custom | No | hard | cross-correlation | 3 |

## Alpha (5 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| premultiply | custom | No | easy | — | 2 |
| unpremultiply | custom | No | easy | — | 2 |
| add_alpha | custom | No | easy | — | 2 |
| remove_alpha | custom | No | easy | — | 2 |
| flatten | custom | No | easy | — | 2 |

## Composite (3 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| blend | custom | No | moderate | two-input | 2 |
| blend_if | custom | No | moderate | two-input + ranges | 2 |
| mask_apply | custom | No | moderate | two-input | 2 |

## Distortion (8 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| wave | custom | No | moderate | bilinear interp | 2 |
| ripple | custom | No | moderate | bilinear interp | 2 |
| barrel | custom | No | moderate | bilinear interp | 2 |
| polar | custom | No | moderate | coordinate remap | 2 |
| depolar | custom | No | moderate | coordinate remap | 2 |
| swirl | custom | No | moderate | bilinear interp | 2 |
| spherize | custom | No | moderate | bilinear interp | 2 |
| mesh_warp | custom | No | moderate | quad interp | 2 |

## Advanced (2 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| perspective_warp | custom | No | moderate | bilinear interp | 2 |
| perspective_correct | custom | No | moderate | bilinear interp | 2 |

## Transform (3 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| smart_crop | custom | No | hard | energy analysis | 3 |
| seam_carve_width | custom | No | hard | energy seams | 3 |
| seam_carve_height | custom | No | hard | energy seams | 3 |

## Draw (8 filters)

| Filter | Pattern | f32 Ready | Migration | Blocker | Wave |
|--------|---------|-----------|-----------|---------|------|
| draw_rect | custom | No | easy | — | 2 |
| draw_circle | custom | No | easy | — | 2 |
| draw_ellipse | custom | No | easy | — | 2 |
| draw_line | custom | No | easy | — | 2 |
| draw_polygon | custom | No | easy | — | 2 |
| draw_arc | custom | No | easy | — | 2 |
| draw_text | custom | No | moderate | font rendering | 2 |
| draw_text_ttf | custom | No | moderate | font rendering | 2 |

---

## Wave Summary

| Wave | Count | Description |
|------|-------|-------------|
| 1 | 47 | Point ops, adjustment, color ops, evaluate, grading, tonemap, simple per-pixel |
| 2 | 78 | Spatial filters, distortion, noise, composite, alpha, draw, moderate complexity |
| 3 | 34 | Hard: external deps, complex state, multi-stage pipelines |

## Wave 1 Targets (47 filters)

### Infrastructure changes required:
1. **point_ops.rs** — Add `apply_point_op_f32()` that applies the f32 math directly (no LUT)
2. **apply_color_op()** — Add f32 format branch (skip u8/u16 conversion, pass samples directly)
3. **apply_rgb_transform()** — Add f32 format branch (skip u8 conversion, read/write f32 samples)

### Wave 1 filter list:
- **Adjustment (10):** brightness, contrast, gamma, exposure, levels, sigmoidal_contrast, posterize, invert, invert_v2, color_balance
- **Color (10):** hue_rotate, saturate, vibrance, modulate, sepia, channel_mixer, colorize, photo_filter, selective_color, white_balance_temperature
- **Evaluate (9):** evaluate_add/subtract/multiply/divide/min/max/pow/log/abs
- **Grading (11):** curves_master/red/green/blue, lift_gamma_gain, asc_cdl, split_toning, hue_vs_sat/lum, lum_vs_sat, sat_vs_sat
- **Tonemapping (3):** tonemap_reinhard, tonemap_filmic, tonemap_drago
- **Enhancement (4):** dodge, burn, vignette, vignette_powerlaw

### Migration approach:
- **point_op filters:** Add `is_f32()` check before LUT path → call new `apply_point_op_f32()` which applies the f32 math directly to each sample
- **color_op filters:** Add f32 branch to `apply_color_op()` — samples are already f32, just read/apply/write
- **rgb_tx filters:** Add f32 branch to `apply_rgb_transform()` — samples are already in [0,1], skip conversion
- **custom filters (dodge, burn, vignette, etc.):** Add `is_f32()` branch that reads f32 samples and applies same math
