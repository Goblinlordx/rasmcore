# Filter Audit Report

Tests both code paths:
- **Direct**: ParamMap built from defaults (Rust-only, always works)
- **Binary**: Params serialized → deserialized (what the playground does)

## Summary

| Metric | Count |
|--------|-------|
| Total filters | 81 |
| Both paths pass | 81 |
| Binary-only failure | 0 |
| Both paths fail | 0 |
| Param serialization mismatch | 0 |
| Output differs between paths | 0 |

## Passing Filters (81/81)

- scope_parade (analysis)
- scope_histogram (analysis)
- scope_waveform (analysis)
- scope_vectorscope (analysis)
- lift_gamma_gain ()
- tonemap_filmic (grading)
- film_grain_grading (grading)
- tonemap_reinhard (grading)
- tonemap_drago (grading)
- split_toning ()
- asc_cdl ()
- film_grain (effect)
- emboss (effect)
- light_leak (effect)
- chromatic_aberration (effect)
- poisson_noise (effect)
- halftone (effect)
- chromatic_split (effect)
- salt_pepper_noise (effect)
- charcoal (effect)
- glitch (effect)
- uniform_noise (effect)
- pixelate (effect)
- mirror_kaleidoscope (effect)
- gaussian_noise (effect)
- oil_paint (effect)
- sigmoidal_contrast (adjustment)
- solarize (adjustment)
- brightness (adjustment)
- posterize (adjustment)
- contrast (adjustment)
- levels (adjustment)
- gamma (adjustment)
- burn (adjustment)
- invert (adjustment)
- exposure (adjustment)
- dodge (adjustment)
- median (spatial)
- high_pass (spatial)
- bilateral (spatial)
- gaussian_blur (spatial)
- sharpen (spatial)
- motion_blur (spatial)
- box_blur (spatial)
- frequency_low (enhancement)
- clarity (enhancement)
- nlm_denoise (enhancement)
- auto_level (enhancement)
- retinex_msr (enhancement)
- vignette (enhancement)
- normalize (enhancement)
- pyramid_detail_remap (enhancement)
- retinex_msrcr (enhancement)
- dehaze (enhancement)
- shadow_highlight (enhancement)
- frequency_high (enhancement)
- clahe (enhancement)
- vignette_powerlaw (enhancement)
- equalize (enhancement)
- retinex_ssr (enhancement)
- aces_cg_to_cct (color)
- colorize (color)
- selective_color (color)
- modulate (color)
- aces_odt (color)
- vibrance (color)
- saturate (color)
- aces_idt (color)
- dither_floyd_steinberg (color)
- aces_cct_to_cg (color)
- white_balance_temperature (color)
- white_balance_gray_world (color)
- kmeans_quantize (color)
- replace_color (color)
- quantize (color)
- sepia (color)
- lab_sharpen (color)
- photo_filter (color)
- dither_ordered (color)
- lab_adjust (color)
- hue_rotate (color)

## Limitations

This audit tests the **Rust binary serialization round-trip**: encode params
in the same format the JS playground uses, deserialize back, apply filter.
This verifies the Rust-side deserializer and filter implementations work
correctly with the binary wire format.

**What this does NOT test:**
- The actual JS `serializeParams()` code in the docs playground
- Browser-specific behavior (Float32Array endianness, TypedArray alignment)
- The JCO-transpiled WIT boundary (JS ↔ WASM type lifting/lowering)
- Edge cases like: slider at min/max bounds, rapid value changes, NaN from division

**To test the full JS path**, a browser-based test harness is needed that:
1. Loads the WASM module in a browser/Node.js environment
2. For each filter, calls `applyFilter(source, name, serializeParams(defaults))`
3. Compares the result against the Rust-verified baseline

The Rust-side round-trip passing for all 81 filters means: if a filter is
broken in the playground, the bug is in the JS serialization or JCO boundary,
not in the filter implementation or Rust deserializer.
