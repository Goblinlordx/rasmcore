# Pro Filter Validation Report

Date: 2026-03-30
Platform: macOS Darwin 25.3.0, Apple Silicon

## Reference Parity Results

All formula-based filters validated against Python numpy reference implementations
using the same algorithms. Test file: `crates/rasmcore-image/tests/pro_filter_parity.rs`

| Filter | Reference | Algorithm | MAE | max_err | Verdict |
|---|---|---|---|---|---|
| asc_cdl | Python numpy | `clamp((in * slope + offset) ^ power)` per ITU-R BT.1886 | < 0.5 | -- | PASS |
| asc_cdl (identity) | Self-test | Default params = no-op | < 0.5 | -- | PASS |
| lift_gamma_gain | Python numpy | `gain * (in + lift*(1-in)) ^ (1/gamma)` DaVinci formula | < 0.5 | -- | PASS |
| split_toning | Python numpy | Luminance-based shadow/highlight blend, Rec.709 weights | < 1.0 | -- | PASS |
| curves_master | Python numpy | Fritsch-Carlson monotone cubic Hermite spline LUT | 0.0000 | 0 | EXACT |
| curves_red/green/blue | Channel isolation | Apply to one channel, verify others unchanged | exact | 0 | EXACT |
| film_grain | Determinism | Same seed = identical output; midtone_mae > shadow_mae | exact | 0 | PASS |
| tonemap_reinhard | Python numpy | `L / (1 + L)` per-channel (Reinhard 2002) | < 0.5 | -- | PASS |
| tonemap_drago | Python numpy | `log(1+L)/log(1+Lmax)` with bias power (Drago 2003) | < 1.0 | -- | PASS |
| tonemap_filmic | Python numpy | Narkowicz ACES: `(x*(a*x+b))/(x*(c*x+d)+e)` (2015) | < 0.5 | -- | PASS |
| selective_color | Python (pure) | HSL hue-range selection + cosine taper + shift/sat/light | < 1.0 | -- | PASS |
| smart_crop | Behavioral | Output 64x64, variance > 100 (selects detailed region) | -- | -- | PASS |
| seam_carve_width | Dimensional | 64->48 width, correct buffer size, PSNR > 5dB vs naive | -- | -- | PASS |
| seam_carve_height | Dimensional | 64->48 height, correct buffer size | -- | -- | PASS |

### Validation tiers

- **EXACT**: Pixel-identical output (MAE=0, max_err=0). Both implementations use the same algorithm.
- **PASS (MAE < N)**: Formula-exact implementations agree within floating-point rounding (u8 quantization).
- **Behavioral/Dimensional**: No pixel-level reference exists. Validated via output properties (dimensions, variance, content preservation).

### Why some filters lack pixel-exact references

- **Smart crop**: Implementation-dependent heuristic (entropy/attention scoring). The existing `smart_crop_vs_vips_reference` test in `smart_crop.rs` compares against libvips when available.
- **Seam carving**: Seam selection depends on energy function and DP tiebreaking. ImageMagick `liquid-rescale` uses different energy (LQIP vs Sobel). Correct behavior = preserving high-energy content, not matching a specific reference pixel-for-pixel.
- **Film grain**: Hash-based noise with custom midtone emphasis. No external standard. Determinism is the meaningful property.

## Benchmark Results

Profile: release (criterion), sample size: 20

| Filter | 256x256 | 1024x1024 | ns/pixel (256) | ns/pixel (1024) |
|---|---|---|---|---|
| asc_cdl | 562 us | 8.9 ms | 8.6 | 8.5 |
| lift_gamma_gain | 644 us | 10.2 ms | 9.8 | 9.7 |
| split_toning | 357 us | 5.7 ms | 5.4 | 5.4 |
| curves | 296 us | 4.4 ms | 4.5 | 4.2 |
| film_grain | 226 us | 3.6 ms | 3.4 | 3.4 |
| tonemap_reinhard | 181 us | 2.9 ms | 2.8 | 2.8 |
| tonemap_drago | 1.09 ms | 17.4 ms | 16.6 | 16.6 |
| tonemap_filmic | 276 us | 4.4 ms | 4.2 | 4.2 |
| selective_color | 877 us | 11.8 ms | 13.4 | 11.2 |
| smart_crop | 56 us | 406 us | 0.9 | 0.4 |
| seam_carve_width (256 only) | 16.3 ms | -- | 249.0 | -- |
| seam_carve_height (256 only) | 16.5 ms | -- | 251.7 | -- |

### Notes

- Point-op filters (CDL, LGG, tonemap) scale linearly with pixel count
- Drago is ~4x slower than Reinhard due to per-pixel log+pow operations
- Smart crop is very fast due to downsampled analysis (~32px working resolution)
- Seam carving is O(n*seams) — benchmarked at 256 only (16ms for 25% reduction)
- Curves is LUT-based, making it one of the fastest grading operations
