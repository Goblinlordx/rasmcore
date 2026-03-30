# Pro Filter Benchmark Report

Date: 2026-03-30
Platform: macOS Darwin 25.3.0, Apple Silicon
Profile: release (criterion)
Sample size: 20

## Results

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

## Notes

- Point-op filters (CDL, LGG, tonemap) scale linearly with pixel count
- Drago is ~4x slower than Reinhard due to per-pixel log+pow operations
- Smart crop is very fast due to downsampled analysis (~32px working resolution)
- Seam carving is O(n*seams) — benchmarked at 256 only (16ms for 25% reduction)
- Curves is LUT-based, making it one of the fastest grading operations
