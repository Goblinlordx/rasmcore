//! Enhancement operation reference implementations — histogram-based global and local contrast ops.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged.
//! Histograms quantize f32 values to 256 bins via `(v * 255.0).round() as usize`.
//! Each function documents the formula and external validation source.

const NUM_BINS: usize = 256;

/// Quantize an f32 value in [0,1] to a histogram bin index in [0,255].
fn to_bin(v: f32) -> usize {
    (v.clamp(0.0, 1.0) * 255.0).round() as usize
}

/// Build a 256-bin histogram for a single channel from RGBA interleaved data.
/// `channel` is 0=R, 1=G, 2=B.
fn build_histogram(input: &[f32], channel: usize) -> [u32; NUM_BINS] {
    let mut hist = [0u32; NUM_BINS];
    for px in input.chunks_exact(4) {
        let bin = to_bin(px[channel]);
        hist[bin] += 1;
    }
    hist
}

/// Build a cumulative distribution function from a histogram.
fn build_cdf(hist: &[u32; NUM_BINS]) -> [u32; NUM_BINS] {
    let mut cdf = [0u32; NUM_BINS];
    cdf[0] = hist[0];
    for i in 1..NUM_BINS {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    cdf
}

/// Histogram equalization.
///
/// Per channel independently: quantize to 256 bins, build CDF, remap each
/// pixel so that `out[c] = cdf[c][bin] / total_pixels`.
///
/// Validated against: ImageMagick 7.1.1 `-colorspace Linear -equalize`
pub fn equalize(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let total = (w * h) as f32;
    let mut cdfs = [[0u32; NUM_BINS]; 3];
    for c in 0..3 {
        let hist = build_histogram(input, c);
        cdfs[c] = build_cdf(&hist);
    }

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let bin = to_bin(px[c]);
            px[c] = cdfs[c][bin] as f32 / total;
        }
        // alpha unchanged
    }
    out
}

/// Normalize to full range per channel.
///
/// Per channel: find min and max, remap `out[c] = (in[c] - min) / (max - min)`.
/// If max == min, output 0.0 for that channel.
///
/// Validated against: ImageMagick 7.1.1 `-colorspace Linear -normalize`
pub fn normalize(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut mins = [f32::MAX; 3];
    let mut maxs = [f32::MIN; 3];

    for px in input.chunks_exact(4) {
        for c in 0..3 {
            mins[c] = mins[c].min(px[c]);
            maxs[c] = maxs[c].max(px[c]);
        }
    }

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let range = maxs[c] - mins[c];
            if range == 0.0 {
                px[c] = 0.0;
            } else {
                px[c] = (px[c] - mins[c]) / range;
            }
        }
    }
    out
}

/// Auto levels with clip percentage.
///
/// Per channel: build histogram, find the bin where cumulative count first
/// reaches `clip_pct`% of total pixels from the low end (new black point),
/// and from the high end (new white point). Remap that range to [0,1].
///
/// `clip_pct` is in percent (e.g., 1.0 = 1%).
///
/// Validated against: ImageMagick 7.1.1 `-colorspace Linear -auto-level`
pub fn auto_level(input: &[f32], w: u32, h: u32, clip_pct: f32) -> Vec<f32> {
    let total = (w * h) as f32;
    let clip_count = total * clip_pct / 100.0;

    let mut lo = [0usize; 3];
    let mut hi = [NUM_BINS - 1; 3];

    for c in 0..3 {
        let hist = build_histogram(input, c);
        let cdf = build_cdf(&hist);

        // Find low clip point: first bin where CDF > clip_count
        for i in 0..NUM_BINS {
            if cdf[i] as f32 > clip_count {
                lo[c] = i;
                break;
            }
        }

        // Find high clip point: last bin where (total - CDF) > clip_count
        for i in (0..NUM_BINS).rev() {
            let above = total - cdf[i] as f32;
            if above > clip_count {
                hi[c] = i;
                break;
            }
        }
    }

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let lo_val = lo[c] as f32 / 255.0;
            let hi_val = hi[c] as f32 / 255.0;
            let range = hi_val - lo_val;
            if range <= 0.0 {
                px[c] = 0.0;
            } else {
                px[c] = ((px[c] - lo_val) / range).clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// Contrast Limited Adaptive Histogram Equalization (CLAHE).
///
/// Divide the image into `grid_size x grid_size` tiles. For each tile:
/// - Build a 256-bin histogram
/// - Clip histogram at `clip_limit` and redistribute excess evenly
/// - Build CDF from clipped histogram
///
/// For each pixel, bilinear interpolate between the CDFs of the 4 nearest
/// tile centers.
///
/// Validated against: OpenCV `cv::createCLAHE(clipLimit, Size(grid,grid))`
pub fn clahe(input: &[f32], w: u32, h: u32, clip_limit: f32, grid_size: u32) -> Vec<f32> {
    let gw = grid_size as usize;
    let gh = grid_size as usize;
    let width = w as usize;
    let height = h as usize;

    // Tile dimensions (float for precise boundary computation).
    let tile_w = width as f64 / gw as f64;
    let tile_h = height as f64 / gh as f64;

    // Build per-tile CDFs for each channel.
    // cdfs[channel][tile_row][tile_col][bin] -> CDF value as f32 in [0,1].
    let mut cdfs: Vec<Vec<Vec<[f32; NUM_BINS]>>> = vec![
        vec![vec![[0.0f32; NUM_BINS]; gw]; gh];
        3
    ];

    for ty in 0..gh {
        for tx in 0..gw {
            // Pixel boundaries for this tile.
            let x0 = (tx as f64 * tile_w).round() as usize;
            let x1 = (((tx + 1) as f64) * tile_w).round() as usize;
            let y0 = (ty as f64 * tile_h).round() as usize;
            let y1 = (((ty + 1) as f64) * tile_h).round() as usize;

            let tile_pixels = ((x1 - x0) * (y1 - y0)) as f32;
            if tile_pixels == 0.0 {
                continue;
            }

            // Actual clip limit in histogram counts.
            // clip_limit is a multiplier of the "uniform" count per bin.
            let uniform_count = tile_pixels / NUM_BINS as f32;
            let abs_clip = (clip_limit * uniform_count).max(1.0);

            for c in 0..3 {
                // Build histogram for this tile and channel.
                let mut hist = [0u32; NUM_BINS];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = (y * width + x) * 4 + c;
                        let bin = to_bin(input[idx]);
                        hist[bin] += 1;
                    }
                }

                // Clip and redistribute.
                let mut excess = 0u32;
                for bin in 0..NUM_BINS {
                    if hist[bin] as f32 > abs_clip {
                        excess += hist[bin] - abs_clip as u32;
                        hist[bin] = abs_clip as u32;
                    }
                }
                let per_bin = excess / NUM_BINS as u32;
                let remainder = (excess % NUM_BINS as u32) as usize;
                for bin in 0..NUM_BINS {
                    hist[bin] += per_bin;
                    if bin < remainder {
                        hist[bin] += 1;
                    }
                }

                // Build CDF normalized to [0,1].
                let cdf_raw = build_cdf(&hist);
                let total = cdf_raw[NUM_BINS - 1] as f32;
                for bin in 0..NUM_BINS {
                    cdfs[c][ty][tx][bin] = if total > 0.0 {
                        cdf_raw[bin] as f32 / total
                    } else {
                        bin as f32 / (NUM_BINS - 1) as f32
                    };
                }
            }
        }
    }

    // Map each pixel using bilinear interpolation of the 4 nearest tile CDFs.
    let mut out = input.to_vec();

    for y in 0..height {
        for x in 0..width {
            // Position of this pixel relative to tile centers.
            // Tile centers are at (tx + 0.5) * tile_w, (ty + 0.5) * tile_h.
            let fx = (x as f64 + 0.5) / tile_w - 0.5;
            let fy = (y as f64 + 0.5) / tile_h - 0.5;

            let tx0 = (fx.floor() as isize).max(0).min(gw as isize - 1) as usize;
            let ty0 = (fy.floor() as isize).max(0).min(gh as isize - 1) as usize;
            let tx1 = (tx0 + 1).min(gw - 1);
            let ty1 = (ty0 + 1).min(gh - 1);

            let ax = (fx - tx0 as f64).max(0.0).min(1.0) as f32;
            let ay = (fy - ty0 as f64).max(0.0).min(1.0) as f32;

            let px_idx = (y * width + x) * 4;

            for c in 0..3 {
                let bin = to_bin(input[px_idx + c]);
                let v00 = cdfs[c][ty0][tx0][bin];
                let v10 = cdfs[c][ty0][tx1][bin];
                let v01 = cdfs[c][ty1][tx0][bin];
                let v11 = cdfs[c][ty1][tx1][bin];

                let top = v00 * (1.0 - ax) + v10 * ax;
                let bot = v01 * (1.0 - ax) + v11 * ax;
                out[px_idx + c] = top * (1.0 - ay) + bot * ay;
            }
            // alpha unchanged
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a gradient where each channel goes from 0.0 to 1.0
    /// uniformly across all pixels. Specifically, pixel i has value i/(n-1).
    fn uniform_gradient(n: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.5 };
            data.push(v); // R
            data.push(v); // G
            data.push(v); // B
            data.push(1.0); // A
        }
        data
    }

    /// Helper: max absolute difference between two buffers, ignoring alpha.
    fn max_rgb_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .flat_map(|(pa, pb)| {
                (0..3).map(move |c| (pa[c] - pb[c]).abs())
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn equalize_uniform_gradient_approx_identity() {
        // A uniform gradient has a flat histogram, so equalization
        // should be approximately the identity transform.
        let w = 256u32;
        let h = 1u32;
        let input = uniform_gradient((w * h) as usize);
        let result = equalize(&input, w, h);

        let diff = max_rgb_diff(&input, &result);
        // Quantization to 256 bins introduces up to ~1/256 error.
        assert!(
            diff < 0.02,
            "equalize on uniform gradient should be near identity, got max diff {diff}"
        );
    }

    #[test]
    fn equalize_preserves_alpha() {
        let input = vec![0.2, 0.4, 0.6, 0.77, 0.8, 0.5, 0.3, 0.99];
        let result = equalize(&input, 2, 1);
        assert_eq!(result[3], 0.77);
        assert_eq!(result[7], 0.99);
    }

    #[test]
    fn normalize_maps_subrange_to_full() {
        // Input: all channels in [0.2, 0.8] range.
        let w = 4u32;
        let h = 1u32;
        let input = vec![
            0.2, 0.2, 0.2, 1.0,
            0.4, 0.4, 0.4, 1.0,
            0.6, 0.6, 0.6, 1.0,
            0.8, 0.8, 0.8, 1.0,
        ];
        let result = normalize(&input, w, h);

        // Min was 0.2, max was 0.8 => range 0.6
        // 0.2 -> 0.0, 0.8 -> 1.0
        let eps = 1e-6;
        assert!((result[0] - 0.0).abs() < eps, "min should map to 0.0");
        assert!((result[12] - 1.0).abs() < eps, "max should map to 1.0");

        // 0.4 -> (0.4-0.2)/0.6 = 1/3
        assert!((result[4] - 1.0 / 3.0).abs() < eps);
        // 0.6 -> (0.6-0.2)/0.6 = 2/3
        assert!((result[8] - 2.0 / 3.0).abs() < eps);
    }

    #[test]
    fn normalize_constant_image_yields_zero() {
        let input = vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0,
                         0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0];
        let result = normalize(&input, 2, 2);
        for px in result.chunks_exact(4) {
            for c in 0..3 {
                assert_eq!(px[c], 0.0, "constant image channel should be 0.0");
            }
            assert_eq!(px[3], 1.0, "alpha preserved");
        }
    }

    #[test]
    fn normalize_preserves_alpha() {
        let input = vec![0.0, 0.0, 0.0, 0.42, 1.0, 1.0, 1.0, 0.88];
        let result = normalize(&input, 2, 1);
        assert_eq!(result[3], 0.42);
        assert_eq!(result[7], 0.88);
    }

    #[test]
    fn auto_level_zero_clip_approx_normalize() {
        // With 0% clipping, auto_level should behave like normalize
        // (up to quantization error from the 256-bin histogram).
        let w = 64u32;
        let h = 1u32;
        let mut input = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            let v = 0.2 + 0.6 * (i as f32 / (w * h - 1) as f32);
            input.extend_from_slice(&[v, v, v, 1.0]);
        }

        let auto_result = auto_level(&input, w, h, 0.0);
        let norm_result = normalize(&input, w, h);

        let diff = max_rgb_diff(&auto_result, &norm_result);
        // Quantization bins are ~1/255 wide, so error can be up to ~1/128
        assert!(
            diff < 0.02,
            "auto_level(clip=0) should approximate normalize, got max diff {diff}"
        );
    }

    #[test]
    fn auto_level_preserves_alpha() {
        let input = vec![0.0, 0.0, 0.0, 0.33, 1.0, 1.0, 1.0, 0.67];
        let result = auto_level(&input, 2, 1, 0.0);
        assert_eq!(result[3], 0.33);
        assert_eq!(result[7], 0.67);
    }

    #[test]
    fn clahe_high_clip_approx_equalize() {
        // With a very high clip limit, no histogram clipping occurs,
        // so CLAHE should approximate plain equalization (with some
        // difference from tile boundary interpolation).
        let w = 64u32;
        let h = 64u32;
        let input = uniform_gradient((w * h) as usize);

        let eq_result = equalize(&input, w, h);
        // Very high clip limit effectively disables clipping.
        // grid_size=1 means a single tile covering the whole image,
        // so no interpolation artifacts.
        let clahe_result = clahe(&input, w, h, 1000.0, 1);

        let diff = max_rgb_diff(&eq_result, &clahe_result);
        assert!(
            diff < 0.02,
            "clahe with high clip_limit and grid=1 should approximate equalize, got max diff {diff}"
        );
    }

    #[test]
    fn clahe_preserves_alpha() {
        let input = vec![
            0.0, 0.0, 0.0, 0.11,
            0.5, 0.5, 0.5, 0.22,
            0.25, 0.25, 0.25, 0.33,
            1.0, 1.0, 1.0, 0.44,
        ];
        let result = clahe(&input, 2, 2, 2.0, 2);
        assert_eq!(result[3], 0.11);
        assert_eq!(result[7], 0.22);
        assert_eq!(result[11], 0.33);
        assert_eq!(result[15], 0.44);
    }

    #[test]
    fn clahe_grid_subdivides_image() {
        // Smoke test: CLAHE with grid_size=4 on a 32x32 image should
        // produce valid output (no panics, values in [0,1]).
        let w = 32u32;
        let h = 32u32;
        let input = crate::noise(w, h, 99);
        let result = clahe(&input, w, h, 3.0, 4);

        assert_eq!(result.len(), input.len());
        for px in result.chunks_exact(4) {
            for c in 0..3 {
                assert!(
                    (0.0..=1.0).contains(&px[c]),
                    "output pixel out of range: {}", px[c]
                );
            }
        }
    }
}
