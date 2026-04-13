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
/// pixel so that `out[c] = (cdf[c][bin] - cdf_min) / (N - cdf_min)`.
///
/// Standard histogram equalization formula matching numpy/OpenCV.
pub fn equalize(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let npixels = (w * h) as u32;
    let mut luts = [[0.0f32; NUM_BINS]; 3];

    // Use truncation binning (matching pipeline): floor(clamp(v,0,inf)*255)
    let trunc_bin = |v: f32| -> usize { ((v.max(0.0) * 255.0) as usize).min(255) };

    for c in 0..3 {
        let mut hist = [0u32; NUM_BINS];
        for px in input.chunks_exact(4) {
            hist[trunc_bin(px[c])] += 1;
        }
        let cdf = build_cdf(&hist);
        let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
        let denom = npixels.saturating_sub(cdf_min).max(1) as f32;
        for i in 0..NUM_BINS {
            luts[c][i] = (cdf[i].saturating_sub(cdf_min)) as f32 / denom;
        }
    }

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            px[c] = luts[c][trunc_bin(px[c])];
        }
    }
    out
}

/// Normalize — linear contrast stretch with percentile clipping.
///
/// Per channel: build 256-bin histogram (truncation binning), find percentile
/// clip points for black and white, linearly remap. No output clamping.
pub fn normalize(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    // Default: 2% black, 1% white (matching pipeline defaults)
    normalize_clipped(input, _w, _h, 0.02, 0.01)
}

/// Normalize with explicit clip percentages.
pub fn normalize_clipped(input: &[f32], _w: u32, _h: u32, black_clip: f32, white_clip: f32) -> Vec<f32> {
    let npixels = input.len() / 4;
    let mut out = input.to_vec();

    for c in 0..3 {
        let mut hist = [0u32; NUM_BINS];
        for px in input.chunks_exact(4) {
            let bin = ((px[c].max(0.0) * 255.0) as usize).min(255);
            hist[bin] += 1;
        }

        // Find black point
        let black_threshold = (npixels as f32 * black_clip) as u32;
        let mut accum = 0u32;
        let mut black_bin = 0;
        for (i, &h) in hist.iter().enumerate() {
            accum += h;
            if accum >= black_threshold {
                black_bin = i;
                break;
            }
        }

        // Find white point
        let white_threshold = (npixels as f32 * white_clip) as u32;
        accum = 0;
        let mut white_bin = 255;
        for i in (0..NUM_BINS).rev() {
            accum += hist[i];
            if accum >= white_threshold {
                white_bin = i;
                break;
            }
        }

        let black = black_bin as f32 / 255.0;
        let white = white_bin as f32 / 255.0;
        let range = white - black;

        if range > 1e-10 {
            for px in out.chunks_exact_mut(4) {
                px[c] = (px[c] - black) / range;
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
/// Matches OpenCV's cv2.createCLAHE algorithm exactly:
/// 1. BT.709 luminance → u8
/// 2. Per-tile: histogram, clip with strided redistribution, CDF → u8 LUT
/// 3. Bilinear interpolation between u8 LUTs (tile centers at x/tw - 0.5)
/// 4. Apply luma ratio to RGB
pub fn clahe(input: &[f32], w: u32, h: u32, clip_limit: f32, grid_size: u32) -> Vec<f32> {
    let grid = grid_size as usize;
    let width = w as usize;
    let height = h as usize;

    if grid == 0 { return input.to_vec(); }

    let tile_w = width / grid;
    let tile_h = height / grid;
    if tile_w == 0 || tile_h == 0 { return input.to_vec(); }

    let npixels_per_tile = tile_w * tile_h;
    let clip = (clip_limit * npixels_per_tile as f32 / 256.0).max(1.0) as u32;
    let lut_scale = 255.0 / npixels_per_tile as f32;

    // Compute luminance and quantize to u8
    let luma: Vec<f32> = input.chunks_exact(4)
        .map(|p| (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).clamp(0.0, 1.0))
        .collect();
    let luma_u8: Vec<u8> = luma.iter()
        .map(|&v| (v * 255.0 + 0.5).min(255.0) as u8)
        .collect();

    // Build per-tile LUTs as u8 (matching OpenCV)
    let mut tile_luts = vec![[0u8; NUM_BINS]; grid * grid];
    for ty in 0..grid {
        for tx in 0..grid {
            let mut hist = [0u32; NUM_BINS];
            let y0 = ty * tile_h;
            let x0 = tx * tile_w;
            for dy in 0..tile_h {
                for dx in 0..tile_w {
                    let py = (y0 + dy).min(height - 1);
                    let px = (x0 + dx).min(width - 1);
                    hist[luma_u8[py * width + px] as usize] += 1;
                }
            }

            // Clip and redistribute (OpenCV single-pass with strided residual)
            let mut excess = 0u32;
            for h in &mut hist {
                if *h > clip {
                    excess += *h - clip;
                    *h = clip;
                }
            }
            let redist_batch = excess / 256;
            let residual = excess - redist_batch * 256;
            for h in hist.iter_mut() {
                *h += redist_batch;
            }
            if residual > 0 {
                let step = (256 / residual).max(1) as usize;
                let mut remaining = residual;
                let mut i = 0;
                while i < NUM_BINS && remaining > 0 {
                    hist[i] += 1;
                    remaining -= 1;
                    i += step;
                }
            }

            // CDF → u8 LUT: round(CDF * 255 / total)
            let mut cdf_sum = 0u32;
            let lut = &mut tile_luts[ty * grid + tx];
            for i in 0..NUM_BINS {
                cdf_sum += hist[i];
                lut[i] = (cdf_sum as f32 * lut_scale + 0.5).min(255.0) as u8;
            }
        }
    }

    // Bilinear interpolation (OpenCV: txf = x / tileWidth - 0.5)
    let inv_tw = 1.0f32 / tile_w as f32;
    let inv_th = 1.0f32 / tile_h as f32;

    let mut out = input.to_vec();
    for y in 0..height {
        let tyf = y as f32 * inv_th - 0.5;
        let ty1 = (tyf.floor() as i32).max(0).min(grid as i32 - 1) as usize;
        let ty2 = (ty1 + 1).min(grid - 1);
        let ya = (tyf - ty1 as f32).clamp(0.0, 1.0);
        let ya1 = 1.0 - ya;

        for x in 0..width {
            let txf = x as f32 * inv_tw - 0.5;
            let tx1 = (txf.floor() as i32).max(0).min(grid as i32 - 1) as usize;
            let tx2 = (tx1 + 1).min(grid - 1);
            let xa = (txf - tx1 as f32).clamp(0.0, 1.0);
            let xa1 = 1.0 - xa;

            let bin = luma_u8[y * width + x] as usize;
            let v_top = tile_luts[ty1 * grid + tx1][bin] as f32 * xa1
                + tile_luts[ty1 * grid + tx2][bin] as f32 * xa;
            let v_bot = tile_luts[ty2 * grid + tx1][bin] as f32 * xa1
                + tile_luts[ty2 * grid + tx2][bin] as f32 * xa;
            let new_luma = (v_top * ya1 + v_bot * ya) / 255.0;

            let old_luma = luma[y * width + x].max(1e-10);
            let ratio = new_luma / old_luma;

            let idx = (y * width + x) * 4;
            out[idx] *= ratio;
            out[idx + 1] *= ratio;
            out[idx + 2] *= ratio;
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
    fn normalize_with_clip_stretches() {
        // With small clip percentages, normalize should stretch the range
        let w = 256u32;
        let h = 1u32;
        let mut input = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            let v = i as f32 / (w * h - 1) as f32; // full [0,1] range
            input.extend_from_slice(&[v, v, v, 1.0]);
        }
        // With 1% clip on each end, result should still cover near [0,1]
        let result = normalize_clipped(&input, w, h, 0.01, 0.01);
        assert!(result[0] < 0.0, "darkest pixels should map below 0 with clip");
        let last = ((w * h - 1) * 4) as usize;
        assert!(result[last] > 1.0, "brightest pixels should map above 1 with clip");
    }

    #[test]
    fn normalize_constant_image_unchanged() {
        let input = vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0,
                         0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0];
        let result = normalize(&input, 2, 2);
        // With histogram-based approach and 0 range, values stay the same
        for px in result.chunks_exact(4) {
            for c in 0..3 {
                assert_eq!(px[c], 0.5, "constant image channel should be unchanged");
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
    fn auto_level_basic_stretch() {
        // Verify auto_level stretches a sub-range image
        let w = 64u32;
        let h = 1u32;
        let mut input = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            let v = 0.2 + 0.6 * (i as f32 / (w * h - 1) as f32);
            input.extend_from_slice(&[v, v, v, 1.0]);
        }

        let result = auto_level(&input, w, h, 0.0);
        // First pixel should be near 0, last near 1
        assert!(result[0] < 0.05, "min should map near 0.0, got {}", result[0]);
        let last = ((w * h - 1) * 4) as usize;
        assert!(result[last] > 0.95, "max should map near 1.0, got {}", result[last]);
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
        // produce valid output (no panics, reasonable values).
        let w = 32u32;
        let h = 32u32;
        let input = crate::noise(w, h, 99);
        let result = clahe(&input, w, h, 3.0, 4);

        assert_eq!(result.len(), input.len());
        // Luma-ratio CLAHE can produce values > 1.0 for saturated pixels
        // but should not produce NaN or extreme values
        for px in result.chunks_exact(4) {
            for c in 0..3 {
                assert!(!px[c].is_nan(), "output pixel is NaN");
                assert!(px[c] >= -1.0 && px[c] <= 10.0, "output pixel out of range: {}", px[c]);
            }
        }
    }
}
