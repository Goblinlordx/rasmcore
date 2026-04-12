//! Extended grading and morphology reference implementations.
//!
//! All operations work in **linear f32** space (RGBA, 4 channels interleaved).
//! Alpha is preserved unchanged. Pure math — no SIMD, no GPU, no external crates.

use core::f32::consts::PI;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Deterministic hash → [0, 1) float. Same algorithm as lib.rs / effect_ops.rs.
#[inline]
fn hash_f32(mut x: u32) -> f32 {
    x = x.wrapping_mul(0x9e3779b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85ebca6b);
    x ^= x >> 13;
    (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

/// Luminance from linear RGB (BT.709 coefficients).
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Sample a pixel channel with clamped (replicate-border) addressing.
#[inline]
fn sample(input: &[f32], w: u32, h: u32, x: i32, y: i32, c: usize) -> f32 {
    let cx = x.clamp(0, w as i32 - 1) as u32;
    let cy = y.clamp(0, h as i32 - 1) as u32;
    input[((cy * w + cx) * 4 + c as u32) as usize]
}

// ─── 1. Reinhard Tonemap ──────────────────────────────────────────────────────

/// Reinhard tonemapping — simple per-channel operator.
///
/// Formula: `L = channel * exposure; out = L / (1 + L)`
///
/// Maps HDR values to [0, 1) range. exposure=0 maps everything to 0.
///
/// Validated against: reference Reinhard 2002 operator.
pub fn tonemap_reinhard(input: &[f32], _w: u32, _h: u32, exposure: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let l = px[c] * exposure;
            px[c] = l / (1.0 + l);
        }
    }
    out
}

// ─── 2. Filmic Tonemap (Uncharted 2) ─────────────────────────────────────────

/// Uncharted 2 filmic tonemapping.
///
/// Attempt to produce a film-like response curve. Uses the Hable/Uncharted 2
/// formula with standard constants:
///   A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30
///
/// `f(x) = ((x*(A*x+C*B)+D*E) / (x*(A*x+B)+D*F)) - E/F`
///
/// Output is `f(channel * exposure) / f(W)` where W=11.2 (white point).
///
/// Validated against: Uncharted 2 GDC presentation reference.
pub fn tonemap_filmic(input: &[f32], _w: u32, _h: u32, exposure: f32) -> Vec<f32> {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    const W: f32 = 11.2;

    #[inline]
    fn uncharted2(x: f32) -> f32 {
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }

    let white_scale = 1.0 / uncharted2(W);

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let mapped = uncharted2(px[c] * exposure);
            px[c] = mapped * white_scale;
        }
    }
    out
}

// ─── 3. Drago Tonemap ─────────────────────────────────────────────────────────

/// Drago logarithmic tonemapping.
///
/// Formula per channel:
///   `out = log(1 + channel * exposure) / log(1 + max_luminance * exposure)`
///
/// where max_luminance is the maximum luminance across the entire image.
/// This normalizes the output so the brightest pixel maps to 1.0.
///
/// Validated against: Drago 2003 logarithmic mapping operator.
pub fn tonemap_drago(input: &[f32], w: u32, h: u32, exposure: f32) -> Vec<f32> {
    let n = (w * h) as usize;

    // Compute max luminance across the image
    let mut max_lum = 0.0f32;
    for i in 0..n {
        let idx = i * 4;
        let lum = luminance(input[idx], input[idx + 1], input[idx + 2]);
        if lum > max_lum {
            max_lum = lum;
        }
    }

    let denom = (1.0 + max_lum * exposure).ln();
    // Avoid division by zero if image is all black and exposure is 0
    let denom = if denom.abs() < 1e-10 { 1.0 } else { denom };

    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            px[c] = (1.0 + px[c] * exposure).ln() / denom;
        }
    }
    out
}

// ─── 4. Film Grain with Size ──────────────────────────────────────────────────

/// Film grain with spatial grain size control.
///
/// Same as film_grain but noise is scaled spatially: coordinates are divided
/// by `size` before hashing, producing larger grain clumps when size > 1.
/// Uses Box-Muller for Gaussian noise, weighted by luminance midtone curve
/// `4 * luma * (1 - luma)`.
///
/// Deterministic based on seed.
pub fn film_grain_grading(
    input: &[f32],
    w: u32,
    h: u32,
    amount: f32,
    size: f32,
    seed: u32,
) -> Vec<f32> {
    let size = size.max(1.0);
    let mut out = input.to_vec();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let r = input[idx * 4];
            let g = input[idx * 4 + 1];
            let b = input[idx * 4 + 2];
            let luma = luminance(r, g, b).clamp(0.0, 1.0);
            let weight = 4.0 * luma * (1.0 - luma);

            // Quantize coordinates by grain size for spatial coherence
            let gx = (x as f32 / size).floor() as u32;
            let gy = (y as f32 / size).floor() as u32;
            let grain_key = seed
                .wrapping_add(gx.wrapping_mul(0x45d9f3b))
                .wrapping_add(gy.wrapping_mul(0x6c62272e));

            let u1 = hash_f32(grain_key.wrapping_mul(2)).max(1e-10);
            let u2 = hash_f32(grain_key.wrapping_mul(2).wrapping_add(1));
            let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let noise = gauss * amount * weight;

            out[idx * 4] = (r + noise).clamp(0.0, 1.0);
            out[idx * 4 + 1] = (g + noise).clamp(0.0, 1.0);
            out[idx * 4 + 2] = (b + noise).clamp(0.0, 1.0);
            // alpha unchanged
        }
    }
    out
}

// ─── 5. Morphological Top-hat ─────────────────────────────────────────────────

/// Helper: per-channel dilate (max in square window).
fn dilate(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let r = radius as i32;
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            for c in 0..3usize {
                let mut max_val = f32::NEG_INFINITY;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let v = sample(input, w, h, x as i32 + dx, y as i32 + dy, c);
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                out[idx + c] = max_val;
            }
            out[idx + 3] = input[idx + 3];
        }
    }
    out
}

/// Helper: per-channel erode (min in square window).
fn erode(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let r = radius as i32;
    let mut out = vec![0.0f32; input.len()];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            for c in 0..3usize {
                let mut min_val = f32::INFINITY;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let v = sample(input, w, h, x as i32 + dx, y as i32 + dy, c);
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                out[idx + c] = min_val;
            }
            out[idx + 3] = input[idx + 3];
        }
    }
    out
}

/// Morphological top-hat: original - opening.
///
/// Top-hat extracts small bright features smaller than the structuring element.
/// opening = dilate(erode(input)).
///
/// Validated against: ImageMagick `-morphology TopHat Square:{radius}`
pub fn morph_tophat(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let eroded = erode(input, w, h, radius);
    let opening = dilate(&eroded, w, h, radius);

    let mut out = vec![0.0f32; input.len()];
    for i in 0..(input.len() / 4) {
        let idx = i * 4;
        for c in 0..3 {
            out[idx + c] = input[idx + c] - opening[idx + c];
        }
        out[idx + 3] = input[idx + 3];
    }
    out
}

// ─── 6. Morphological Black-hat ───────────────────────────────────────────────

/// Morphological black-hat: closing - original.
///
/// Black-hat extracts small dark features smaller than the structuring element.
/// closing = erode(dilate(input)).
///
/// Validated against: ImageMagick `-morphology BottomHat Square:{radius}`
pub fn morph_blackhat(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let dilated = dilate(input, w, h, radius);
    let closing = erode(&dilated, w, h, radius);

    let mut out = vec![0.0f32; input.len()];
    for i in 0..(input.len() / 4) {
        let idx = i * 4;
        for c in 0..3 {
            out[idx + c] = closing[idx + c] - input[idx + c];
        }
        out[idx + 3] = input[idx + 3];
    }
    out
}

// ─── 7. Morphological Gradient ────────────────────────────────────────────────

/// Morphological gradient: dilate - erode.
///
/// Highlights edges/boundaries by computing the difference between dilation
/// and erosion.
///
/// Validated against: ImageMagick `-morphology Gradient Square:{radius}`
pub fn morph_gradient(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let dilated = dilate(input, w, h, radius);
    let eroded = erode(input, w, h, radius);

    let mut out = vec![0.0f32; input.len()];
    for i in 0..(input.len() / 4) {
        let idx = i * 4;
        for c in 0..3 {
            out[idx + c] = dilated[idx + c] - eroded[idx + c];
        }
        out[idx + 3] = input[idx + 3];
    }
    out
}

// ─── 8. Skeletonize (Zhang-Suen) ─────────────────────────────────────────────

/// Zhang-Suen thinning / skeletonization on binarized luminance.
///
/// 1. Binarize: pixel is foreground if luminance >= 0.5.
/// 2. Iteratively thin using the Zhang-Suen two-sub-iteration algorithm
///    until no more pixels can be removed.
/// 3. Output: 1.0 for skeleton pixels, 0.0 for background (same in R,G,B).
///
/// The algorithm preserves connectivity and topology of the original shape.
///
/// Validated against: OpenCV `cv::ximgproc::thinning` with THINNING_ZHANGSUEN.
pub fn skeletonize(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let n = w * h;

    // Step 1: Binarize based on luminance threshold 0.5
    let mut img = vec![false; n];
    for i in 0..n {
        let idx = i * 4;
        let lum = luminance(input[idx], input[idx + 1], input[idx + 2]);
        img[i] = lum >= 0.5;
    }

    // Zhang-Suen helper: get pixel value with bounds check (false for out-of-bounds)
    let get = |grid: &[bool], x: i32, y: i32| -> bool {
        if x < 0 || x >= w as i32 || y < 0 || y >= h as i32 {
            false
        } else {
            grid[y as usize * w + x as usize]
        }
    };

    // Zhang-Suen: iterate until convergence
    loop {
        let mut changed = false;

        // Sub-iteration 1
        let mut to_remove = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if !img[y * w + x] {
                    continue;
                }
                let xi = x as i32;
                let yi = y as i32;

                // 8-neighbors: P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
                let p2 = get(&img, xi, yi - 1);
                let p3 = get(&img, xi + 1, yi - 1);
                let p4 = get(&img, xi + 1, yi);
                let p5 = get(&img, xi + 1, yi + 1);
                let p6 = get(&img, xi, yi + 1);
                let p7 = get(&img, xi - 1, yi + 1);
                let p8 = get(&img, xi - 1, yi);
                let p9 = get(&img, xi - 1, yi - 1);

                let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];

                // B(P1) = number of non-zero neighbors
                let b: u32 = neighbors.iter().map(|&v| v as u32).sum();
                if !(2..=6).contains(&b) {
                    continue;
                }

                // A(P1) = number of 0→1 transitions in the ordered sequence P2..P9..P2
                let mut a = 0u32;
                for i in 0..8 {
                    if !neighbors[i] && neighbors[(i + 1) % 8] {
                        a += 1;
                    }
                }
                if a != 1 {
                    continue;
                }

                // Conditions for sub-iteration 1:
                // P2 * P4 * P6 == 0
                // P4 * P6 * P8 == 0
                if (p2 && p4 && p6) || (p4 && p6 && p8) {
                    continue;
                }

                to_remove.push(y * w + x);
            }
        }
        for &idx in &to_remove {
            img[idx] = false;
            changed = true;
        }

        // Sub-iteration 2
        let mut to_remove = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if !img[y * w + x] {
                    continue;
                }
                let xi = x as i32;
                let yi = y as i32;

                let p2 = get(&img, xi, yi - 1);
                let p3 = get(&img, xi + 1, yi - 1);
                let p4 = get(&img, xi + 1, yi);
                let p5 = get(&img, xi + 1, yi + 1);
                let p6 = get(&img, xi, yi + 1);
                let p7 = get(&img, xi - 1, yi + 1);
                let p8 = get(&img, xi - 1, yi);
                let p9 = get(&img, xi - 1, yi - 1);

                let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];

                let b: u32 = neighbors.iter().map(|&v| v as u32).sum();
                if !(2..=6).contains(&b) {
                    continue;
                }

                let mut a = 0u32;
                for i in 0..8 {
                    if !neighbors[i] && neighbors[(i + 1) % 8] {
                        a += 1;
                    }
                }
                if a != 1 {
                    continue;
                }

                // Conditions for sub-iteration 2:
                // P2 * P4 * P8 == 0
                // P2 * P6 * P8 == 0
                if (p2 && p4 && p8) || (p2 && p6 && p8) {
                    continue;
                }

                to_remove.push(y * w + x);
            }
        }
        for &idx in &to_remove {
            img[idx] = false;
            changed = true;
        }

        if !changed {
            break;
        }
    }

    // Build RGBA output
    let mut out = vec![0.0f32; n * 4];
    for i in 0..n {
        let v = if img[i] { 1.0 } else { 0.0 };
        out[i * 4] = v;
        out[i * 4 + 1] = v;
        out[i * 4 + 2] = v;
        out[i * 4 + 3] = input[i * 4 + 3];
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn solid(w: u32, h: u32, r: f32, g: f32, b: f32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            out.extend_from_slice(&[r, g, b, 1.0]);
        }
        out
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn reinhard_exposure_zero_maps_to_zero() {
        let input = crate::gradient(8, 8);
        let result = tonemap_reinhard(&input, 8, 8, 0.0);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL, "reinhard with exposure=0 should map to 0");
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
            assert!((px[3] - 1.0).abs() < TOL, "alpha must be preserved");
        }
    }

    #[test]
    fn reinhard_preserves_zero_input() {
        let input = solid(4, 4, 0.0, 0.0, 0.0);
        let result = tonemap_reinhard(&input, 4, 4, 2.0);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL);
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
        }
    }

    #[test]
    fn filmic_preserves_alpha() {
        let input = crate::solid(4, 4, [0.5, 0.5, 0.5, 0.7]);
        let result = tonemap_filmic(&input, 4, 4, 1.0);
        for px in result.chunks_exact(4) {
            assert!((px[3] - 0.7).abs() < TOL, "alpha must be preserved");
        }
    }

    #[test]
    fn filmic_output_in_range() {
        let input = crate::gradient(16, 16);
        let result = tonemap_filmic(&input, 16, 16, 2.0);
        for px in result.chunks_exact(4) {
            for c in 0..3 {
                assert!(px[c] >= -0.01, "filmic output should be near-non-negative");
                assert!(px[c] <= 1.01, "filmic output should not exceed ~1.0");
            }
        }
    }

    #[test]
    fn drago_preserves_alpha() {
        let input = crate::solid(4, 4, [0.5, 0.3, 0.7, 0.8]);
        let result = tonemap_drago(&input, 4, 4, 1.0);
        for px in result.chunks_exact(4) {
            assert!((px[3] - 0.8).abs() < TOL, "alpha must be preserved");
        }
    }

    #[test]
    fn drago_exposure_zero_maps_to_zero() {
        let input = crate::gradient(8, 8);
        let result = tonemap_drago(&input, 8, 8, 0.0);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL, "drago with exposure=0 should map to 0");
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
        }
    }

    #[test]
    fn film_grain_grading_deterministic() {
        let input = solid(8, 8, 0.5, 0.5, 0.5);
        let a = film_grain_grading(&input, 8, 8, 0.1, 2.0, 42);
        let b = film_grain_grading(&input, 8, 8, 0.1, 2.0, 42);
        assert_eq!(a, b, "same seed must produce identical output");
    }

    #[test]
    fn film_grain_grading_amount_zero_identity() {
        let input = solid(8, 8, 0.5, 0.5, 0.5);
        let result = film_grain_grading(&input, 8, 8, 0.0, 2.0, 42);
        // With amount=0, noise contribution is 0, but weight might make it
        // effectively zero anyway. The luma weighting on midtones is nonzero
        // but amount=0 means gauss*0*weight = 0.
        assert!(
            max_diff(&input, &result) < TOL,
            "amount=0 should be identity"
        );
    }

    #[test]
    fn morph_gradient_on_solid_is_zero() {
        let input = solid(8, 8, 0.5, 0.3, 0.7);
        let result = morph_gradient(&input, 8, 8, 2);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL, "gradient on solid should be 0");
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
        }
    }

    #[test]
    fn morph_tophat_on_solid_is_zero() {
        let input = solid(8, 8, 0.5, 0.3, 0.7);
        let result = morph_tophat(&input, 8, 8, 2);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL, "tophat on solid should be 0");
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
        }
    }

    #[test]
    fn morph_blackhat_on_solid_is_zero() {
        let input = solid(8, 8, 0.5, 0.3, 0.7);
        let result = morph_blackhat(&input, 8, 8, 2);
        for px in result.chunks_exact(4) {
            assert!(px[0].abs() < TOL, "blackhat on solid should be 0");
            assert!(px[1].abs() < TOL);
            assert!(px[2].abs() < TOL);
        }
    }

    #[test]
    fn morph_gradient_preserves_alpha() {
        let mut input = solid(4, 4, 0.5, 0.5, 0.5);
        input[3] = 0.3; // first pixel
        let result = morph_gradient(&input, 4, 4, 1);
        assert!((result[3] - 0.3).abs() < TOL, "alpha must be preserved");
    }

    #[test]
    fn skeletonize_preserves_single_pixel_line() {
        // Create a horizontal line 1 pixel wide in the middle of a 9x9 image.
        // The skeleton of a single-pixel-wide line should be the line itself.
        let w = 9u32;
        let h = 9u32;
        let mut input = vec![0.0f32; (w * h * 4) as usize];
        // Set alpha to 1.0 for all pixels
        for i in 0..(w * h) as usize {
            input[i * 4 + 3] = 1.0;
        }
        // Draw horizontal line at y=4 from x=1 to x=7
        let y = 4;
        for x in 1..=7u32 {
            let idx = (y * w + x) as usize * 4;
            input[idx] = 1.0;
            input[idx + 1] = 1.0;
            input[idx + 2] = 1.0;
        }

        let result = skeletonize(&input, w, h);

        // The single-pixel-wide horizontal line should remain (interior pixels
        // preserved; endpoints might be trimmed by Zhang-Suen but the core line stays).
        // Check that at least the interior of the line (x=2..6) is preserved.
        for x in 2..=6u32 {
            let idx = (y * w + x) as usize * 4;
            assert!(
                result[idx] > 0.5,
                "skeleton should preserve single-pixel line at x={x}"
            );
        }
    }

    #[test]
    fn skeletonize_empty_image() {
        let input = solid(8, 8, 0.0, 0.0, 0.0);
        let result = skeletonize(&input, 8, 8);
        for px in result.chunks_exact(4) {
            assert_eq!(px[0], 0.0, "empty image skeleton should be all zero");
            assert_eq!(px[1], 0.0);
            assert_eq!(px[2], 0.0);
        }
    }

    #[test]
    fn skeletonize_preserves_alpha() {
        let input = crate::solid(4, 4, [0.0, 0.0, 0.0, 0.6]);
        let result = skeletonize(&input, 4, 4);
        for px in result.chunks_exact(4) {
            assert!((px[3] - 0.6).abs() < TOL, "alpha must be preserved");
        }
    }
}
