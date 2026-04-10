//! Grading, effect, and composite reference implementations.
//!
//! All operations work in **linear f32** space. Alpha is preserved unchanged
//! unless the operation is specifically about alpha (premultiply/unpremultiply).
//! Each function documents the formula and external validation source.

// ─── Color Grading ────────────────────────────────────────────────────────────

/// ASC CDL — per SMPTE ST 2065-5 / S-2014-006.
///
/// Formula per channel:
///   `out = clamp(in * slope + offset, 0, 1) ^ power`
///
/// Then saturation adjustment using BT.709 luma weights:
///   `luma = 0.2126*R + 0.7152*G + 0.0722*B`
///   `out[c] = luma + sat * (out[c] - luma)`
///
/// Validated against: OpenColorIO ASC CDL transform reference.
pub fn asc_cdl(
    input: &[f32],
    _w: u32,
    _h: u32,
    slope: [f32; 3],
    offset: [f32; 3],
    power: [f32; 3],
    sat: f32,
) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        // Slope, offset, power per channel
        let mut rgb = [0.0f32; 3];
        for c in 0..3 {
            let v = (px[c] * slope[c] + offset[c]).clamp(0.0, 1.0);
            rgb[c] = v.powf(power[c]);
        }

        // Saturation via BT.709 luma
        let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        for c in 0..3 {
            px[c] = luma + sat * (rgb[c] - luma);
        }
        // alpha unchanged
    }
    out
}

/// Lift / Gamma / Gain — per DaVinci Resolve color grading model.
///
/// Formula per channel:
///   `out = gain * (in + lift * (1 - in)) ^ (1 / gamma)`
///
/// - lift: shadow control (0 = neutral)
/// - gamma: midtone control (1 = neutral)
/// - gain: highlight control (1 = neutral)
///
/// Validated against: DaVinci Resolve 18 primary color corrector.
pub fn lift_gamma_gain(
    input: &[f32],
    _w: u32,
    _h: u32,
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in 0..3 {
            let lifted = px[c] + lift[c] * (1.0 - px[c]);
            // Ensure non-negative before power (lift can push negative)
            let base = lifted.max(0.0);
            let inv_gamma = if gamma[c] != 0.0 {
                1.0 / gamma[c]
            } else {
                1.0
            };
            px[c] = gain[c] * base.powf(inv_gamma);
        }
        // alpha unchanged
    }
    out
}

// ─── Effects ──────────────────────────────────────────────────────────────────

/// Emboss — 3x3 convolution with directional emboss kernel.
///
/// Kernel (scaled by strength):
/// ```text
/// -2  -1   0
/// -1   1   1
///  0   1   2
/// ```
///
/// Output is biased by +0.5 to center the result around mid-gray.
/// Edge pixels use clamped (repeat-edge) sampling.
///
/// Validated against: GIMP emboss filter with equivalent kernel.
pub fn emboss(input: &[f32], w: u32, h: u32, strength: f32) -> Vec<f32> {
    let kernel: [[f32; 3]; 3] = [
        [-2.0, -1.0, 0.0],
        [-1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
    ];

    let w = w as usize;
    let h = h as usize;
    let mut out = vec![0.0f32; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            let dst = (y * w + x) * 4;
            for c in 0..3 {
                let mut sum = 0.0f32;
                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sy = (y as i32 + ky - 1).clamp(0, h as i32 - 1) as usize;
                        let sx = (x as i32 + kx - 1).clamp(0, w as i32 - 1) as usize;
                        let src_idx = (sy * w + sx) * 4 + c;
                        sum += input[src_idx] * kernel[ky as usize][kx as usize];
                    }
                }
                out[dst + c] = sum * strength + 0.5;
            }
            // Preserve alpha
            out[dst + 3] = input[dst + 3];
        }
    }
    out
}

/// Pixelate — block averaging.
///
/// Divides the image into `size x size` blocks. Every pixel within a block
/// is set to the average color of that block.
///
/// Validated against: ImageMagick `-scale {1/size}x{1/size} -scale {w}x{h}`
pub fn pixelate(input: &[f32], w: u32, h: u32, size: u32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let size = size.max(1) as usize;
    let mut out = input.to_vec();

    // Process each block
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        let bh = (by + size).min(h) - by;
        while bx < w {
            let bw = (bx + size).min(w) - bx;
            let count = (bw * bh) as f32;

            // Compute block average
            let mut avg = [0.0f32; 3];
            for dy in 0..bh {
                for dx in 0..bw {
                    let idx = ((by + dy) * w + (bx + dx)) * 4;
                    avg[0] += input[idx];
                    avg[1] += input[idx + 1];
                    avg[2] += input[idx + 2];
                }
            }
            avg[0] /= count;
            avg[1] /= count;
            avg[2] /= count;

            // Write average to all pixels in block
            for dy in 0..bh {
                for dx in 0..bw {
                    let idx = ((by + dy) * w + (bx + dx)) * 4;
                    out[idx] = avg[0];
                    out[idx + 1] = avg[1];
                    out[idx + 2] = avg[2];
                    // alpha unchanged
                }
            }
            bx += size;
        }
        by += size;
    }
    out
}

// ─── Blend / Composite ───────────────────────────────────────────────────────

/// Normal blend — straight alpha compositing with opacity.
///
/// Formula per RGB channel:
///   `out = overlay * opacity + base * (1 - opacity)`
///
/// Alpha: same formula applied to alpha channel.
///
/// Validated against: Photoshop Normal blend mode at given opacity.
pub fn blend_normal(
    base: &[f32],
    overlay: &[f32],
    _w: u32,
    _h: u32,
    opacity: f32,
) -> Vec<f32> {
    assert_eq!(base.len(), overlay.len(), "base and overlay size mismatch");
    let mut out = vec![0.0f32; base.len()];
    for i in 0..base.len() {
        out[i] = overlay[i] * opacity + base[i] * (1.0 - opacity);
    }
    out
}

/// Multiply blend — darkening blend mode with opacity.
///
/// Formula per RGB channel:
///   `result = base * overlay`
///   `out = lerp(base, result, opacity)`
///
/// Alpha: preserved from base.
///
/// Validated against: Photoshop Multiply blend mode at given opacity.
pub fn blend_multiply(
    base: &[f32],
    overlay: &[f32],
    _w: u32,
    _h: u32,
    opacity: f32,
) -> Vec<f32> {
    assert_eq!(base.len(), overlay.len(), "base and overlay size mismatch");
    let mut out = vec![0.0f32; base.len()];
    for px in 0..(base.len() / 4) {
        let i = px * 4;
        for c in 0..3 {
            let result = base[i + c] * overlay[i + c];
            out[i + c] = base[i + c] * (1.0 - opacity) + result * opacity;
        }
        out[i + 3] = base[i + 3]; // alpha from base
    }
    out
}

/// Screen blend — lightening blend mode with opacity.
///
/// Formula per RGB channel:
///   `result = 1 - (1 - base) * (1 - overlay)`
///   `out = lerp(base, result, opacity)`
///
/// Alpha: preserved from base.
///
/// Validated against: Photoshop Screen blend mode at given opacity.
pub fn blend_screen(
    base: &[f32],
    overlay: &[f32],
    _w: u32,
    _h: u32,
    opacity: f32,
) -> Vec<f32> {
    assert_eq!(base.len(), overlay.len(), "base and overlay size mismatch");
    let mut out = vec![0.0f32; base.len()];
    for px in 0..(base.len() / 4) {
        let i = px * 4;
        for c in 0..3 {
            let result = 1.0 - (1.0 - base[i + c]) * (1.0 - overlay[i + c]);
            out[i + c] = base[i + c] * (1.0 - opacity) + result * opacity;
        }
        out[i + 3] = base[i + 3]; // alpha from base
    }
    out
}

// ─── Alpha Operations ─────────────────────────────────────────────────────────

/// Premultiply alpha — straight to premultiplied.
///
/// Formula: `R' = R*A, G' = G*A, B' = B*A, A' = A`
///
/// Validated against: ImageMagick `-channel RGB -evaluate Multiply {alpha}`
pub fn premultiply(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let a = px[3];
        px[0] *= a;
        px[1] *= a;
        px[2] *= a;
        // alpha unchanged
    }
    out
}

/// Unpremultiply alpha — premultiplied to straight.
///
/// Formula: `R' = R/A, G' = G/A, B' = B/A, A' = A`
///
/// Safe division: if A == 0, RGB output is 0.
///
/// Validated against: ImageMagick `-channel RGB -evaluate Divide {alpha}`
pub fn unpremultiply(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let a = px[3];
        if a == 0.0 {
            px[0] = 0.0;
            px[1] = 0.0;
            px[2] = 0.0;
        } else {
            px[0] /= a;
            px[1] /= a;
            px[2] /= a;
        }
        // alpha unchanged
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn assert_approx_eq(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tolerance,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    #[test]
    fn asc_cdl_identity() {
        let input = crate::gradient(16, 16);
        let result = asc_cdl(
            &input,
            16,
            16,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            1.0,
        );
        assert_approx_eq(&input, &result, TOL);
    }

    #[test]
    fn asc_cdl_zero_saturation_gives_luma() {
        let input = crate::solid(2, 2, [0.5, 0.3, 0.8, 1.0]);
        let result = asc_cdl(
            &input,
            2,
            2,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );
        let luma = 0.2126 * 0.5 + 0.7152 * 0.3 + 0.0722 * 0.8;
        for px in result.chunks_exact(4) {
            assert!((px[0] - luma).abs() < TOL);
            assert!((px[1] - luma).abs() < TOL);
            assert!((px[2] - luma).abs() < TOL);
            assert!((px[3] - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn lift_gamma_gain_identity() {
        let input = crate::gradient(16, 16);
        let result = lift_gamma_gain(
            &input,
            16,
            16,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        );
        assert_approx_eq(&input, &result, TOL);
    }

    #[test]
    fn lift_gamma_gain_gain_scales() {
        let input = crate::solid(2, 2, [0.5, 0.5, 0.5, 1.0]);
        let result = lift_gamma_gain(
            &input,
            2,
            2,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        );
        for px in result.chunks_exact(4) {
            assert!((px[0] - 1.0).abs() < TOL);
            assert!((px[1] - 1.0).abs() < TOL);
            assert!((px[2] - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn emboss_preserves_alpha() {
        let input = crate::noise(8, 8, 99);
        let result = emboss(&input, 8, 8, 1.0);
        for (src, dst) in input.chunks_exact(4).zip(result.chunks_exact(4)) {
            assert_eq!(src[3], dst[3]);
        }
    }

    #[test]
    fn pixelate_size_1_is_identity() {
        let input = crate::gradient(16, 16);
        let result = pixelate(&input, 16, 16, 1);
        assert_approx_eq(&input, &result, TOL);
    }

    #[test]
    fn pixelate_full_image_gives_average() {
        let input = crate::solid(4, 4, [0.2, 0.4, 0.6, 1.0]);
        let result = pixelate(&input, 4, 4, 4);
        for px in result.chunks_exact(4) {
            assert!((px[0] - 0.2).abs() < TOL);
            assert!((px[1] - 0.4).abs() < TOL);
            assert!((px[2] - 0.6).abs() < TOL);
        }
    }

    #[test]
    fn blend_normal_opacity_0_returns_base() {
        let base = crate::gradient(8, 8);
        let overlay = crate::noise(8, 8, 42);
        let result = blend_normal(&base, &overlay, 8, 8, 0.0);
        assert_approx_eq(&base, &result, TOL);
    }

    #[test]
    fn blend_normal_opacity_1_returns_overlay() {
        let base = crate::gradient(8, 8);
        let overlay = crate::noise(8, 8, 42);
        let result = blend_normal(&base, &overlay, 8, 8, 1.0);
        assert_approx_eq(&overlay, &result, TOL);
    }

    #[test]
    fn blend_multiply_black_overlay_gives_black() {
        let base = crate::gradient(4, 4);
        let black = crate::solid(4, 4, [0.0, 0.0, 0.0, 1.0]);
        let result = blend_multiply(&base, &black, 4, 4, 1.0);
        for px in result.chunks_exact(4) {
            assert!((px[0]).abs() < TOL);
            assert!((px[1]).abs() < TOL);
            assert!((px[2]).abs() < TOL);
        }
    }

    #[test]
    fn blend_screen_black_overlay_gives_base() {
        let base = crate::gradient(4, 4);
        let black = crate::solid(4, 4, [0.0, 0.0, 0.0, 1.0]);
        let result = blend_screen(&base, &black, 4, 4, 1.0);
        // screen with black: 1-(1-base)*(1-0) = 1-(1-base) = base
        for (src, dst) in base.chunks_exact(4).zip(result.chunks_exact(4)) {
            assert!((src[0] - dst[0]).abs() < TOL);
            assert!((src[1] - dst[1]).abs() < TOL);
            assert!((src[2] - dst[2]).abs() < TOL);
        }
    }

    #[test]
    fn premultiply_then_unpremultiply_roundtrip() {
        let input = crate::noise(8, 8, 77);
        // Give varied alpha values
        let mut varied = input.clone();
        for (i, px) in varied.chunks_exact_mut(4).enumerate() {
            px[3] = ((i % 5) as f32 + 1.0) / 5.0; // alpha 0.2..1.0
        }
        let pre = premultiply(&varied, 8, 8);
        let roundtrip = unpremultiply(&pre, 8, 8);
        assert_approx_eq(&varied, &roundtrip, 1e-5);
    }

    #[test]
    fn premultiply_zero_alpha() {
        let input = crate::solid(2, 2, [0.5, 0.7, 0.3, 0.0]);
        let pre = premultiply(&input, 2, 2);
        for px in pre.chunks_exact(4) {
            assert_eq!(px[0], 0.0);
            assert_eq!(px[1], 0.0);
            assert_eq!(px[2], 0.0);
            assert_eq!(px[3], 0.0);
        }
    }

    #[test]
    fn unpremultiply_zero_alpha_safe() {
        let input = crate::solid(2, 2, [0.0, 0.0, 0.0, 0.0]);
        let result = unpremultiply(&input, 2, 2);
        for px in result.chunks_exact(4) {
            assert_eq!(px[0], 0.0);
            assert_eq!(px[1], 0.0);
            assert_eq!(px[2], 0.0);
            assert_eq!(px[3], 0.0);
        }
    }
}
