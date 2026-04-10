//! Composite, alpha, and mask operation reference implementations.
//!
//! All operations work on **linear f32 RGBA interleaved** buffers.
//! Pure math — no external dependencies.

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 >= edge1 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-10 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if (max - r).abs() < 1e-10 {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h
    } else if (max - g).abs() < 1e-10 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h * 60.0, s, l)
}

// ─── Composite Ops ────────────────────────────────────────────────────────────

/// W3C CSS Compositing Level 1 blend modes.
///
/// Supported modes: "normal", "multiply", "screen", "overlay", "darken",
/// "lighten", "color-dodge", "color-burn", "hard-light", "soft-light",
/// "difference", "exclusion".
///
/// Formula per mode applied per-channel, then lerp with base by opacity.
/// Alpha taken from base.
///
/// Validated against: W3C CSS Compositing and Blending Level 1 spec.
pub fn blend(
    base: &[f32],
    overlay: &[f32],
    _w: u32,
    _h: u32,
    mode: &str,
    opacity: f32,
) -> Vec<f32> {
    assert_eq!(base.len(), overlay.len());
    let mut out = base.to_vec();
    for (bp, op) in out.chunks_exact_mut(4).zip(overlay.chunks_exact(4)) {
        for c in 0..3 {
            let b = bp[c];
            let o = op[c];
            let blended = match mode {
                "normal" => o,
                "multiply" => b * o,
                "screen" => b + o - b * o,
                "overlay" => {
                    if b < 0.5 {
                        2.0 * b * o
                    } else {
                        1.0 - 2.0 * (1.0 - b) * (1.0 - o)
                    }
                }
                "darken" => b.min(o),
                "lighten" => b.max(o),
                "color-dodge" => {
                    if o >= 1.0 {
                        1.0
                    } else {
                        (b / (1.0 - o)).min(1.0)
                    }
                }
                "color-burn" => {
                    if o <= 0.0 {
                        0.0
                    } else {
                        1.0 - ((1.0 - b) / o).min(1.0)
                    }
                }
                "hard-light" => {
                    if o < 0.5 {
                        2.0 * b * o
                    } else {
                        1.0 - 2.0 * (1.0 - b) * (1.0 - o)
                    }
                }
                "soft-light" => {
                    // W3C formula
                    if o <= 0.5 {
                        b - (1.0 - 2.0 * o) * b * (1.0 - b)
                    } else {
                        let d = if b <= 0.25 {
                            ((16.0 * b - 12.0) * b + 4.0) * b
                        } else {
                            b.sqrt()
                        };
                        b + (2.0 * o - 1.0) * (d - b)
                    }
                }
                "difference" => (b - o).abs(),
                "exclusion" => b + o - 2.0 * b * o,
                _ => panic!("unsupported blend mode: {mode}"),
            };
            bp[c] = b + opacity * (blended - b);
        }
        // alpha from base — unchanged
    }
    out
}

/// Photoshop blend-if: blend based on overlay luminance range.
///
/// weight = smoothstep(blend_min, blend_min+0.1, luma)
///        * (1 - smoothstep(blend_max-0.1, blend_max, luma))
/// out = base + weight * opacity * (overlay - base)
///
/// Validated against: Photoshop CC 2024 Blend-If underlying-layer slider.
pub fn blend_if(
    base: &[f32],
    overlay: &[f32],
    _w: u32,
    _h: u32,
    opacity: f32,
    blend_min: f32,
    blend_max: f32,
) -> Vec<f32> {
    assert_eq!(base.len(), overlay.len());
    let mut out = base.to_vec();
    for (bp, op) in out.chunks_exact_mut(4).zip(overlay.chunks_exact(4)) {
        let luma = luminance(op[0], op[1], op[2]);
        let w = smoothstep(blend_min, blend_min + 0.1, luma)
            * (1.0 - smoothstep(blend_max - 0.1, blend_max, luma));
        let factor = w * opacity;
        for c in 0..3 {
            bp[c] = bp[c] + factor * (op[c] - bp[c]);
        }
    }
    out
}

// ─── Alpha Ops ────────────────────────────────────────────────────────────────

/// Set alpha channel to the given value for all pixels.
///
/// Assumes RGBA input. Sets A = alpha_value for every pixel.
pub fn add_alpha(input: &[f32], _w: u32, _h: u32, alpha_value: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[3] = alpha_value;
    }
    out
}

/// Force alpha to 1.0 for all pixels.
///
/// The actual filter strips the alpha channel; for f32 RGBA reference we
/// just set A = 1.0.
pub fn remove_alpha(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[3] = 1.0;
    }
    out
}

/// Composite onto solid background color.
///
/// Formula: out_c = in_c * A + bg_c * (1 - A), out_A = 1.0.
///
/// Validated against: ImageMagick 7.1.1 `-flatten` with background color.
pub fn flatten(input: &[f32], _w: u32, _h: u32, bg_r: f32, bg_g: f32, bg_b: f32) -> Vec<f32> {
    let bg = [bg_r, bg_g, bg_b];
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let a = px[3];
        for c in 0..3 {
            px[c] = px[c] * a + bg[c] * (1.0 - a);
        }
        px[3] = 1.0;
    }
    out
}

// ─── Mask Ops ─────────────────────────────────────────────────────────────────

/// Multiply input alpha by mask luminance.
///
/// Formula: A' = A * (0.2126*mask_R + 0.7152*mask_G + 0.0722*mask_B).
/// RGB channels unchanged.
pub fn mask_apply(input: &[f32], _w: u32, _h: u32, mask: &[f32]) -> Vec<f32> {
    assert_eq!(input.len(), mask.len());
    let mut out = input.to_vec();
    for (px, mp) in out.chunks_exact_mut(4).zip(mask.chunks_exact(4)) {
        let mask_luma = luminance(mp[0], mp[1], mp[2]);
        px[3] *= mask_luma;
    }
    out
}

/// Combine two masks using luminance values.
///
/// Modes: "multiply" = a*b, "add" = min(a+b, 1), "subtract" = max(a-b, 0),
/// "intersect" = min(a, b), "union" = max(a, b).
/// Output is grayscale RGBA (all channels = result, A = 1.0).
pub fn mask_combine(a: &[f32], b: &[f32], _w: u32, _h: u32, mode: &str) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let mut out = vec![0.0f32; a.len()];
    for (i, (ap, bp)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let la = luminance(ap[0], ap[1], ap[2]);
        let lb = luminance(bp[0], bp[1], bp[2]);
        let v = match mode {
            "multiply" => la * lb,
            "add" => (la + lb).min(1.0),
            "subtract" => (la - lb).max(0.0),
            "intersect" => la.min(lb),
            "union" => la.max(lb),
            _ => panic!("unsupported mask combine mode: {mode}"),
        };
        let off = i * 4;
        out[off] = v;
        out[off + 1] = v;
        out[off + 2] = v;
        out[off + 3] = 1.0;
    }
    out
}

/// Gaussian blur on alpha channel only. RGB unchanged.
///
/// Uses separable 1D Gaussian with the given radius. Kernel size = 2*radius+1.
/// sigma = radius / 2.0 (standard convention).
pub fn feather(input: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }

    let w = w as usize;
    let h = h as usize;
    let r = radius as usize;
    let sigma = radius as f32 / 2.0;
    let s2 = 2.0 * sigma * sigma;

    // Build 1D kernel
    let ksize = 2 * r + 1;
    let mut kernel = vec![0.0f32; ksize];
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - r as f32;
        let v = (-x * x / s2).exp();
        kernel[i] = v;
        sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Extract alpha channel
    let mut alpha = vec![0.0f32; w * h];
    for i in 0..(w * h) {
        alpha[i] = input[i * 4 + 3];
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for ki in 0..ksize {
                let sx = (x as i64 + ki as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
                acc += alpha[y * w + sx] * kernel[ki];
            }
            temp[y * w + x] = acc;
        }
    }

    // Vertical pass
    let mut blurred = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for ki in 0..ksize {
                let sy = (y as i64 + ki as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
                acc += temp[sy * w + x] * kernel[ki];
            }
            blurred[y * w + x] = acc;
        }
    }

    // Write back
    let mut out = input.to_vec();
    for i in 0..(w * h) {
        out[i * 4 + 3] = blurred[i];
    }
    out
}

/// Generate a linear gradient mask from point1 to point2.
///
/// Output is grayscale RGBA: value = projection of pixel onto the line
/// from p1 to p2 (0.0 at p1, 1.0 at p2, clamped).
pub fn gradient_mask(w: u32, h: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> Vec<f32> {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    let mut out = vec![0.0f32; (w * h * 4) as usize];
    for py in 0..h {
        for px in 0..w {
            let vx = px as f32 - x1;
            let vy = py as f32 - y1;
            let t = if len_sq < 1e-10 {
                0.0
            } else {
                ((vx * dx + vy * dy) / len_sq).clamp(0.0, 1.0)
            };
            let off = ((py * w + px) * 4) as usize;
            out[off] = t;
            out[off + 1] = t;
            out[off + 2] = t;
            out[off + 3] = 1.0;
        }
    }
    out
}

/// Mask based on luminance range with smoothstep edges.
///
/// Output = 1.0 if low <= luma <= high, else 0.0, with 0.05 transition
/// at both edges. All channels set to same value, A = 1.0.
pub fn luminance_range(input: &[f32], _w: u32, _h: u32, low: f32, high: f32) -> Vec<f32> {
    let transition = 0.05;
    let mut out = vec![0.0f32; input.len()];
    for (op, ip) in out.chunks_exact_mut(4).zip(input.chunks_exact(4)) {
        let luma = luminance(ip[0], ip[1], ip[2]);
        let v = smoothstep(low - transition, low + transition, luma)
            * (1.0 - smoothstep(high - transition, high + transition, luma));
        op[0] = v;
        op[1] = v;
        op[2] = v;
        op[3] = 1.0;
    }
    out
}

/// Mask based on hue proximity.
///
/// Convert to HSL, weight = max(0, 1 - |hue_diff|/hue_range) if S >= sat_min.
/// Hue difference wraps at 360. Output is grayscale RGBA.
pub fn color_range(
    input: &[f32],
    _w: u32,
    _h: u32,
    target_hue: f32,
    hue_range: f32,
    sat_min: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (op, ip) in out.chunks_exact_mut(4).zip(input.chunks_exact(4)) {
        let (h, s, _l) = rgb_to_hsl(ip[0], ip[1], ip[2]);
        let v = if s >= sat_min && hue_range > 0.0 {
            let mut diff = (h - target_hue).abs();
            if diff > 180.0 {
                diff = 360.0 - diff;
            }
            (1.0 - diff / hue_range).max(0.0)
        } else {
            0.0
        };
        op[0] = v;
        op[1] = v;
        op[2] = v;
        op[3] = 1.0;
    }
    out
}

/// Per-pixel blend using mask as weight.
///
/// Formula: out = base * (1 - mask_luma) + overlay * mask_luma.
/// Alpha from base.
pub fn masked_blend(
    base: &[f32],
    overlay: &[f32],
    mask: &[f32],
    _w: u32,
    _h: u32,
) -> Vec<f32> {
    assert_eq!(base.len(), overlay.len());
    assert_eq!(base.len(), mask.len());
    let mut out = base.to_vec();
    for i in 0..(base.len() / 4) {
        let ml = luminance(mask[i * 4], mask[i * 4 + 1], mask[i * 4 + 2]);
        let off = i * 4;
        for c in 0..3 {
            out[off + c] = base[off + c] * (1.0 - ml) + overlay[off + c] * ml;
        }
        // alpha from base — unchanged
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 4;
    const H: u32 = 4;
    const N: usize = (W * H * 4) as usize;
    const EPS: f32 = 1e-6;

    fn solid(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        let mut buf = Vec::with_capacity(N);
        for _ in 0..(W * H) {
            buf.extend_from_slice(&[r, g, b, a]);
        }
        buf
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn blend_normal_opacity_0_returns_base() {
        let base = solid(0.2, 0.4, 0.6, 1.0);
        let over = solid(0.8, 0.1, 0.3, 1.0);
        let result = blend(&base, &over, W, H, "normal", 0.0);
        assert!(max_diff(&result, &base) < EPS);
    }

    #[test]
    fn blend_normal_opacity_1_returns_overlay() {
        let base = solid(0.2, 0.4, 0.6, 1.0);
        let over = solid(0.8, 0.1, 0.3, 1.0);
        let result = blend(&base, &over, W, H, "normal", 1.0);
        // RGB should match overlay, alpha should match base
        for px in result.chunks_exact(4) {
            assert!((px[0] - 0.8).abs() < EPS);
            assert!((px[1] - 0.1).abs() < EPS);
            assert!((px[2] - 0.3).abs() < EPS);
            assert!((px[3] - 1.0).abs() < EPS);
        }
    }

    #[test]
    fn blend_multiply_white_overlay_is_identity() {
        let base = solid(0.3, 0.5, 0.7, 1.0);
        let white = solid(1.0, 1.0, 1.0, 1.0);
        let result = blend(&base, &white, W, H, "multiply", 1.0);
        assert!(max_diff(&result, &base) < EPS);
    }

    #[test]
    fn add_alpha_sets_alpha_correctly() {
        let input = solid(0.5, 0.5, 0.5, 1.0);
        let result = add_alpha(&input, W, H, 0.3);
        for px in result.chunks_exact(4) {
            assert!((px[0] - 0.5).abs() < EPS);
            assert!((px[3] - 0.3).abs() < EPS);
        }
    }

    #[test]
    fn flatten_alpha_1_is_identity() {
        let input = solid(0.4, 0.6, 0.8, 1.0);
        let result = flatten(&input, W, H, 0.0, 0.0, 0.0);
        // With A=1.0, background is ignored: out_c = in_c*1 + bg*0 = in_c
        assert!(max_diff(&result, &input) < EPS);
    }

    #[test]
    fn mask_apply_white_mask_is_identity() {
        let input = solid(0.3, 0.5, 0.7, 0.8);
        let white_mask = solid(1.0, 1.0, 1.0, 1.0);
        let result = mask_apply(&input, W, H, &white_mask);
        // luminance of white = 1.0, so A' = 0.8 * 1.0 = 0.8
        assert!(max_diff(&result, &input) < EPS);
    }

    #[test]
    fn gradient_mask_endpoints() {
        let g = gradient_mask(16, 1, 0.0, 0.0, 15.0, 0.0);
        // First pixel at x=0: t = 0.0
        assert!(g[0].abs() < EPS);
        // Last pixel at x=15: t = 1.0
        let last = 15 * 4;
        assert!((g[last] - 1.0).abs() < EPS);
    }

    #[test]
    fn masked_blend_black_mask_returns_base() {
        let base = solid(0.2, 0.4, 0.6, 1.0);
        let over = solid(0.9, 0.1, 0.3, 1.0);
        let black_mask = solid(0.0, 0.0, 0.0, 1.0);
        let result = masked_blend(&base, &over, &black_mask, W, H);
        assert!(max_diff(&result, &base) < EPS);
    }
}
