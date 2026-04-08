//! Shared filter helpers — deduplicated functions used across multiple filter modules.
//!
//! Eliminates 30+ duplicate definitions of luminance, HSL, GPU params, and bilinear sampling.

// ─── Luminance ──────────────────────────────────────────────────────────────

/// BT.709 luminance from f32 RGB.
#[inline]
pub fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ─── HSL Conversion ─────────────────────────────────────────────────────────

/// RGB [0,1] → HSL (H in [0,360], S in [0,1], L in [0,1]).
pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-7 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if (max - r).abs() < 1e-7 {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h * 60.0
    } else if (max - g).abs() < 1e-7 {
        ((b - r) / d + 2.0) * 60.0
    } else {
        ((r - g) / d + 4.0) * 60.0
    };
    (h, s, l)
}

/// HSL → RGB [0,1].
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-7 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;
    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

// ─── GPU Parameter Serialization ────────────────────────────────────────────

/// Create GPU uniform buffer with width and height (8 bytes, little-endian).
#[inline]
pub fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}

/// Push f32 to GPU uniform buffer (little-endian).
#[inline]
pub fn gpu_push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Push u32 to GPU uniform buffer (little-endian).
#[inline]
pub fn gpu_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ─── Bilinear Sampling ──────────────────────────────────────────────────────

/// Bilinear sample f32 RGBA pixel at fractional coordinates with edge clamping.
#[inline(always)]
pub fn sample_bilinear(input: &[f32], width: u32, height: u32, fx: f32, fy: f32) -> [f32; 4] {
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let dx = fx - ix as f32;
    let dy = fy - iy as f32;

    let x0 = ix.max(0).min(width as i32 - 1) as usize;
    let x1 = (ix + 1).max(0).min(width as i32 - 1) as usize;
    let y0 = iy.max(0).min(height as i32 - 1) as usize;
    let y1 = (iy + 1).max(0).min(height as i32 - 1) as usize;

    let w = width as usize;
    let i00 = (y0 * w + x0) * 4;
    let i10 = (y0 * w + x1) * 4;
    let i01 = (y1 * w + x0) * 4;
    let i11 = (y1 * w + x1) * 4;

    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let p00 = input[i00 + c];
        let p10 = input[i10 + c];
        let p01 = input[i01 + c];
        let p11 = input[i11 + c];
        let top = p00 + dx * (p10 - p00);
        let bot = p01 + dx * (p11 - p01);
        out[c] = top + dy * (bot - top);
    }
    out
}

// ─── Edge Detection Helpers ─────────────────────────────────────────────────

/// Sample luminance at pixel (x, y) with edge clamping.
#[inline]
pub fn sample_luma(input: &[f32], w: usize, h: usize, x: i32, y: i32) -> f32 {
    let cx = x.max(0).min(w as i32 - 1) as usize;
    let cy = y.max(0).min(h as i32 - 1) as usize;
    let idx = (cy * w + cx) * 4;
    luminance(input[idx], input[idx + 1], input[idx + 2])
}

/// 3x3 convolution at (cx, cy) on luminance values with edge clamping.
#[inline]
pub fn convolve3x3(input: &[f32], w: usize, h: usize, cx: i32, cy: i32, kernel: &[f32; 9]) -> f32 {
    let mut sum = 0.0f32;
    for ky in -1..=1 {
        for kx in -1..=1 {
            let idx = ((ky + 1) * 3 + (kx + 1)) as usize;
            sum += kernel[idx] * sample_luma(input, w, h, cx + kx, cy + ky);
        }
    }
    sum
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_white_is_one() {
        assert!((luminance(1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn luminance_black_is_zero() {
        assert!(luminance(0.0, 0.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn luminance_green_dominant() {
        let l = luminance(0.5, 0.5, 0.5);
        assert!((l - 0.5).abs() < 1e-5);
        // Green coefficient is largest
        assert!(luminance(0.0, 1.0, 0.0) > luminance(1.0, 0.0, 0.0));
        assert!(luminance(0.0, 1.0, 0.0) > luminance(0.0, 0.0, 1.0));
    }

    #[test]
    fn hsl_roundtrip() {
        for (r, g, b) in [(0.3, 0.5, 0.7), (1.0, 0.0, 0.0), (0.0, 1.0, 0.5), (0.5, 0.5, 0.5)] {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let (r2, g2, b2) = hsl_to_rgb(h, s, l);
            assert!((r - r2).abs() < 1e-5, "R mismatch for ({r},{g},{b}): {r2}");
            assert!((g - g2).abs() < 1e-5, "G mismatch for ({r},{g},{b}): {g2}");
            assert!((b - b2).abs() < 1e-5, "B mismatch for ({r},{g},{b}): {b2}");
        }
    }

    #[test]
    fn gpu_params_wh_layout() {
        let buf = gpu_params_wh(100, 200);
        assert_eq!(buf.len(), 8);
        assert_eq!(u32::from_le_bytes(buf[0..4].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 200);
    }

    #[test]
    fn gpu_push_values() {
        let mut buf = Vec::new();
        gpu_push_f32(&mut buf, 1.5);
        gpu_push_u32(&mut buf, 42);
        assert_eq!(buf.len(), 8);
        assert_eq!(f32::from_le_bytes(buf[0..4].try_into().unwrap()), 1.5);
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 42);
    }

    #[test]
    fn bilinear_at_pixel_center() {
        // 2x2 image: each pixel has (x*0.5, y*0.5, 0, 1)
        let pixels = vec![
            0.0, 0.0, 0.0, 1.0,  0.5, 0.0, 0.0, 1.0,
            0.0, 0.5, 0.0, 1.0,  0.5, 0.5, 0.0, 1.0,
        ];
        let p = sample_bilinear(&pixels, 2, 2, 0.0, 0.0);
        assert!((p[0] - 0.0).abs() < 1e-6);
        let p = sample_bilinear(&pixels, 2, 2, 1.0, 1.0);
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
    }
}
