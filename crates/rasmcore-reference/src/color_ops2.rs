//! Color operations reference implementations — Lab, ACES, dithering, quantization.
//!
//! All operate in **linear f32** space. Alpha preserved unchanged.

// ─── CIE D65 Whitepoint ────────────────────────────────────────────────────

const D65_X: f32 = 0.95047;
const D65_Y: f32 = 1.00000;
const D65_Z: f32 = 1.08883;

// ─── Bayer 8x8 Matrix ──────────────────────────────────────────────────────

/// Bayer 8x8 ordered dithering threshold matrix, values in [0, 63].
#[rustfmt::skip]
pub const BAYER_8X8: [[u8; 8]; 8] = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

// ─── sRGB ↔ XYZ (D65) ─────────────────────────────────────────────────────

/// Linear sRGB to CIE XYZ (D65) — standard 3x3 matrix.
fn rgb_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    (x, y, z)
}

/// CIE XYZ (D65) to linear sRGB — inverse of rgb_to_xyz.
fn xyz_to_rgb(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (r, g, b)
}

// ─── XYZ ↔ Lab ─────────────────────────────────────────────────────────────

fn lab_f(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA_SQ: f32 = DELTA * DELTA;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA_SQ) + 4.0 / 29.0
    }
}

fn lab_f_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA_SQ: f32 = DELTA * DELTA;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA_SQ * (t - 4.0 / 29.0)
    }
}

fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);
    (l, a, b_val)
}

fn lab_to_rgb(l: f32, a: f32, b_val: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_val / 200.0;
    let x = D65_X * lab_f_inv(fx);
    let y = D65_Y * lab_f_inv(fy);
    let z = D65_Z * lab_f_inv(fz);
    xyz_to_rgb(x, y, z)
}

// ─── 1. Lab Adjust ─────────────────────────────────────────────────────────

/// Convert RGB to CIE Lab (D65), shift a and b channels by offsets, convert back.
///
/// Useful for color-balance style adjustments in perceptual space.
/// a_offset shifts green↔red axis; b_offset shifts blue↔yellow axis.
pub fn lab_adjust(input: &[f32], _w: u32, _h: u32, a_offset: f32, b_offset: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (l, a, b) = rgb_to_lab(px[0], px[1], px[2]);
        let (r, g, bl) = lab_to_rgb(l, a + a_offset, b + b_offset);
        px[0] = r;
        px[1] = g;
        px[2] = bl;
    }
    out
}

// ─── 2. Lab Sharpen ────────────────────────────────────────────────────────

/// Unsharp-mask sharpening in CIE Lab L channel.
///
/// Convert to Lab, gaussian blur L, compute L_sharp = L + amount*(L - L_blur),
/// convert back preserving a,b. Only luminance is sharpened.
pub fn lab_sharpen(input: &[f32], w: u32, h: u32, amount: f32, radius: f32) -> Vec<f32> {
    let n = (w * h) as usize;
    let w = w as usize;
    let h = h as usize;

    // Convert all pixels to Lab
    let mut l_chan = Vec::with_capacity(n);
    let mut a_chan = Vec::with_capacity(n);
    let mut b_chan = Vec::with_capacity(n);
    for px in input.chunks_exact(4) {
        let (l, a, b) = rgb_to_lab(px[0], px[1], px[2]);
        l_chan.push(l);
        a_chan.push(a);
        b_chan.push(b);
    }

    // Simple 2D gaussian blur on L channel
    let l_blur = gaussian_blur_2d(&l_chan, w, h, radius);

    // Unsharp mask: L_sharp = L + amount * (L - L_blur)
    let mut out = input.to_vec();
    for i in 0..n {
        let l_sharp = l_chan[i] + amount * (l_chan[i] - l_blur[i]);
        let (r, g, b) = lab_to_rgb(l_sharp, a_chan[i], b_chan[i]);
        let base = i * 4;
        out[base] = r;
        out[base + 1] = g;
        out[base + 2] = b;
        // alpha unchanged
    }
    out
}

/// Simple separable gaussian blur for a single-channel buffer.
fn gaussian_blur_2d(data: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let sigma = radius.max(0.5);
    let kernel_radius = (sigma * 3.0).ceil() as i32;
    let kernel_size = (2 * kernel_radius + 1) as usize;

    // Build 1D gaussian kernel
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0f32;
    for i in 0..kernel_size as i32 {
        let x = (i - kernel_radius) as f32;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..kernel_size as i32 {
                let sx = (x as i32 + k - kernel_radius).clamp(0, w as i32 - 1) as usize;
                acc += data[y * w + sx] * kernel[k as usize];
            }
            temp[y * w + x] = acc;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..kernel_size as i32 {
                let sy = (y as i32 + k - kernel_radius).clamp(0, h as i32 - 1) as usize;
                acc += temp[sy * w + x] * kernel[k as usize];
            }
            result[y * w + x] = acc;
        }
    }

    result
}

// ─── 3. ACEScct → Linear (ACEScg) ─────────────────────────────────────────

/// ACEScct to linear (ACEScg) transfer function per channel.
///
/// if v <= 0.155251 then (v - 0.0729) / 10.5402
/// else 2^(v * 17.52 - 9.72)
pub fn aces_cct_to_cg(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in &mut px[..3] {
            let v = *c;
            *c = if v <= 0.155251 {
                (v - 0.0729) / 10.5402
            } else {
                f32::exp2(v * 17.52 - 9.72)
            };
        }
    }
    out
}

// ─── 4. Linear (ACEScg) → ACEScct ─────────────────────────────────────────

/// Linear (ACEScg) to ACEScct transfer function per channel.
///
/// if v <= 0.0078125 then 10.5402 * v + 0.0729
/// else (log2(v) + 9.72) / 17.52
pub fn aces_cg_to_cct(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in &mut px[..3] {
            let v = *c;
            *c = if v <= 0.0078125 {
                10.5402 * v + 0.0729
            } else {
                (v.log2() + 9.72) / 17.52
            };
        }
    }
    out
}

// ─── 5. Ordered (Bayer) Dithering ──────────────────────────────────────────

/// Bayer 8x8 ordered dithering. Quantize to `levels` per channel with threshold bias.
///
/// For each pixel: threshold = (bayer[y%8][x%8] / 64.0 - 0.5) / levels
/// quantized = round((v + threshold) * (levels-1)) / (levels-1), clamped to [0,1].
pub fn dither_ordered(input: &[f32], w: u32, h: u32, levels: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    let lvl = levels as f32;
    let lvl_m1 = (levels - 1) as f32;
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            let threshold = (BAYER_8X8[(y % 8) as usize][(x % 8) as usize] as f32 / 64.0 - 0.5) / lvl;
            for c in 0..3 {
                let v = out[idx + c];
                let q = ((v + threshold) * lvl_m1).round() / lvl_m1;
                out[idx + c] = q.clamp(0.0, 1.0);
            }
        }
    }
    out
}

// ─── 6. Floyd-Steinberg Dithering ──────────────────────────────────────────

/// Floyd-Steinberg error diffusion dithering. Quantize to `levels` per channel.
///
/// Error distribution: right 7/16, bottom-left 3/16, bottom 5/16, bottom-right 1/16.
pub fn dither_floyd_steinberg(input: &[f32], w: u32, h: u32, levels: u32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let lvl_m1 = (levels - 1) as f32;

    // Work on separate R, G, B channels as f32
    let n = w * h;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in input.chunks_exact(4) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }

    // Process each channel with error diffusion
    for chan in [&mut r, &mut g, &mut b] {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let old = chan[i];
                let new_val = (old * lvl_m1).round() / lvl_m1;
                chan[i] = new_val.clamp(0.0, 1.0);
                let err = old - new_val;

                if x + 1 < w {
                    chan[i + 1] += err * 7.0 / 16.0;
                }
                if y + 1 < h {
                    if x > 0 {
                        chan[(y + 1) * w + (x - 1)] += err * 3.0 / 16.0;
                    }
                    chan[(y + 1) * w + x] += err * 5.0 / 16.0;
                    if x + 1 < w {
                        chan[(y + 1) * w + (x + 1)] += err * 1.0 / 16.0;
                    }
                }
            }
        }
    }

    // Reassemble
    let mut out = input.to_vec();
    for i in 0..n {
        out[i * 4] = r[i].clamp(0.0, 1.0);
        out[i * 4 + 1] = g[i].clamp(0.0, 1.0);
        out[i * 4 + 2] = b[i].clamp(0.0, 1.0);
    }
    out
}

// ─── 7. Uniform Quantization ───────────────────────────────────────────────

/// Simple uniform quantization: round(v * (levels-1)) / (levels-1).
pub fn quantize(input: &[f32], _w: u32, _h: u32, levels: u32) -> Vec<f32> {
    let lvl_m1 = (levels - 1) as f32;
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        for c in &mut px[..3] {
            *c = (*c * lvl_m1).round() / lvl_m1;
        }
    }
    out
}

// ─── 8. K-Means Color Quantization ────────────────────────────────────────

/// K-means color quantization. Seeded random centroid init, iterate until
/// convergence or max_iter. Replace each pixel with nearest centroid RGB.
pub fn kmeans_quantize(
    input: &[f32],
    _w: u32,
    _h: u32,
    k: u32,
    max_iter: u32,
    seed: u64,
) -> Vec<f32> {
    let n = input.len() / 4;
    let k = k as usize;

    // Seeded PRNG (simple xorshift64)
    let mut rng_state = seed;
    let mut next_rng = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // Init centroids from random pixel selection
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    for _ in 0..k {
        let idx = (next_rng() % n as u64) as usize;
        let base = idx * 4;
        centroids.push([input[base], input[base + 1], input[base + 2]]);
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign each pixel to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let base = i * 4;
            let r = input[base];
            let g = input[base + 1];
            let b = input[base + 2];

            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let dr = r - c[0];
                let dg = g - c[1];
                let db = b - c[2];
                let dist = dr * dr + dg * dg + db * db;
                if dist < best_dist {
                    best_dist = dist;
                    best = ci;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u64; k];
        for i in 0..n {
            let base = i * 4;
            let ci = assignments[i];
            sums[ci][0] += input[base] as f64;
            sums[ci][1] += input[base + 1] as f64;
            sums[ci][2] += input[base + 2] as f64;
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                centroids[ci][0] = (sums[ci][0] / counts[ci] as f64) as f32;
                centroids[ci][1] = (sums[ci][1] / counts[ci] as f64) as f32;
                centroids[ci][2] = (sums[ci][2] / counts[ci] as f64) as f32;
            }
        }
    }

    // Replace pixels with centroids
    let mut out = input.to_vec();
    for i in 0..n {
        let base = i * 4;
        let c = &centroids[assignments[i]];
        out[base] = c[0];
        out[base + 1] = c[1];
        out[base + 2] = c[2];
        // alpha unchanged
    }
    out
}

// ─── ACES Matrices ─────────────────────────────────────────────────────────

/// sRGB linear → ACEScg 3x3 matrix.
const SRGB_TO_ACESCG: [[f32; 3]; 3] = [
    [0.6131, 0.3395, 0.0474],
    [0.0702, 0.9164, 0.0134],
    [0.0206, 0.1096, 0.8698],
];

/// ACEScg → sRGB linear 3x3 matrix (inverse).
const ACESCG_TO_SRGB: [[f32; 3]; 3] = [
    [1.7051, -0.6218, -0.0833],
    [-0.1302, 1.1408, -0.0106],
    [-0.0240, -0.1290, 1.1530],
];

fn mat3_mul(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

/// Linear → ACEScct transfer (per channel).
fn linear_to_acescct(v: f32) -> f32 {
    if v <= 0.0078125 {
        10.5402 * v + 0.0729
    } else {
        (v.log2() + 9.72) / 17.52
    }
}

/// ACEScct → linear transfer (per channel).
fn acescct_to_linear(v: f32) -> f32 {
    if v <= 0.155251 {
        (v - 0.0729) / 10.5402
    } else {
        f32::exp2(v * 17.52 - 9.72)
    }
}

// ─── 9. ACES IDT (sRGB linear → ACEScg → ACEScct) ─────────────────────────

/// ACES Input Device Transform: sRGB linear → ACEScg matrix → ACEScct log.
pub fn aces_idt(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let (r, g, b) = mat3_mul(&SRGB_TO_ACESCG, px[0], px[1], px[2]);
        px[0] = linear_to_acescct(r);
        px[1] = linear_to_acescct(g);
        px[2] = linear_to_acescct(b);
    }
    out
}

// ─── 10. ACES ODT (ACEScct → ACEScg → sRGB linear) ────────────────────────

/// ACES Output Device Transform: ACEScct log → ACEScg linear → sRGB matrix.
pub fn aces_odt(input: &[f32], _w: u32, _h: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for px in out.chunks_exact_mut(4) {
        let r = acescct_to_linear(px[0]);
        let g = acescct_to_linear(px[1]);
        let b = acescct_to_linear(px[2]);
        let (sr, sg, sb) = mat3_mul(&ACESCG_TO_SRGB, r, g, b);
        px[0] = sr;
        px[1] = sg;
        px[2] = sb;
    }
    out
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_pixels() -> Vec<f32> {
        // A small set of diverse linear RGB pixels with alpha
        vec![
            0.2, 0.4, 0.6, 1.0,
            0.8, 0.1, 0.3, 1.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.01, 0.5, 0.9, 1.0,
        ]
    }

    fn max_channel_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn aces_cct_roundtrip() {
        // ACEScct → linear → ACEScct should be identity
        let input = vec![
            0.2, 0.4, 0.6, 1.0,
            0.8, 0.1, 0.5, 1.0,
            0.01, 0.15, 0.9, 1.0,
        ];
        let linear = aces_cct_to_cg(&input, 3, 1);
        let back = aces_cg_to_cct(&linear, 3, 1);
        let diff = max_channel_diff(&input, &back);
        assert!(
            diff < 1e-4,
            "ACEScct roundtrip max diff {diff:.8} exceeds tolerance"
        );
    }

    #[test]
    fn aces_idt_odt_roundtrip() {
        // sRGB linear → IDT → ODT → sRGB linear should be identity
        let input = make_test_pixels();
        let w = 3u32;
        let h = 2u32;
        let encoded = aces_idt(&input, w, h);
        let decoded = aces_odt(&encoded, w, h);
        let diff = max_channel_diff(&input, &decoded);
        assert!(
            diff < 1e-3,
            "ACES IDT/ODT roundtrip max diff {diff:.8} exceeds tolerance"
        );
    }

    #[test]
    fn quantize_256_identity() {
        let input = make_test_pixels();
        let result = quantize(&input, 3, 2, 256);
        let diff = max_channel_diff(&input, &result);
        // Max quantization error for 256 levels is 0.5/255 ≈ 0.00196
        assert!(
            diff < 0.002,
            "quantize(256) should be near-identity, max diff {diff:.8}"
        );
    }

    #[test]
    fn dither_ordered_preserves_mean() {
        // Generate a larger test image for statistical stability
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let mut input = Vec::with_capacity(n * 4);
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            input.extend_from_slice(&[t, t * 0.7, 0.3 + t * 0.4, 1.0]);
        }

        let result = dither_ordered(&input, w, h, 8);

        // Compare mean of R channel
        let mean_in: f32 = (0..n).map(|i| input[i * 4]).sum::<f32>() / n as f32;
        let mean_out: f32 = (0..n).map(|i| result[i * 4]).sum::<f32>() / n as f32;
        let mean_diff = (mean_in - mean_out).abs();
        assert!(
            mean_diff < 0.05,
            "dither_ordered mean drift {mean_diff:.6} exceeds tolerance"
        );
    }

    #[test]
    fn alpha_preserved_lab_adjust() {
        let input = vec![0.5, 0.5, 0.5, 0.42, 0.2, 0.8, 0.4, 0.99];
        let result = lab_adjust(&input, 2, 1, 10.0, -5.0);
        assert_eq!(result[3], 0.42);
        assert_eq!(result[7], 0.99);
    }

    #[test]
    fn alpha_preserved_kmeans() {
        let input = vec![0.1, 0.2, 0.3, 0.77, 0.9, 0.8, 0.7, 0.33];
        let result = kmeans_quantize(&input, 2, 1, 2, 10, 42);
        assert_eq!(result[3], 0.77);
        assert_eq!(result[7], 0.33);
    }
}
