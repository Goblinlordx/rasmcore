//! Color space conversions and color science operations.
//!
//! - CIE Lab (via XYZ, D65 illuminant)
//! - OKLab (Ottosson 2020)
//! - White balance (gray-world, manual temperature/tint)
//! - Bradford chromatic adaptation
//! - Perspective warp (4-point homography)

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

// ─── sRGB ↔ Linear ───────────────────────────────────────────────────────

#[inline]
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ─── CIE Lab (via XYZ, D65 illuminant) ───────────────────────────────────

/// D65 reference white point (standard daylight).
const D65_X: f64 = 0.95047;
const D65_Y: f64 = 1.00000;
const D65_Z: f64 = 1.08883;

/// sRGB to XYZ matrix (D65, IEC 61966-2-1).
fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;
    (x, y, z)
}

/// XYZ to sRGB.
fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}

#[inline]
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

#[inline]
fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Convert sRGB [0,1] to CIE Lab.
pub fn rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);
    (l, a, b_val)
}

/// Convert CIE Lab to sRGB [0,1].
pub fn lab_to_rgb(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let x = D65_X * lab_f_inv(fx);
    let y = D65_Y * lab_f_inv(fy);
    let z = D65_Z * lab_f_inv(fz);
    xyz_to_rgb(x, y, z)
}

/// Convert an RGB8 image to Lab (output: 3 f64 channels per pixel, interleaved [L,a,b,...]).
pub fn image_rgb_to_lab(pixels: &[u8], info: &ImageInfo) -> Result<Vec<f64>, ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat("Lab conversion requires Rgb8".into()));
    }
    let n = (info.width * info.height) as usize;
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let r = pixels[i * 3] as f64 / 255.0;
        let g = pixels[i * 3 + 1] as f64 / 255.0;
        let b = pixels[i * 3 + 2] as f64 / 255.0;
        let (l, a, bv) = rgb_to_lab(r, g, b);
        out.push(l);
        out.push(a);
        out.push(bv);
    }
    Ok(out)
}

/// Convert Lab image (f64 interleaved [L,a,b,...]) back to RGB8.
pub fn image_lab_to_rgb(lab: &[f64], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    let n = (info.width * info.height) as usize;
    if lab.len() != n * 3 {
        return Err(ImageError::InvalidParameters("Lab buffer size mismatch".into()));
    }
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let (r, g, b) = lab_to_rgb(lab[i * 3], lab[i * 3 + 1], lab[i * 3 + 2]);
        out.push((r.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((g.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((b.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    Ok(out)
}

// ─── OKLab (Ottosson 2020) ───────────────────────────────────────────────

/// Convert sRGB [0,1] to OKLab.
pub fn rgb_to_oklab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    let l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl;
    let m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl;
    let s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    (ok_l, ok_a, ok_b)
}

/// Convert OKLab to sRGB [0,1].
pub fn oklab_to_rgb(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    let rl = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}

// ─── White Balance ───────────────────────────────────────────────────────

/// Gray-world white balance: adjust so mean R = mean G = mean B.
pub fn white_balance_gray_world(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat("White balance requires Rgb8".into()));
    }
    let n = (info.width * info.height) as usize;
    let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
    for i in 0..n {
        sr += pixels[i * 3] as u64;
        sg += pixels[i * 3 + 1] as u64;
        sb += pixels[i * 3 + 2] as u64;
    }
    let avg_r = sr as f64 / n as f64;
    let avg_g = sg as f64 / n as f64;
    let avg_b = sb as f64 / n as f64;
    let avg_all = (avg_r + avg_g + avg_b) / 3.0;

    let scale_r = if avg_r > 0.0 { avg_all / avg_r } else { 1.0 };
    let scale_g = if avg_g > 0.0 { avg_all / avg_g } else { 1.0 };
    let scale_b = if avg_b > 0.0 { avg_all / avg_b } else { 1.0 };

    let mut out = vec![0u8; pixels.len()];
    for i in 0..n {
        out[i * 3] = (pixels[i * 3] as f64 * scale_r).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = (pixels[i * 3 + 1] as f64 * scale_g).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (pixels[i * 3 + 2] as f64 * scale_b).round().clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

/// Manual white balance via color temperature and tint.
/// `temperature`: Kelvin offset from 6500K (negative = cooler/blue, positive = warmer/yellow)
/// `tint`: green-magenta shift (-1.0 to 1.0)
pub fn white_balance_temperature(
    pixels: &[u8], info: &ImageInfo, temperature: f64, tint: f64,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat("White balance requires Rgb8".into()));
    }
    // Simple temperature model: shift R and B channels
    // Positive temp → warm (boost R, reduce B)
    // Positive tint → magenta (reduce G)
    let temp_norm = temperature / 100.0; // normalize to ~±1 range for typical ±100K shifts
    let scale_r = 1.0 + temp_norm * 0.1;
    let scale_b = 1.0 - temp_norm * 0.1;
    let scale_g = 1.0 - tint * 0.1;

    let n = (info.width * info.height) as usize;
    let mut out = vec![0u8; pixels.len()];
    for i in 0..n {
        out[i * 3] = (pixels[i * 3] as f64 * scale_r).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = (pixels[i * 3 + 1] as f64 * scale_g).round().clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (pixels[i * 3 + 2] as f64 * scale_b).round().clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

// ─── Bradford Chromatic Adaptation ───────────────────────────────────────

/// Standard illuminant white points (CIE XYZ).
#[derive(Debug, Clone, Copy)]
pub enum Illuminant {
    D50, // 0.96422, 1.00000, 0.82521
    D65, // 0.95047, 1.00000, 1.08883
    A,   // 1.09850, 1.00000, 0.35585
}

impl Illuminant {
    pub fn xyz(self) -> (f64, f64, f64) {
        match self {
            Illuminant::D50 => (0.96422, 1.00000, 0.82521),
            Illuminant::D65 => (0.95047, 1.00000, 1.08883),
            Illuminant::A => (1.09850, 1.00000, 0.35585),
        }
    }
}

/// Bradford chromatic adaptation matrix: transform XYZ from one illuminant to another.
pub fn bradford_adapt(x: f64, y: f64, z: f64, from: Illuminant, to: Illuminant) -> (f64, f64, f64) {
    // Bradford cone response matrix
    const M: [[f64; 3]; 3] = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    // Inverse
    const M_INV: [[f64; 3]; 3] = [
        [0.9869929, -0.1470543, 0.1599627],
        [0.4323053, 0.5183603, 0.0492912],
        [-0.0085287, 0.0400428, 0.9684867],
    ];

    let (sx, sy, sz) = from.xyz();
    let (dx, dy, dz) = to.xyz();

    // Source cone response
    let s_rho = M[0][0] * sx + M[0][1] * sy + M[0][2] * sz;
    let s_gamma = M[1][0] * sx + M[1][1] * sy + M[1][2] * sz;
    let s_beta = M[2][0] * sx + M[2][1] * sy + M[2][2] * sz;

    // Destination cone response
    let d_rho = M[0][0] * dx + M[0][1] * dy + M[0][2] * dz;
    let d_gamma = M[1][0] * dx + M[1][1] * dy + M[1][2] * dz;
    let d_beta = M[2][0] * dx + M[2][1] * dy + M[2][2] * dz;

    // Transform input to cone space
    let rho = M[0][0] * x + M[0][1] * y + M[0][2] * z;
    let gamma = M[1][0] * x + M[1][1] * y + M[1][2] * z;
    let beta = M[2][0] * x + M[2][1] * y + M[2][2] * z;

    // Scale
    let rho_a = rho * (d_rho / s_rho);
    let gamma_a = gamma * (d_gamma / s_gamma);
    let beta_a = beta * (d_beta / s_beta);

    // Transform back to XYZ
    let xo = M_INV[0][0] * rho_a + M_INV[0][1] * gamma_a + M_INV[0][2] * beta_a;
    let yo = M_INV[1][0] * rho_a + M_INV[1][1] * gamma_a + M_INV[1][2] * beta_a;
    let zo = M_INV[2][0] * rho_a + M_INV[2][1] * gamma_a + M_INV[2][2] * beta_a;
    (xo, yo, zo)
}

// ─── Perspective Warp (4-Point Homography) ───────────────────────────────

/// Solve 4-point homography and warp the image.
///
/// `src_points`: 4 source corner points [(x,y); 4] in the input image
/// `dst_points`: 4 destination corner points [(x,y); 4] in the output image
/// `out_width`, `out_height`: output dimensions
pub fn perspective_warp(
    pixels: &[u8],
    info: &ImageInfo,
    src_points: &[(f64, f64); 4],
    dst_points: &[(f64, f64); 4],
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>, ImageError> {
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
        other => return Err(ImageError::UnsupportedFormat(format!("perspective warp on {other:?}"))),
    };
    let (w, h) = (info.width as usize, info.height as usize);

    // Solve homography: dst = H * src → we need the inverse H_inv: src = H_inv * dst
    let h_mat = solve_homography(dst_points, src_points)?;

    let mut out = vec![0u8; (out_width * out_height) as usize * channels];

    for dy in 0..out_height as usize {
        for dx in 0..out_width as usize {
            // Apply inverse homography
            let dxf = dx as f64;
            let dyf = dy as f64;
            let denom = h_mat[6] * dxf + h_mat[7] * dyf + h_mat[8];
            if denom.abs() < 1e-10 {
                continue;
            }
            let sx = (h_mat[0] * dxf + h_mat[1] * dyf + h_mat[2]) / denom;
            let sy = (h_mat[3] * dxf + h_mat[4] * dyf + h_mat[5]) / denom;

            // Bilinear sampling
            if sx < 0.0 || sy < 0.0 || sx > (w - 1) as f64 || sy > (h - 1) as f64 {
                continue; // out of bounds → black
            }

            let x0 = (sx.floor() as usize).min(w - 1);
            let y0 = (sy.floor() as usize).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;

            let out_off = (dy * out_width as usize + dx) * channels;
            for c in 0..channels {
                let tl = pixels[(y0 * w + x0) * channels + c] as f64;
                let tr = pixels[(y0 * w + x1) * channels + c] as f64;
                let bl = pixels[(y1 * w + x0) * channels + c] as f64;
                let br = pixels[(y1 * w + x1) * channels + c] as f64;
                let v = tl * (1.0 - fx) * (1.0 - fy)
                    + tr * fx * (1.0 - fy)
                    + bl * (1.0 - fx) * fy
                    + br * fx * fy;
                out[out_off + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(out)
}

/// Solve a 3x3 homography matrix from 4 point correspondences.
/// Uses the DLT (Direct Linear Transform) algorithm.
fn solve_homography(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> Result<[f64; 9], ImageError> {
    // Build 8x9 matrix A where A * h = 0
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (sx, sy) = (src[i].0 as f64, src[i].1 as f64);
        let (dx, dy) = (dst[i].0 as f64, dst[i].1 as f64);
        let r = i * 2;
        a[r] = [-sx, -sy, -1.0, 0.0, 0.0, 0.0, dx * sx, dx * sy, dx];
        a[r + 1] = [0.0, 0.0, 0.0, -sx, -sy, -1.0, dy * sx, dy * sy, dy];
    }

    // Solve via Gaussian elimination on the 8x9 augmented matrix
    // Reduce to row echelon, then back-substitute with h[8] = 1
    let mut mat = a;
    for col in 0..8 {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..8 {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(ImageError::InvalidParameters("degenerate homography".into()));
        }
        mat.swap(col, max_row);

        let pivot = mat[col][col];
        for j in col..9 {
            mat[col][j] /= pivot;
        }
        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            for j in col..9 {
                mat[row][j] -= factor * mat[col][j];
            }
        }
    }

    // h[8] = 1, solve for h[0..8]
    let mut h = [0.0f64; 9];
    h[8] = 1.0;
    for i in 0..8 {
        h[i] = -mat[i][8] as f64;
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip() {
        // Test several colors
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (0, 0, 0), (255, 255, 255)] {
            let rf = r as f64 / 255.0;
            let gf = g as f64 / 255.0;
            let bf = b as f64 / 255.0;
            let (l, a, bv) = rgb_to_lab(rf, gf, bf);
            let (r2, g2, b2) = lab_to_rgb(l, a, bv);
            let r_out = (r2.clamp(0.0, 1.0) * 255.0).round() as u8;
            let g_out = (g2.clamp(0.0, 1.0) * 255.0).round() as u8;
            let b_out = (b2.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert!((r as i16 - r_out as i16).abs() <= 1, "R roundtrip: {r} → {r_out}");
            assert!((g as i16 - g_out as i16).abs() <= 1, "G roundtrip: {g} → {g_out}");
            assert!((b as i16 - b_out as i16).abs() <= 1, "B roundtrip: {b} → {b_out}");
        }
    }

    #[test]
    fn oklab_roundtrip() {
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)] {
            let rf = r as f64 / 255.0;
            let gf = g as f64 / 255.0;
            let bf = b as f64 / 255.0;
            let (l, a, bv) = rgb_to_oklab(rf, gf, bf);
            let (r2, g2, b2) = oklab_to_rgb(l, a, bv);
            let r_out = (r2.clamp(0.0, 1.0) * 255.0).round() as u8;
            let g_out = (g2.clamp(0.0, 1.0) * 255.0).round() as u8;
            let b_out = (b2.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert!((r as i16 - r_out as i16).abs() <= 1, "OKLab R roundtrip: {r} → {r_out}");
            assert!((g as i16 - g_out as i16).abs() <= 1, "OKLab G roundtrip: {g} → {g_out}");
            assert!((b as i16 - b_out as i16).abs() <= 1, "OKLab B roundtrip: {b} → {b_out}");
        }
    }

    #[test]
    fn lab_known_values() {
        // White should be L=100, a≈0, b≈0
        let (l, a, b) = rgb_to_lab(1.0, 1.0, 1.0);
        assert!((l - 100.0).abs() < 0.1, "white L={l}");
        assert!(a.abs() < 0.1, "white a={a}");
        assert!(b.abs() < 0.1, "white b={b}");

        // Black should be L=0
        let (l, _, _) = rgb_to_lab(0.0, 0.0, 0.0);
        assert!(l.abs() < 0.1, "black L={l}");
    }

    #[test]
    fn oklab_known_values() {
        // White: L≈1, a≈0, b≈0
        let (l, a, b) = rgb_to_oklab(1.0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 0.01, "OKLab white L={l}");
        assert!(a.abs() < 0.01, "OKLab white a={a}");
        assert!(b.abs() < 0.01, "OKLab white b={b}");
    }

    #[test]
    fn gray_world_corrects_cast() {
        let info = ImageInfo {
            width: 4, height: 1,
            format: PixelFormat::Rgb8,
            color_space: super::super::types::ColorSpace::Srgb,
        };
        // Blue-tinted image (low R, high B)
        let pixels = vec![50, 100, 200, 60, 110, 190, 40, 90, 210, 55, 105, 195];
        let result = white_balance_gray_world(&pixels, &info).unwrap();
        // After correction, mean R ≈ mean G ≈ mean B
        let n = 4;
        let mean_r: f64 = result.iter().step_by(3).map(|&v| v as f64).sum::<f64>() / n as f64;
        let mean_g: f64 = result.iter().skip(1).step_by(3).map(|&v| v as f64).sum::<f64>() / n as f64;
        let mean_b: f64 = result.iter().skip(2).step_by(3).map(|&v| v as f64).sum::<f64>() / n as f64;
        let spread = (mean_r - mean_g).abs().max((mean_g - mean_b).abs());
        assert!(spread < 5.0, "gray-world should equalize means: R={mean_r:.0} G={mean_g:.0} B={mean_b:.0}");
    }

    #[test]
    fn bradford_d65_to_d50() {
        // Adapt white from D65 to D50: should give D50 white point
        let (x, y, z) = bradford_adapt(D65_X, D65_Y, D65_Z, Illuminant::D65, Illuminant::D50);
        let (dx, dy, dz) = Illuminant::D50.xyz();
        assert!((x - dx).abs() < 0.01, "X: {x} vs {dx}");
        assert!((y - dy).abs() < 0.01, "Y: {y} vs {dy}");
        assert!((z - dz).abs() < 0.01, "Z: {z} vs {dz}");
    }

    #[test]
    fn bradford_identity() {
        // Adapting D65→D65 should be identity
        let (x, y, z) = bradford_adapt(0.5, 0.4, 0.3, Illuminant::D65, Illuminant::D65);
        assert!((x - 0.5).abs() < 0.001);
        assert!((y - 0.4).abs() < 0.001);
        assert!((z - 0.3).abs() < 0.001);
    }

    #[test]
    fn perspective_warp_identity() {
        let info = ImageInfo {
            width: 4, height: 4,
            format: PixelFormat::Gray8,
            color_space: super::super::types::ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..16).collect();
        // Identity mapping: src corners = dst corners
        let corners = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)];
        let result = perspective_warp(&pixels, &info, &corners, &corners, 4, 4).unwrap();
        // Should be very close to original (within bilinear rounding)
        for i in 0..16 {
            assert!(
                (pixels[i] as i16 - result[i] as i16).abs() <= 1,
                "pixel {i}: {}->{}", pixels[i], result[i]
            );
        }
    }
}
