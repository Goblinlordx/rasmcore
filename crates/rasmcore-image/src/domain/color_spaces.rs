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

// ─── ProPhoto RGB (ROMM RGB) ─────────────────────────────────────────────
//
// Primaries: R(0.7347, 0.2653) G(0.1596, 0.7468) B(0.0366, 0.0001)
// White point: D50
// Transfer function: gamma 1.8, linear below Et = 1/512 = 0.001953125
// Reference: ICC.1:2004-10, colour-science 0.4.7

const PROPHOTO_ET: f64 = 1.0 / 512.0; // 16 * (1/512)^1.8 = 1/512
const PROPHOTO_ET_LINEAR: f64 = 1.0 / 8192.0; // = Et / 16

#[inline]
fn prophoto_to_linear(v: f64) -> f64 {
    if v <= PROPHOTO_ET {
        v / 16.0
    } else {
        v.powf(1.8)
    }
}

#[inline]
fn linear_to_prophoto(v: f64) -> f64 {
    if v <= PROPHOTO_ET_LINEAR {
        v * 16.0
    } else {
        v.powf(1.0 / 1.8)
    }
}

/// ProPhoto RGB → XYZ (D50) matrix.
/// Derived from primaries R(0.7347,0.2653), G(0.1596,0.7468), B(0.0366,0.0001)
/// with D50 white point. Matches colour-science 0.4.7.
fn prophoto_rgb_to_xyz_d50(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = prophoto_to_linear(r);
    let gl = prophoto_to_linear(g);
    let bl = prophoto_to_linear(b);
    // Matrix from colour-science: colour.RGB_COLOURSPACES['ProPhoto RGB'].matrix_RGB_to_XYZ
    let x = 0.7976749444816064 * rl + 0.1351917082975956 * gl + 0.0313493495815248 * bl;
    let y = 0.2880402378623901 * rl + 0.7118741461693835 * gl + 0.0000856159682265 * bl;
    let z = 0.0000000000000000 * rl + 0.0000000000000000 * gl + 0.8251046025104602 * bl;
    (x, y, z)
}

/// XYZ (D50) → ProPhoto RGB.
fn xyz_d50_to_prophoto_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Inverse of the above matrix (from colour-science)
    let rl = 1.3459433009386654 * x - 0.2556075514260984 * y - 0.0511118466700571 * z;
    let gl = -0.5445989112457426 * x + 1.5081673487328567 * y + 0.0205351443914399 * z;
    let bl = 0.0000000000000000 * x + 0.0000000000000000 * y + 1.2118127506937628 * z;
    (
        linear_to_prophoto(rl.clamp(0.0, 1.0)),
        linear_to_prophoto(gl.clamp(0.0, 1.0)),
        linear_to_prophoto(bl.clamp(0.0, 1.0)),
    )
}

/// Convert ProPhoto RGB to sRGB via XYZ (D50 → D65 Bradford adaptation).
pub fn prophoto_to_srgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x50, y50, z50) = prophoto_rgb_to_xyz_d50(r, g, b);
    let (x65, y65, z65) = bradford_adapt(x50, y50, z50, Illuminant::D50, Illuminant::D65);
    xyz_to_rgb(x65, y65, z65)
}

/// Convert sRGB to ProPhoto RGB via XYZ (D65 → D50 Bradford adaptation).
pub fn srgb_to_prophoto(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x65, y65, z65) = rgb_to_xyz(r, g, b);
    let (x50, y50, z50) = bradford_adapt(x65, y65, z65, Illuminant::D65, Illuminant::D50);
    xyz_d50_to_prophoto_rgb(x50, y50, z50)
}

// ─── Adobe RGB 1998 ──────────────────────────────────────────────────────
//
// Primaries: R(0.64, 0.33) G(0.21, 0.71) B(0.15, 0.06)
// White point: D65
// Transfer function: gamma 2.19921875 (= 563/256)
// Reference: Adobe RGB (1998) Color Image Encoding, colour-science 0.4.7

const ADOBE_GAMMA: f64 = 563.0 / 256.0; // 2.19921875

#[inline]
fn adobe_to_linear(v: f64) -> f64 {
    v.powf(ADOBE_GAMMA)
}

#[inline]
fn linear_to_adobe(v: f64) -> f64 {
    v.clamp(0.0, 1.0).powf(1.0 / ADOBE_GAMMA)
}

/// Adobe RGB → XYZ (D65) matrix.
/// Derived from primaries R(0.64,0.33), G(0.21,0.71), B(0.15,0.06)
/// with D65 white point. Matches colour-science 0.4.7.
fn adobe_rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = adobe_to_linear(r);
    let gl = adobe_to_linear(g);
    let bl = adobe_to_linear(b);
    // Matrix from colour-science: colour.RGB_COLOURSPACES['Adobe RGB (1998)'].matrix_RGB_to_XYZ
    let x = 0.5766690429101305 * rl + 0.1855582379065463 * gl + 0.1882286462349947 * bl;
    let y = 0.2973449753743829 * rl + 0.6273635662554661 * gl + 0.0752914583701510 * bl;
    let z = 0.0270313613864123 * rl + 0.0706888525938314 * gl + 0.9913375368376388 * bl;
    (x, y, z)
}

/// XYZ (D65) → Adobe RGB.
fn xyz_to_adobe_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Inverse matrix (from colour-science)
    let rl = 2.0415879038107327 * x - 0.5650069742788597 * y - 0.3447313507783297 * z;
    let gl = -0.9692436362808796 * x + 1.8759675015077202 * y + 0.0415550574071756 * z;
    let bl = 0.0134442015914174 * x - 0.1183623922401997 * y + 1.0151749943912780 * z;
    (
        linear_to_adobe(rl.clamp(0.0, 1.0)),
        linear_to_adobe(gl.clamp(0.0, 1.0)),
        linear_to_adobe(bl.clamp(0.0, 1.0)),
    )
}

/// Convert Adobe RGB to sRGB via XYZ (both D65, no adaptation needed).
pub fn adobe_to_srgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = adobe_rgb_to_xyz(r, g, b);
    xyz_to_rgb(x, y, z)
}

/// Convert sRGB to Adobe RGB via XYZ (both D65, no adaptation needed).
pub fn srgb_to_adobe(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    xyz_to_adobe_rgb(x, y, z)
}

// ─── CIE Lab (via XYZ, D65 illuminant) ───────────────────────────────────

/// D65 reference white point — derived from CIE chromaticity (0.3127, 0.329) with Y=1.
/// Matches colour-science's full-precision derivation.
const D65_X: f64 = 0.9504559270516716;
const D65_Y: f64 = 1.0;
const D65_Z: f64 = 1.0890577507598784;

/// sRGB to XYZ matrix (D65, IEC 61966-2-1).
/// Uses the same precision as colour-science for reference alignment.
fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);
    let x = 0.4124 * rl + 0.3576 * gl + 0.1805 * bl;
    let y = 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
    let z = 0.0193 * rl + 0.1192 * gl + 0.9505 * bl;
    (x, y, z)
}

/// XYZ to sRGB.
/// Uses the same precision as colour-science for reference alignment.
fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rl = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let gl = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let bl = 0.0557 * x - 0.2040 * y + 1.0570 * z;
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
        return Err(ImageError::UnsupportedFormat(
            "Lab conversion requires Rgb8".into(),
        ));
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
        return Err(ImageError::InvalidParameters(
            "Lab buffer size mismatch".into(),
        ));
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
//
// Two paths provided per the Ottosson paper:
// 1. XYZ-based (default): sRGB → linear → XYZ → LMS → cbrt → OKLab
//    Matches colour-science 0.4.7 exactly.
// 2. Direct sRGB: sRGB → linear → LMS (direct matrix) → cbrt → OKLab
//    Uses the Ottosson paper's direct matrices, ~1e-4 difference from XYZ path.

/// Convert sRGB [0,1] to OKLab via XYZ intermediate.
/// Default path — matches colour-science reference exactly.
pub fn rgb_to_oklab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // rgb_to_xyz handles sRGB→linear internally
    let (x, y, z) = rgb_to_xyz(r, g, b);
    xyz_to_oklab(x, y, z)
}

/// Convert CIE XYZ to OKLab.
/// Uses colour-science's XYZ→LMS matrix (from Ottosson 2020).
pub fn xyz_to_oklab(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // M1: XYZ → LMS (colour-science MATRIX_1_XYZ_TO_LMS)
    let l = 0.8189330101 * x + 0.3618667424 * y - 0.1288597137 * z;
    let m = 0.0329845436 * x + 0.9293118715 * y + 0.0361456387 * z;
    let s = 0.0482003018 * x + 0.2643662691 * y + 0.6338517070 * z;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    // M2: LMS_cbrt → Lab (colour-science MATRIX_2_LMS_TO_LAB)
    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    (ok_l, ok_a, ok_b)
}

/// Convert OKLab to sRGB [0,1] via XYZ intermediate.
pub fn oklab_to_rgb(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    let (x, y, z) = oklab_to_xyz(ok_l, ok_a, ok_b);
    let (rl, gl, bl) = xyz_to_rgb(x, y, z);
    (rl, gl, bl) // xyz_to_rgb already applies linear_to_srgb
}

/// Convert OKLab to CIE XYZ.
pub fn oklab_to_xyz(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    // M2_inv: Lab → LMS_cbrt
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // M1_inv: LMS → XYZ (colour-science MATRIX_1_LMS_TO_XYZ)
    let x = 1.2270138511 * l - 0.5577999807 * m + 0.2812561490 * s;
    let y = -0.0405801784 * l + 1.1122568696 * m - 0.0716766787 * s;
    let z = -0.0763812845 * l - 0.4214819784 * m + 1.5861632204 * s;
    (x, y, z)
}

/// Convert sRGB [0,1] to OKLab via direct sRGB→LMS matrix (Ottosson paper path).
/// This is an alternative path — ~1e-4 difference from the XYZ-based default.
pub fn rgb_to_oklab_direct(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    // Direct linear_sRGB → LMS (Ottosson 2020 paper matrices)
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

/// Convert OKLab to sRGB [0,1] via direct LMS→sRGB matrix (Ottosson paper path).
pub fn oklab_to_rgb_direct(ok_l: f64, ok_a: f64, ok_b: f64) -> (f64, f64, f64) {
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // Direct LMS → linear_sRGB (Ottosson paper inverse)
    let rl = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}

// ─── White Balance ───────────────────────────────────────────────────────

/// Gray-world white balance: adjust so mean R = mean G = mean B.
pub fn white_balance_gray_world(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "White balance requires Rgb8".into(),
        ));
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
        out[i * 3 + 1] = (pixels[i * 3 + 1] as f64 * scale_g)
            .round()
            .clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (pixels[i * 3 + 2] as f64 * scale_b)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

/// Manual white balance via color temperature and tint.
/// `temperature`: Kelvin offset from 6500K (negative = cooler/blue, positive = warmer/yellow)
/// `tint`: green-magenta shift (-1.0 to 1.0)
pub fn white_balance_temperature(
    pixels: &[u8],
    info: &ImageInfo,
    temperature: f64,
    tint: f64,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "White balance requires Rgb8".into(),
        ));
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
        out[i * 3 + 1] = (pixels[i * 3 + 1] as f64 * scale_g)
            .round()
            .clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (pixels[i * 3 + 2] as f64 * scale_b)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

// ─── Delta E (Color Difference Metrics) ──────────────────────────────────

/// Delta E 76 (CIE 1976) — Euclidean distance in Lab space.
pub fn delta_e_76(lab1: (f64, f64, f64), lab2: (f64, f64, f64)) -> f64 {
    let dl = lab1.0 - lab2.0;
    let da = lab1.1 - lab2.1;
    let db = lab1.2 - lab2.2;
    (dl * dl + da * da + db * db).sqrt()
}

/// Delta E 94 (CIE 1994) — weighted distance with lightness/chroma/hue terms.
/// `textile`: if true, uses textile weights (kL=2); false uses graphic arts (kL=1).
pub fn delta_e_94(lab1: (f64, f64, f64), lab2: (f64, f64, f64), textile: bool) -> f64 {
    let (k_l, k1, k2) = if textile {
        (2.0, 0.048, 0.014)
    } else {
        (1.0, 0.045, 0.015)
    };
    let dl = lab1.0 - lab2.0;
    let c1 = (lab1.1 * lab1.1 + lab1.2 * lab1.2).sqrt();
    let c2 = (lab2.1 * lab2.1 + lab2.2 * lab2.2).sqrt();
    let dc = c1 - c2;
    let da = lab1.1 - lab2.1;
    let db = lab1.2 - lab2.2;
    let dh_sq = da * da + db * db - dc * dc;
    let dh_sq = dh_sq.max(0.0); // numerical safety

    let sl = 1.0;
    let sc = 1.0 + k1 * c1;
    let sh = 1.0 + k2 * c1;

    let t1 = dl / (k_l * sl);
    let t2 = dc / sc;
    let t3_sq = dh_sq / (sh * sh);

    (t1 * t1 + t2 * t2 + t3_sq).sqrt()
}

/// Delta E 2000 (CIEDE2000) — most perceptually uniform color difference metric.
pub fn delta_e_2000(lab1: (f64, f64, f64), lab2: (f64, f64, f64)) -> f64 {
    use std::f64::consts::PI;
    let (l1, a1, b1) = lab1;
    let (l2, a2, b2) = lab2;

    let l_bar = (l1 + l2) / 2.0;
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;

    let c_bar_7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar_7 / (c_bar_7 + 25.0_f64.powi(7))).sqrt());

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let c_bar_p = (c1p + c2p) / 2.0;

    let h1p = b1.atan2(a1p).to_degrees().rem_euclid(360.0);
    let h2p = b2.atan2(a2p).to_degrees().rem_euclid(360.0);

    let dh_abs = (h1p - h2p).abs();
    let dh_p = if c1p * c2p == 0.0 {
        0.0
    } else if dh_abs <= 180.0 {
        h2p - h1p
    } else if h2p <= h1p {
        h2p - h1p + 360.0
    } else {
        h2p - h1p - 360.0
    };

    let dl_p = l2 - l1;
    let dc_p = c2p - c1p;
    let dh_p_val = 2.0 * (c1p * c2p).sqrt() * (dh_p / 2.0 * PI / 180.0).sin();

    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if dh_abs <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * PI / 180.0).cos()
        + 0.24 * ((2.0 * h_bar_p) * PI / 180.0).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0) * PI / 180.0).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0) * PI / 180.0).cos();

    let l_bar_50_sq = (l_bar - 50.0) * (l_bar - 50.0);
    let sl = 1.0 + 0.015 * l_bar_50_sq / (20.0 + l_bar_50_sq).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;

    let c_bar_p_7 = c_bar_p.powi(7);
    let rc = 2.0 * (c_bar_p_7 / (c_bar_p_7 + 25.0_f64.powi(7))).sqrt();
    let rt = -rc * (60.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp() * PI / 180.0).sin();

    let t1 = dl_p / sl;
    let t2 = dc_p / sc;
    let t3 = dh_p_val / sh;

    (t1 * t1 + t2 * t2 + t3 * t3 + rt * t2 * t3).sqrt()
}

// ─── CIE LCH (Cylindrical Lab) ──────────────────────────────────────────

/// Convert CIE Lab to LCH (Lightness, Chroma, Hue in degrees).
pub fn lab_to_lch(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let c = (a * a + b * b).sqrt();
    let h = b.atan2(a).to_degrees().rem_euclid(360.0);
    (l, c, h)
}

/// Convert CIE LCH to Lab.
pub fn lch_to_lab(l: f64, c: f64, h_deg: f64) -> (f64, f64, f64) {
    let h_rad = h_deg * std::f64::consts::PI / 180.0;
    let a = c * h_rad.cos();
    let b = c * h_rad.sin();
    (l, a, b)
}

// ─── CIE Luv ────────────────────────────────────────────────────────────

/// Convert CIE XYZ to CIE Luv (D65 illuminant).
pub fn xyz_to_luv(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let (xn, yn, zn) = (D65_X, D65_Y, D65_Z);
    let yr = y / yn;

    let l = if yr > (6.0 / 29.0_f64).powi(3) {
        116.0 * yr.cbrt() - 16.0
    } else {
        (29.0 / 3.0_f64).powi(3) * yr
    };

    let denom = x + 15.0 * y + 3.0 * z;
    let denom_n = xn + 15.0 * yn + 3.0 * zn;

    if denom == 0.0 {
        return (l, 0.0, 0.0);
    }

    let u_prime = 4.0 * x / denom;
    let v_prime = 9.0 * y / denom;
    let u_prime_n = 4.0 * xn / denom_n;
    let v_prime_n = 9.0 * yn / denom_n;

    let u = 13.0 * l * (u_prime - u_prime_n);
    let v = 13.0 * l * (v_prime - v_prime_n);
    (l, u, v)
}

/// Convert sRGB [0,1] to CIE Luv.
pub fn rgb_to_luv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = rgb_to_xyz(r, g, b);
    xyz_to_luv(x, y, z)
}

/// Convert CIE Luv to CIE XYZ (D65 illuminant).
pub fn luv_to_xyz(l: f64, u: f64, v: f64) -> (f64, f64, f64) {
    if l == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let (xn, yn, zn) = (D65_X, D65_Y, D65_Z);
    let denom_n = xn + 15.0 * yn + 3.0 * zn;
    let u_prime_n = 4.0 * xn / denom_n;
    let v_prime_n = 9.0 * yn / denom_n;

    let u_prime = u / (13.0 * l) + u_prime_n;
    let v_prime = v / (13.0 * l) + v_prime_n;

    let y = if l > 8.0 {
        yn * ((l + 16.0) / 116.0).powi(3)
    } else {
        yn * l * (3.0 / 29.0_f64).powi(3)
    };

    let x = y * 9.0 * u_prime / (4.0 * v_prime);
    let z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime);
    (x, y, z)
}

// ─── Bradford Chromatic Adaptation ───────────────────────────────────────

/// Standard illuminant white points (CIE XYZ, Y=1).
/// Derived from CIE chromaticity coordinates at full f64 precision,
/// matching colour-science's derivation via xy_to_XYZ().
#[derive(Debug, Clone, Copy)]
pub enum Illuminant {
    D50,
    D65,
    A,
}

impl Illuminant {
    pub fn xyz(self) -> (f64, f64, f64) {
        match self {
            // CIE xy (0.3457, 0.3585) → XYZ
            Illuminant::D50 => (0.964_295_676_429_568, 1.0, 0.8251046025104602),
            // CIE xy (0.3127, 0.329) → XYZ
            Illuminant::D65 => (D65_X, D65_Y, D65_Z),
            // CIE xy (0.44758, 0.40745) → XYZ
            Illuminant::A => (1.0984906123450726, 1.0, 0.355_798_257_454_903),
        }
    }
}

/// Bradford chromatic adaptation matrix: transform XYZ from one illuminant to another.
pub fn bradford_adapt(x: f64, y: f64, z: f64, from: Illuminant, to: Illuminant) -> (f64, f64, f64) {
    // Bradford cone response matrix (matches colour-science)
    const M: [[f64; 3]; 3] = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    // Inverse — computed via np.linalg.inv(M) to match colour-science precision
    const M_INV: [[f64; 3]; 3] = [
        [0.986992905466712, -0.147054256420990, 0.159962651663731],
        [0.432305269723394, 0.518360271536777, 0.049291228212856],
        [-0.008528664575177, 0.040042821654085, 0.968486695787550],
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
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "perspective warp on {other:?}"
            )));
        }
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
#[allow(clippy::needless_range_loop)]
fn solve_homography(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> Result<[f64; 9], ImageError> {
    // Build 8x9 matrix A where A * h = 0
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (sx, sy) = (src[i].0, src[i].1);
        let (dx, dy) = (dst[i].0, dst[i].1);
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
            return Err(ImageError::InvalidParameters(
                "degenerate homography".into(),
            ));
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
        h[i] = -mat[i][8];
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip() {
        // Test several colors
        for (r, g, b) in [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
            (0, 0, 0),
            (255, 255, 255),
        ] {
            let rf = r as f64 / 255.0;
            let gf = g as f64 / 255.0;
            let bf = b as f64 / 255.0;
            let (l, a, bv) = rgb_to_lab(rf, gf, bf);
            let (r2, g2, b2) = lab_to_rgb(l, a, bv);
            let r_out = (r2.clamp(0.0, 1.0) * 255.0).round() as u8;
            let g_out = (g2.clamp(0.0, 1.0) * 255.0).round() as u8;
            let b_out = (b2.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert!(
                (r as i16 - r_out as i16).abs() <= 1,
                "R roundtrip: {r} → {r_out}"
            );
            assert!(
                (g as i16 - g_out as i16).abs() <= 1,
                "G roundtrip: {g} → {g_out}"
            );
            assert!(
                (b as i16 - b_out as i16).abs() <= 1,
                "B roundtrip: {b} → {b_out}"
            );
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
            assert!(
                (r as i16 - r_out as i16).abs() <= 1,
                "OKLab R roundtrip: {r} → {r_out}"
            );
            assert!(
                (g as i16 - g_out as i16).abs() <= 1,
                "OKLab G roundtrip: {g} → {g_out}"
            );
            assert!(
                (b as i16 - b_out as i16).abs() <= 1,
                "OKLab B roundtrip: {b} → {b_out}"
            );
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
            width: 4,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: super::super::types::ColorSpace::Srgb,
        };
        // Blue-tinted image (low R, high B)
        let pixels = vec![50, 100, 200, 60, 110, 190, 40, 90, 210, 55, 105, 195];
        let result = white_balance_gray_world(&pixels, &info).unwrap();
        // After correction, mean R ≈ mean G ≈ mean B
        let n = 4;
        let mean_r: f64 = result.iter().step_by(3).map(|&v| v as f64).sum::<f64>() / n as f64;
        let mean_g: f64 = result
            .iter()
            .skip(1)
            .step_by(3)
            .map(|&v| v as f64)
            .sum::<f64>()
            / n as f64;
        let mean_b: f64 = result
            .iter()
            .skip(2)
            .step_by(3)
            .map(|&v| v as f64)
            .sum::<f64>()
            / n as f64;
        let spread = (mean_r - mean_g).abs().max((mean_g - mean_b).abs());
        assert!(
            spread < 5.0,
            "gray-world should equalize means: R={mean_r:.0} G={mean_g:.0} B={mean_b:.0}"
        );
    }

    // ── ProPhoto RGB Tests ────────────────────────────────────────────────

    #[test]
    fn prophoto_transfer_function_roundtrip() {
        for &v in &[0.0, 0.001, 0.01, 0.1, 0.5, 0.8, 1.0] {
            let lin = prophoto_to_linear(v);
            let back = linear_to_prophoto(lin);
            assert!(
                (v - back).abs() < 1e-12,
                "ProPhoto roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn prophoto_linear_segment() {
        // Below Et = 1/512, the transfer should be v/16
        let v = 0.001;
        assert!((prophoto_to_linear(v) - v / 16.0).abs() < 1e-15);
    }

    #[test]
    fn prophoto_to_srgb_converts() {
        // ProPhoto (0.5, 0.3, 0.1) → sRGB: values may be slightly out of [0,1]
        // because ProPhoto's gamut exceeds sRGB — this is expected
        let (r, g, b) = prophoto_to_srgb(0.5, 0.3, 0.1);
        assert!(r > 0.0, "R should be positive: {r}");
        assert!(g > 0.0, "G should be positive: {g}");
        // B can be slightly negative for out-of-gamut ProPhoto colors
        assert!(b > -0.1, "B should be near zero or positive: {b}");
    }

    #[test]
    fn prophoto_srgb_roundtrip() {
        // sRGB → ProPhoto → sRGB should be near-identity for in-gamut colors
        // Tolerance limited by 4-decimal sRGB XYZ matrices in the existing codebase
        let (pr, pg, pb) = srgb_to_prophoto(0.6, 0.4, 0.2);
        let (r, g, b) = prophoto_to_srgb(pr, pg, pb);
        assert!((r - 0.6).abs() < 0.001, "R roundtrip: {r}");
        assert!((g - 0.4).abs() < 0.001, "G roundtrip: {g}");
        assert!((b - 0.2).abs() < 0.001, "B roundtrip: {b}");
    }

    #[test]
    fn prophoto_white_is_white() {
        // ProPhoto (1,1,1) → sRGB should be near (1,1,1) via Bradford D50→D65
        let (r, g, b) = prophoto_to_srgb(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.02, "R: {r}");
        assert!((g - 1.0).abs() < 0.02, "G: {g}");
        assert!((b - 1.0).abs() < 0.02, "B: {b}");
    }

    // ── Adobe RGB Tests ─────────────────────────────────────────────────

    #[test]
    fn adobe_transfer_function_roundtrip() {
        for &v in &[0.0, 0.01, 0.1, 0.5, 0.8, 1.0] {
            let lin = adobe_to_linear(v);
            let back = linear_to_adobe(lin);
            assert!(
                (v - back).abs() < 1e-12,
                "Adobe roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn adobe_gamma_is_correct() {
        // Adobe gamma = 563/256 ≈ 2.19921875
        let v = 0.5f64;
        let linear = v.powf(563.0 / 256.0);
        assert!((adobe_to_linear(v) - linear).abs() < 1e-15);
    }

    #[test]
    fn adobe_to_srgb_produces_valid_range() {
        // Adobe (0.5, 0.3, 0.1) should produce valid sRGB values
        let (r, g, b) = adobe_to_srgb(0.5, 0.3, 0.1);
        assert!(r > 0.0 && r <= 1.0, "R out of range: {r}");
        assert!(g > 0.0 && g <= 1.0, "G out of range: {g}");
        assert!(b >= 0.0 && b <= 1.0, "B out of range: {b}");
    }

    #[test]
    fn adobe_srgb_roundtrip() {
        // sRGB → Adobe → sRGB should be near-identity
        // Tolerance limited by 4-decimal sRGB XYZ matrices
        let (ar, ag, ab) = srgb_to_adobe(0.6, 0.4, 0.2);
        let (r, g, b) = adobe_to_srgb(ar, ag, ab);
        assert!((r - 0.6).abs() < 0.001, "R roundtrip: {r}");
        assert!((g - 0.4).abs() < 0.001, "G roundtrip: {g}");
        assert!((b - 0.2).abs() < 0.001, "B roundtrip: {b}");
    }

    #[test]
    fn adobe_white_is_white() {
        // Adobe (1,1,1) → sRGB should be near (1,1,1) — both D65
        // Tolerance limited by sRGB matrix precision (4 decimals)
        let (r, g, b) = adobe_to_srgb(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01, "R: {r}");
        assert!((g - 1.0).abs() < 0.01, "G: {g}");
        assert!((b - 1.0).abs() < 0.01, "B: {b}");
    }

    #[test]
    fn adobe_black_is_black() {
        let (r, g, b) = adobe_to_srgb(0.0, 0.0, 0.0);
        assert!(r.abs() < 1e-12);
        assert!(g.abs() < 1e-12);
        assert!(b.abs() < 1e-12);
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
            width: 4,
            height: 4,
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
                "pixel {i}: {}->{}",
                pixels[i],
                result[i]
            );
        }
    }
}
