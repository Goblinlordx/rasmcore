//! Color math — transfer functions, matrices, and color space conversions.
//!
//! All functions operate on f32 channel values (linear [0,1] or unbounded HDR).
//! Transfer functions are per-channel. Matrix operations are per-pixel (3 channels).

// ─── sRGB Transfer Functions ──────────────────────────────────────────────────
// IEC 61966-2-1 standard

/// sRGB gamma-encoded → Linear. Per-channel.
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear → sRGB gamma-encoded. Per-channel.
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ─── ACEScct Transfer Functions ───────────────────────────────────────────────
// S-2016-001

const ACESCCT_CUT: f32 = 0.0078125; // 2^-7
const ACESCCT_A: f32 = 10.540_237;
const ACESCCT_B: f32 = 0.072_905_53;
const ACESCCT_CUT_LOG: f32 = 0.155_251_15;

/// ACEScg Linear → ACEScct Log. Per-channel.
#[inline]
pub fn linear_to_acescct(v: f32) -> f32 {
    if v <= ACESCCT_CUT {
        ACESCCT_A * v + ACESCCT_B
    } else {
        (v.log2() + 9.72) / 17.52
    }
}

/// ACEScct Log → ACEScg Linear. Per-channel.
#[inline]
pub fn acescct_to_linear(v: f32) -> f32 {
    if v <= ACESCCT_CUT_LOG {
        (v - ACESCCT_B) / ACESCCT_A
    } else {
        2.0f32.powf(v * 17.52 - 9.72)
    }
}

// ─── ACEScc Transfer Functions ────────────────────────────────────────────────
// S-2014-003

/// ACEScg Linear → ACEScc Log. Per-channel.
#[inline]
pub fn linear_to_acescc(v: f32) -> f32 {
    if v <= 0.0 {
        // Clamp negative to minimum representable
        (2.0f32.powi(-16).log2() + 9.72) / 17.52
    } else if v < 2.0f32.powi(-15) {
        (((2.0f32.powi(-16)) + v * 0.5).log2() + 9.72) / 17.52
    } else {
        (v.log2() + 9.72) / 17.52
    }
}

/// ACEScc Log → ACEScg Linear. Per-channel.
#[inline]
pub fn acescc_to_linear(v: f32) -> f32 {
    let min_val = (9.72 - 15.0) / 17.52;
    if v <= min_val {
        (2.0f32.powf(v * 17.52 - 9.72) - 2.0f32.powi(-16)) * 2.0
    } else {
        2.0f32.powf(v * 17.52 - 9.72)
    }
}

// ─── 3x3 Matrix Operations ───────────────────────────────────────────────────

/// 3x3 matrix (row-major) for color space conversions.
pub type Mat3 = [f64; 9];

/// Multiply a 3x3 matrix by an RGB triplet. f64 intermediate precision.
#[inline]
pub fn mat3_mul(m: &Mat3, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let rd = r as f64;
    let gd = g as f64;
    let bd = b as f64;
    (
        (m[0] * rd + m[1] * gd + m[2] * bd) as f32,
        (m[3] * rd + m[4] * gd + m[5] * bd) as f32,
        (m[6] * rd + m[7] * gd + m[8] * bd) as f32,
    )
}

/// Multiply two 3x3 matrices (row-major). Result = A * B.
pub fn mat3_compose(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    r
}

// ─── ACES Color Space Matrices ────────────────────────────────────────────────
// From aces-dev CTL: ACESlib.Utilities_Color.ctl

/// sRGB linear (D65) → XYZ (D65)
pub const SRGB_TO_XYZ_D65: Mat3 = [
    0.4123907993,
    0.3575843394,
    0.1804807884,
    0.2126390059,
    0.7151686788,
    0.0721923154,
    0.0193308187,
    0.1191947798,
    0.9505321522,
];

/// XYZ (D65) → sRGB linear (D65)
pub const XYZ_D65_TO_SRGB: Mat3 = [
    3.2409699419,
    -1.5373831776,
    -0.4986107603,
    -0.9692436363,
    1.8759675015,
    0.0415550574,
    0.0556300797,
    -0.2039769589,
    1.0569715142,
];

/// Bradford chromatic adaptation: D65 → D60
pub const BRADFORD_D65_TO_D60: Mat3 = [
    1.01303,
    0.00610531,
    -0.014971,
    0.00769823,
    0.998165,
    -0.00503203,
    -0.00284131,
    0.00468516,
    0.924507,
];

/// Bradford chromatic adaptation: D60 → D65
pub const BRADFORD_D60_TO_D65: Mat3 = [
    0.987224,
    -0.00611327,
    0.0159533,
    -0.00759836,
    1.00186,
    0.00533002,
    0.00307257,
    -0.00509596,
    1.08168,
];

/// AP0 (ACES2065-1) → XYZ (D60)
pub const AP0_TO_XYZ_D60: Mat3 = [
    0.9525523959,
    0.0000000000,
    0.0000936786,
    0.3439664498,
    0.7281660966,
    -0.0721325464,
    0.0000000000,
    0.0000000000,
    1.0088251844,
];

/// XYZ (D60) → AP0 (ACES2065-1)
pub const XYZ_D60_TO_AP0: Mat3 = [
    1.0498110175,
    0.0000000000,
    -0.0000974845,
    -0.4959030231,
    1.3733130458,
    0.0982400361,
    0.0000000000,
    0.0000000000,
    0.9912520182,
];

/// AP1 (ACEScg) → XYZ (D60)
pub const AP1_TO_XYZ_D60: Mat3 = [
    0.6624541811,
    0.1340042065,
    0.1561876870,
    0.2722287168,
    0.6740817658,
    0.0536895174,
    -0.0055746495,
    0.0040607335,
    1.0103391003,
];

/// XYZ (D60) → AP1 (ACEScg)
pub const XYZ_D60_TO_AP1: Mat3 = [
    1.6410233797,
    -0.3248032942,
    -0.2364246952,
    -0.6636628587,
    1.6153315917,
    0.0167563477,
    0.0117218943,
    -0.0082844420,
    0.9883948585,
];

// ─── Convenience: Pre-composed Conversion Matrices ────────────────────────────

/// sRGB linear (D65) → ACEScg (AP1, D60)
pub fn srgb_to_acescg_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| {
        mat3_compose(
            &XYZ_D60_TO_AP1,
            &mat3_compose(&BRADFORD_D65_TO_D60, &SRGB_TO_XYZ_D65),
        )
    });
    *M
}

/// ACEScg (AP1, D60) → sRGB linear (D65)
pub fn acescg_to_srgb_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| {
        mat3_compose(
            &XYZ_D65_TO_SRGB,
            &mat3_compose(&BRADFORD_D60_TO_D65, &AP1_TO_XYZ_D60),
        )
    });
    *M
}

/// AP0 → AP1 (same illuminant, gamut change only)
pub fn ap0_to_ap1_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| mat3_compose(&XYZ_D60_TO_AP1, &AP0_TO_XYZ_D60));
    *M
}

/// AP1 → AP0
pub fn ap1_to_ap0_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| mat3_compose(&XYZ_D60_TO_AP0, &AP1_TO_XYZ_D60));
    *M
}

// ─── Pixel-level Color Conversions ────────────────────────────────────────────

/// Apply a per-channel transfer function to f32 RGBA pixel data.
///
/// Operates on all R, G, B channels. Alpha is passed through unchanged.
pub fn apply_transfer(pixels: &mut [f32], f: fn(f32) -> f32) {
    for chunk in pixels.chunks_exact_mut(4) {
        chunk[0] = f(chunk[0]);
        chunk[1] = f(chunk[1]);
        chunk[2] = f(chunk[2]);
        // chunk[3] (alpha) unchanged
    }
}

/// Apply a 3x3 matrix to f32 RGBA pixel data.
///
/// Transforms R, G, B via the matrix. Alpha unchanged.
pub fn apply_matrix(pixels: &mut [f32], m: &Mat3) {
    for chunk in pixels.chunks_exact_mut(4) {
        let (r, g, b) = mat3_mul(m, chunk[0], chunk[1], chunk[2]);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
}

// ─── High-level Color Space Conversion ────────────────────────────────────────

use crate::color_space::ColorSpace;

/// Convert f32 RGBA pixel data between color spaces.
///
/// Handles the full conversion path: transfer function + matrix + transfer.
/// Modifies pixels in-place.
pub fn convert_color_space(pixels: &mut [f32], from: ColorSpace, to: ColorSpace) {
    if from == to {
        return;
    }

    // Strategy: convert FROM → Linear sRGB → TO
    // For ACES spaces, go through ACEScg linear as intermediate.

    // Step 1: FROM → Linear sRGB
    match from {
        ColorSpace::Linear => {} // already linear sRGB
        ColorSpace::Srgb => apply_transfer(pixels, srgb_to_linear),
        ColorSpace::AcesCg => apply_matrix(pixels, &acescg_to_srgb_matrix()),
        ColorSpace::AcesCct => {
            apply_transfer(pixels, acescct_to_linear);
            apply_matrix(pixels, &acescg_to_srgb_matrix());
        }
        ColorSpace::AcesCc => {
            apply_transfer(pixels, acescc_to_linear);
            apply_matrix(pixels, &acescg_to_srgb_matrix());
        }
        ColorSpace::Aces2065_1 => {
            let m = mat3_compose(&acescg_to_srgb_matrix(), &ap0_to_ap1_matrix());
            apply_matrix(pixels, &m);
        }
        _ => {} // Unknown/unmanaged — pass through
    }

    // Step 2: Linear sRGB → TO
    match to {
        ColorSpace::Linear => {} // already there
        ColorSpace::Srgb => apply_transfer(pixels, linear_to_srgb),
        ColorSpace::AcesCg => apply_matrix(pixels, &srgb_to_acescg_matrix()),
        ColorSpace::AcesCct => {
            apply_matrix(pixels, &srgb_to_acescg_matrix());
            apply_transfer(pixels, linear_to_acescct);
        }
        ColorSpace::AcesCc => {
            apply_matrix(pixels, &srgb_to_acescg_matrix());
            apply_transfer(pixels, linear_to_acescc);
        }
        ColorSpace::Aces2065_1 => {
            let m = mat3_compose(&ap1_to_ap0_matrix(), &srgb_to_acescg_matrix());
            apply_matrix(pixels, &m);
        }
        _ => {} // Unknown
    }
}

// ─── Codec Conversion Helpers ─────────────────────────────────────────────────
// These helpers bridge between the f32 pipeline and native codec formats.
// Codec implementations are independent — they work with native pixel bytes.
// These helpers are used by codec adapters, not by codecs themselves.

/// Convert f32 linear RGBA → sRGB u8 RGBA (gamma encode + quantize).
/// Alpha channel is quantized directly (no gamma).
pub fn f32_linear_to_srgb_rgba8(pixels: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        out.push((linear_to_srgb(chunk[0]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((linear_to_srgb(chunk[1]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((linear_to_srgb(chunk[2]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((chunk[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }
    out
}

/// Convert f32 linear RGBA → sRGB u8 RGB (gamma encode + quantize, drop alpha).
pub fn f32_linear_to_srgb_rgb8(pixels: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() / 4 * 3);
    for chunk in pixels.chunks_exact(4) {
        out.push((linear_to_srgb(chunk[0]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((linear_to_srgb(chunk[1]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((linear_to_srgb(chunk[2]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }
    out
}

/// Convert sRGB u8 RGBA → f32 linear RGBA (degamma + normalize).
pub fn srgb_rgba8_to_f32_linear(pixels: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        out.push(srgb_to_linear(chunk[0] as f32 / 255.0));
        out.push(srgb_to_linear(chunk[1] as f32 / 255.0));
        out.push(srgb_to_linear(chunk[2] as f32 / 255.0));
        out.push(chunk[3] as f32 / 255.0); // alpha: no gamma
    }
    out
}

/// Convert sRGB u8 RGB → f32 linear RGBA (degamma + normalize, alpha = 1.0).
pub fn srgb_rgb8_to_f32_linear(pixels: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels.len() / 3 * 4);
    for chunk in pixels.chunks_exact(3) {
        out.push(srgb_to_linear(chunk[0] as f32 / 255.0));
        out.push(srgb_to_linear(chunk[1] as f32 / 255.0));
        out.push(srgb_to_linear(chunk[2] as f32 / 255.0));
        out.push(1.0); // opaque alpha
    }
    out
}

// ─── OKLab / OKLCH Color Model ──────────────────────────────────────────────
// Ottosson, B. (2020). "A perceptual color space for image processing."
// https://bottosson.github.io/posts/oklab/
// Also: W3C CSS Color Level 4, Section 8.

/// M1: linear sRGB → LMS (Ottosson 2020).
const OKLAB_M1: [[f32; 3]; 3] = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
];

/// M2: cube-rooted LMS → OKLab (Ottosson 2020).
const OKLAB_M2: [[f32; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

/// M2 inverse: OKLab → cube-rooted LMS.
const OKLAB_M2_INV: [[f32; 3]; 3] = [
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
];

/// M1 inverse: LMS → linear sRGB.
const OKLAB_M1_INV: [[f32; 3]; 3] = [
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
];

/// Convert linear sRGB to OKLab (L, a, b).
///
/// L is lightness [0, 1], a and b are chromatic components (unbounded, typically ±0.5).
#[inline]
pub fn linear_srgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Step 1: linear sRGB -> LMS
    let l = OKLAB_M1[0][0] * r + OKLAB_M1[0][1] * g + OKLAB_M1[0][2] * b;
    let m = OKLAB_M1[1][0] * r + OKLAB_M1[1][1] * g + OKLAB_M1[1][2] * b;
    let s = OKLAB_M1[2][0] * r + OKLAB_M1[2][1] * g + OKLAB_M1[2][2] * b;

    // Step 2: cube root
    let l_ = l.max(0.0).cbrt();
    let m_ = m.max(0.0).cbrt();
    let s_ = s.max(0.0).cbrt();

    // Step 3: cube-rooted LMS -> OKLab
    let ok_l = OKLAB_M2[0][0] * l_ + OKLAB_M2[0][1] * m_ + OKLAB_M2[0][2] * s_;
    let ok_a = OKLAB_M2[1][0] * l_ + OKLAB_M2[1][1] * m_ + OKLAB_M2[1][2] * s_;
    let ok_b = OKLAB_M2[2][0] * l_ + OKLAB_M2[2][1] * m_ + OKLAB_M2[2][2] * s_;

    (ok_l, ok_a, ok_b)
}

/// Convert OKLab (L, a, b) to linear sRGB.
#[inline]
pub fn oklab_to_linear_srgb(ok_l: f32, ok_a: f32, ok_b: f32) -> (f32, f32, f32) {
    // Step 1: OKLab -> cube-rooted LMS
    let l_ = OKLAB_M2_INV[0][0] * ok_l + OKLAB_M2_INV[0][1] * ok_a + OKLAB_M2_INV[0][2] * ok_b;
    let m_ = OKLAB_M2_INV[1][0] * ok_l + OKLAB_M2_INV[1][1] * ok_a + OKLAB_M2_INV[1][2] * ok_b;
    let s_ = OKLAB_M2_INV[2][0] * ok_l + OKLAB_M2_INV[2][1] * ok_a + OKLAB_M2_INV[2][2] * ok_b;

    // Step 2: cube (inverse of cube root)
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // Step 3: LMS -> linear sRGB
    let r = OKLAB_M1_INV[0][0] * l + OKLAB_M1_INV[0][1] * m + OKLAB_M1_INV[0][2] * s;
    let g = OKLAB_M1_INV[1][0] * l + OKLAB_M1_INV[1][1] * m + OKLAB_M1_INV[1][2] * s;
    let b = OKLAB_M1_INV[2][0] * l + OKLAB_M1_INV[2][1] * m + OKLAB_M1_INV[2][2] * s;

    (r, g, b)
}

/// Convert OKLab (L, a, b) to OKLCH (L, C, h).
///
/// C = chroma (distance from achromatic axis), h = hue angle in radians.
#[inline]
pub fn oklab_to_oklch(ok_l: f32, ok_a: f32, ok_b: f32) -> (f32, f32, f32) {
    let c = (ok_a * ok_a + ok_b * ok_b).sqrt();
    let h = ok_b.atan2(ok_a);
    (ok_l, c, h)
}

/// Convert OKLCH (L, C, h) to OKLab (L, a, b).
#[inline]
pub fn oklch_to_oklab(ok_l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let a = c * h.cos();
    let b = c * h.sin();
    (ok_l, a, b)
}

// ─── CIE Colorimetry ─────────────────────────────────────────────────────────
// CIE 015:2018 D-illuminant series, Planckian locus, CAT16 adaptation.

/// Standard CIE D65 illuminant chromaticity (CIE 015:2018, Table 1).
///
/// This is the canonical D65 value from the CIE standard illuminant table,
/// NOT computed from the D-illuminant formula at 6500K (which gives slightly
/// different values because D65 has CCT ≈ 6504K, not 6500K).
///
/// Use this for white balance target, sRGB whitepoint, etc.
pub const CIE_D65_XY: (f32, f32) = (0.31270, 0.32900);

/// CIE D-illuminant chromaticity (CIE 015:2018, eq 4.1).
///
/// Valid for 4000K <= T <= 25000K. Outside this range, clamps to boundary.
/// Returns (x, y) CIE 1931 chromaticity coordinates.
///
/// NOTE: For D65 specifically, use `CIE_D65_XY` constant instead.
/// The formula at 6500K gives slightly different values.
pub fn cie_d_illuminant_xy(temperature_k: f32) -> (f32, f32) {
    let t = temperature_k.clamp(4000.0, 25000.0) as f64;
    let t2 = t * t;
    let t3 = t2 * t;

    let xd = if t <= 7000.0 {
        -4.6070e9 / t3 + 2.9678e6 / t2 + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t3 + 1.9018e6 / t2 + 0.24748e3 / t + 0.237040
    };
    let yd = -3.000 * xd * xd + 2.870 * xd - 0.275;
    (xd as f32, yd as f32)
}

/// Planckian locus xy approximation (Hernandez-Andres et al. 1999).
///
/// Valid for 1667K <= T <= 25000K. Wider range than D-illuminant series.
pub fn planckian_locus_xy(temperature_k: f32) -> (f32, f32) {
    let t = temperature_k.clamp(1667.0, 25000.0) as f64;
    let t2 = t * t;
    let t3 = t2 * t;

    let xc = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };

    let xc2 = xc * xc;
    let xc3 = xc2 * xc;
    let yc = if t <= 2222.0 {
        -1.1063814 * xc3 - 1.34811020 * xc2 + 2.18555832 * xc - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * xc3 - 1.37418593 * xc2 + 2.09137015 * xc - 0.16748867
    } else {
        3.0817580 * xc3 - 5.87338670 * xc2 + 3.75112997 * xc - 0.37001483
    };
    (xc as f32, yc as f32)
}

/// Convert CIE xy chromaticity to XYZ (Y=1 normalization).
#[inline]
pub fn xy_to_xyz(x: f32, y: f32) -> [f32; 3] {
    if y.abs() < 1e-10 {
        return [0.0, 1.0, 0.0];
    }
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// CAT16 forward matrix (Li et al. 2017, Table 1).
pub const CAT16: [[f64; 3]; 3] = [
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127],
];

/// CAT16 inverse matrix (precomputed via Cramer's rule at f64 precision).
pub const CAT16_INV: [[f64; 3]; 3] = [
    [
        1.862067855087232715,
        -1.011254630531684295,
        0.149186775444451747,
    ],
    [
        0.387526543236137111,
        0.621447441931475275,
        -0.008973985167612520,
    ],
    [
        -0.015841498849333856,
        -0.034122938028515563,
        1.049964436877849350,
    ],
];

/// sRGB to XYZ (D65) matrix — IEC 61966-2-1.
const M_SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// XYZ (D65) to sRGB matrix — IEC 61966-2-1 inverse.
const M_XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

/// Compute the CAT16 chromatic adaptation matrix from source to target illuminant.
///
/// Both source and target are specified as CIE xy chromaticity.
/// Returns a row-major 3x3 matrix that operates in **linear sRGB space**:
/// `[r', g', b'] = M * [r, g, b]`.
///
/// Internally computes the XYZ-space Von Kries adaptation, then wraps with
/// sRGB↔XYZ transforms: `M_srgb = M_XYZ_to_sRGB × M_cat16_xyz × M_sRGB_to_XYZ`.
pub fn cat16_adaptation_matrix(source_xy: (f32, f32), target_xy: (f32, f32)) -> [f32; 9] {
    let src_xyz = xy_to_xyz(source_xy.0, source_xy.1);
    let tgt_xyz = xy_to_xyz(target_xy.0, target_xy.1);

    let src = [src_xyz[0] as f64, src_xyz[1] as f64, src_xyz[2] as f64];
    let tgt = [tgt_xyz[0] as f64, tgt_xyz[1] as f64, tgt_xyz[2] as f64];

    let src_cone = mat3x3_mul_vec(&CAT16, &src);
    let tgt_cone = mat3x3_mul_vec(&CAT16, &tgt);

    let d = [
        tgt_cone[0] / src_cone[0],
        tgt_cone[1] / src_cone[1],
        tgt_cone[2] / src_cone[2],
    ];

    let d_cat = [
        [d[0] * CAT16[0][0], d[0] * CAT16[0][1], d[0] * CAT16[0][2]],
        [d[1] * CAT16[1][0], d[1] * CAT16[1][1], d[1] * CAT16[1][2]],
        [d[2] * CAT16[2][0], d[2] * CAT16[2][1], d[2] * CAT16[2][2]],
    ];

    // XYZ-space adaptation matrix
    let m_xyz = mat3x3_mul(&CAT16_INV, &d_cat);

    // Wrap with sRGB↔XYZ: M_srgb = M_XYZ_to_sRGB × M_xyz × M_sRGB_to_XYZ
    let m_temp = mat3x3_mul(&m_xyz, &M_SRGB_TO_XYZ);
    let m = mat3x3_mul(&M_XYZ_TO_SRGB, &m_temp);

    [
        m[0][0] as f32,
        m[0][1] as f32,
        m[0][2] as f32,
        m[1][0] as f32,
        m[1][1] as f32,
        m[1][2] as f32,
        m[2][0] as f32,
        m[2][1] as f32,
        m[2][2] as f32,
    ]
}

/// Apply a tint shift perpendicular to the Planckian locus.
///
/// `tint` in [-1, 1], mapped to duv [-0.02, 0.02] (green-magenta range).
pub fn tint_shift_xy(xy: (f32, f32), tint: f32) -> (f32, f32) {
    if tint.abs() < 1e-6 {
        return xy;
    }
    let (x, y) = (xy.0 as f64, xy.1 as f64);
    let denom = -2.0 * x + 12.0 * y + 3.0;
    let u = 4.0 * x / denom;
    let v = 6.0 * y / denom;
    let v_shifted = v - tint as f64 * 0.02;
    let denom2 = 2.0 * u - 8.0 * v_shifted + 4.0;
    let x_out = 3.0 * u / denom2;
    let y_out = 2.0 * v_shifted / denom2;
    (x_out as f32, y_out as f32)
}

fn mat3x3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3x3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Apply a row-major 3x3 matrix to a linear RGB pixel.
#[inline]
pub fn apply_3x3(m: &[f32; 9], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0] * r + m[1] * g + m[2] * b,
        m[3] * r + m[4] * g + m[5] * b,
        m[6] * r + m[7] * g + m[8] * b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrip() {
        for i in 0..256 {
            let v = i as f32 / 255.0;
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!(
                (v - back).abs() < 1e-5,
                "roundtrip failed at {v}: got {back}"
            );
        }
    }

    #[test]
    fn srgb_to_linear_known_values() {
        assert!((srgb_to_linear(0.0) - 0.0).abs() < 1e-7);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-7);
        // Mid-gray: sRGB 0.5 → linear ≈ 0.214
        assert!((srgb_to_linear(0.5) - 0.214).abs() < 0.001);
    }

    #[test]
    fn acescct_linear_roundtrip() {
        let values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0];
        for &v in &values {
            let log = linear_to_acescct(v);
            let back = acescct_to_linear(log);
            assert!(
                (v - back).abs() < 1e-5,
                "ACEScct roundtrip failed at {v}: got {back}"
            );
        }
    }

    #[test]
    fn acescc_linear_roundtrip() {
        let values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0];
        for &v in &values {
            let log = linear_to_acescc(v);
            let back = acescc_to_linear(log);
            assert!(
                (v - back).abs() / v.max(1e-6) < 0.01,
                "ACEScc roundtrip failed at {v}: got {back}"
            );
        }
    }

    #[test]
    fn mat3_identity() {
        let id: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (r, g, b) = mat3_mul(&id, 0.5, 0.3, 0.1);
        assert!((r - 0.5).abs() < 1e-7);
        assert!((g - 0.3).abs() < 1e-7);
        assert!((b - 0.1).abs() < 1e-7);
    }

    #[test]
    fn srgb_acescg_roundtrip() {
        let mut pixels = vec![0.5f32, 0.3, 0.1, 1.0];
        let orig = pixels.clone();
        apply_matrix(&mut pixels, &srgb_to_acescg_matrix());
        apply_matrix(&mut pixels, &acescg_to_srgb_matrix());
        for i in 0..3 {
            assert!(
                (pixels[i] - orig[i]).abs() < 1e-4,
                "sRGB↔ACEScg roundtrip ch{i}: {:.6} vs {:.6}",
                pixels[i],
                orig[i]
            );
        }
    }

    #[test]
    fn convert_color_space_identity() {
        let mut pixels = vec![0.5, 0.3, 0.1, 1.0];
        let orig = pixels.clone();
        convert_color_space(&mut pixels, ColorSpace::Linear, ColorSpace::Linear);
        assert_eq!(pixels, orig);
    }

    #[test]
    fn convert_srgb_to_linear_and_back() {
        let mut pixels = vec![0.5, 0.3, 0.8, 1.0];
        let orig = pixels.clone();
        convert_color_space(&mut pixels, ColorSpace::Srgb, ColorSpace::Linear);
        // Linear values should be different (darker)
        assert!(pixels[0] < orig[0]);
        convert_color_space(&mut pixels, ColorSpace::Linear, ColorSpace::Srgb);
        for i in 0..3 {
            assert!(
                (pixels[i] - orig[i]).abs() < 1e-5,
                "sRGB roundtrip ch{i}: {:.6} vs {:.6}",
                pixels[i],
                orig[i]
            );
        }
    }

    #[test]
    fn full_aces_roundtrip() {
        // sRGB → Linear → ACEScg → ACEScct → ACEScg → Linear → sRGB
        let mut pixels = vec![0.6, 0.4, 0.2, 1.0];
        let orig = pixels.clone();

        convert_color_space(&mut pixels, ColorSpace::Srgb, ColorSpace::Linear);
        convert_color_space(&mut pixels, ColorSpace::Linear, ColorSpace::AcesCg);
        convert_color_space(&mut pixels, ColorSpace::AcesCg, ColorSpace::AcesCct);
        convert_color_space(&mut pixels, ColorSpace::AcesCct, ColorSpace::AcesCg);
        convert_color_space(&mut pixels, ColorSpace::AcesCg, ColorSpace::Linear);
        convert_color_space(&mut pixels, ColorSpace::Linear, ColorSpace::Srgb);

        for i in 0..3 {
            assert!(
                (pixels[i] - orig[i]).abs() < 0.001,
                "ACES roundtrip ch{i}: {:.6} vs {:.6}",
                pixels[i],
                orig[i]
            );
        }
    }

    #[test]
    fn hdr_values_survive_conversion() {
        let mut pixels = vec![5.0, -0.5, 100.0, 1.0];
        convert_color_space(&mut pixels, ColorSpace::Linear, ColorSpace::AcesCg);
        // Values should not be clamped
        assert!(pixels.iter().any(|&v| v.abs() > 1.0), "HDR should survive");
        convert_color_space(&mut pixels, ColorSpace::AcesCg, ColorSpace::Linear);
        assert!((pixels[0] - 5.0).abs() < 0.01);
        assert!((pixels[2] - 100.0).abs() < 0.1);
    }

    #[test]
    fn f32_to_srgb_rgba8_roundtrip() {
        let input = vec![0.5f32, 0.3, 0.1, 0.8]; // linear f32
        let u8_pixels = f32_linear_to_srgb_rgba8(&input);
        let back = srgb_rgba8_to_f32_linear(&u8_pixels);
        // u8 quantization limits precision to ~1/255
        for i in 0..3 {
            assert!(
                (input[i] - back[i]).abs() < 0.005,
                "ch{i}: {:.4} vs {:.4}",
                input[i],
                back[i]
            );
        }
        assert!((input[3] - back[3]).abs() < 0.005); // alpha
    }

    #[test]
    fn f32_to_srgb_rgb8_drops_alpha() {
        let input = vec![0.5, 0.3, 0.1, 0.7];
        let rgb = f32_linear_to_srgb_rgb8(&input);
        assert_eq!(rgb.len(), 3); // no alpha
        let back = srgb_rgb8_to_f32_linear(&rgb);
        assert_eq!(back.len(), 4); // alpha added as 1.0
        assert!((back[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn codec_conversion_preserves_black_white() {
        let black = vec![0.0f32, 0.0, 0.0, 1.0];
        let white = vec![1.0f32, 1.0, 1.0, 1.0];
        let b8 = f32_linear_to_srgb_rgba8(&black);
        let w8 = f32_linear_to_srgb_rgba8(&white);
        assert_eq!(b8, vec![0, 0, 0, 255]);
        assert_eq!(w8, vec![255, 255, 255, 255]);
    }

    // ── OKLab Tests ──

    #[test]
    fn oklab_roundtrip_neutral_gray() {
        let (l, a, b) = linear_srgb_to_oklab(0.5, 0.5, 0.5);
        let (r, g, b2) = oklab_to_linear_srgb(l, a, b);
        assert!((r - 0.5).abs() < 1e-5, "r={r}");
        assert!((g - 0.5).abs() < 1e-5, "g={g}");
        assert!((b2 - 0.5).abs() < 1e-5, "b={b2}");
    }

    #[test]
    fn oklab_roundtrip_primary_colors() {
        for (r, g, b) in [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
        ] {
            let (ol, oa, ob) = linear_srgb_to_oklab(r, g, b);
            let (r2, g2, b2) = oklab_to_linear_srgb(ol, oa, ob);
            assert!((r - r2).abs() < 1e-4, "r roundtrip: {r} -> {r2}");
            assert!((g - g2).abs() < 1e-4, "g roundtrip: {g} -> {g2}");
            assert!((b - b2).abs() < 1e-4, "b roundtrip: {b} -> {b2}");
        }
    }

    #[test]
    fn oklab_black_is_zero_lightness() {
        let (l, a, b) = linear_srgb_to_oklab(0.0, 0.0, 0.0);
        assert!(l.abs() < 1e-6, "black L={l}");
        assert!(a.abs() < 1e-6, "black a={a}");
        assert!(b.abs() < 1e-6, "black b={b}");
    }

    #[test]
    fn oklab_white_is_unit_lightness() {
        let (l, a, b) = linear_srgb_to_oklab(1.0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-4, "white L={l}");
        assert!(a.abs() < 1e-3, "white a={a}");
        assert!(b.abs() < 1e-3, "white b={b}");
    }

    #[test]
    fn oklch_roundtrip() {
        let (l, a, b) = linear_srgb_to_oklab(0.8, 0.2, 0.4);
        let (l2, c, h) = oklab_to_oklch(l, a, b);
        assert_eq!(l, l2);
        assert!(c > 0.0, "non-gray should have chroma");
        let (l3, a2, b2) = oklch_to_oklab(l2, c, h);
        assert!((l - l3).abs() < 1e-6);
        assert!((a - a2).abs() < 1e-6);
        assert!((b - b2).abs() < 1e-6);
    }

    #[test]
    fn oklch_neutral_has_zero_chroma() {
        let (l, a, b) = linear_srgb_to_oklab(0.5, 0.5, 0.5);
        let (_, c, _) = oklab_to_oklch(l, a, b);
        assert!(
            c < 1e-4,
            "neutral gray should have near-zero chroma, got {c}"
        );
    }

    // ── CIE Colorimetry Tests ──

    #[test]
    fn d65_chromaticity() {
        let (x, y) = cie_d_illuminant_xy(6500.0);
        assert!((x - 0.31272).abs() < 0.001, "D65 x: {x}");
        assert!((y - 0.32903).abs() < 0.001, "D65 y: {y}");
    }

    #[test]
    fn cat16_d65_self_adaptation_is_identity() {
        let d65 = cie_d_illuminant_xy(6500.0);
        let m = cat16_adaptation_matrix(d65, d65);
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!(
                (m[i] - id[i]).abs() < 1e-5,
                "identity m[{i}]: {} vs {}",
                m[i],
                id[i]
            );
        }
    }

    #[test]
    fn cat16_warm_cool_shift() {
        let d65 = cie_d_illuminant_xy(6500.0);
        let warm = cie_d_illuminant_xy(3200.0);
        let m = cat16_adaptation_matrix(d65, warm);
        let (r, _g, b) = apply_3x3(&m, 1.0, 1.0, 1.0);
        assert!(r > 1.0, "warm should boost red: {r}");
        assert!(b < 1.0, "warm should reduce blue: {b}");
    }

    #[test]
    fn tint_zero_is_identity() {
        let xy = cie_d_illuminant_xy(6500.0);
        let shifted = tint_shift_xy(xy, 0.0);
        assert!((shifted.0 - xy.0).abs() < 1e-6);
        assert!((shifted.1 - xy.1).abs() < 1e-6);
    }

    #[test]
    fn cat16_inverse_roundtrip() {
        let product = mat3x3_mul(&CAT16, &CAT16_INV);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-8,
                    "CAT16*INV [{i}][{j}]: {} vs {expected}",
                    product[i][j]
                );
            }
        }
    }
}
