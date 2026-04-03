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
            r[i * 3 + j] = a[i * 3] * b[j]
                + a[i * 3 + 1] * b[3 + j]
                + a[i * 3 + 2] * b[6 + j];
        }
    }
    r
}

// ─── ACES Color Space Matrices ────────────────────────────────────────────────
// From aces-dev CTL: ACESlib.Utilities_Color.ctl

/// sRGB linear (D65) → XYZ (D65)
pub const SRGB_TO_XYZ_D65: Mat3 = [
    0.4123907993, 0.3575843394, 0.1804807884,
    0.2126390059, 0.7151686788, 0.0721923154,
    0.0193308187, 0.1191947798, 0.9505321522,
];

/// XYZ (D65) → sRGB linear (D65)
pub const XYZ_D65_TO_SRGB: Mat3 = [
     3.2409699419, -1.5373831776, -0.4986107603,
    -0.9692436363,  1.8759675015,  0.0415550574,
     0.0556300797, -0.2039769589,  1.0569715142,
];

/// Bradford chromatic adaptation: D65 → D60
pub const BRADFORD_D65_TO_D60: Mat3 = [
     1.01303,    0.00610531, -0.014971,
     0.00769823, 0.998165,   -0.00503203,
    -0.00284131, 0.00468516,  0.924507,
];

/// Bradford chromatic adaptation: D60 → D65
pub const BRADFORD_D60_TO_D65: Mat3 = [
     0.987224,   -0.00611327,  0.0159533,
    -0.00759836,  1.00186,     0.00533002,
     0.00307257, -0.00509596,  1.08168,
];

/// AP0 (ACES2065-1) → XYZ (D60)
pub const AP0_TO_XYZ_D60: Mat3 = [
    0.9525523959, 0.0000000000, 0.0000936786,
    0.3439664498, 0.7281660966, -0.0721325464,
    0.0000000000, 0.0000000000, 1.0088251844,
];

/// XYZ (D60) → AP0 (ACES2065-1)
pub const XYZ_D60_TO_AP0: Mat3 = [
     1.0498110175, 0.0000000000, -0.0000974845,
    -0.4959030231, 1.3733130458,  0.0982400361,
     0.0000000000, 0.0000000000,  0.9912520182,
];

/// AP1 (ACEScg) → XYZ (D60)
pub const AP1_TO_XYZ_D60: Mat3 = [
     0.6624541811, 0.1340042065, 0.1561876870,
     0.2722287168, 0.6740817658, 0.0536895174,
    -0.0055746495, 0.0040607335, 1.0103391003,
];

/// XYZ (D60) → AP1 (ACEScg)
pub const XYZ_D60_TO_AP1: Mat3 = [
     1.6410233797, -0.3248032942, -0.2364246952,
    -0.6636628587,  1.6153315917,  0.0167563477,
     0.0117218943, -0.0082844420,  0.9883948585,
];

// ─── Convenience: Pre-composed Conversion Matrices ────────────────────────────

/// sRGB linear (D65) → ACEScg (AP1, D60)
pub fn srgb_to_acescg_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| {
        mat3_compose(&XYZ_D60_TO_AP1, &mat3_compose(&BRADFORD_D65_TO_D60, &SRGB_TO_XYZ_D65))
    });
    *M
}

/// ACEScg (AP1, D60) → sRGB linear (D65)
pub fn acescg_to_srgb_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| {
        mat3_compose(&XYZ_D65_TO_SRGB, &mat3_compose(&BRADFORD_D60_TO_D65, &AP1_TO_XYZ_D60))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrip() {
        for i in 0..256 {
            let v = i as f32 / 255.0;
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!((v - back).abs() < 1e-5, "roundtrip failed at {v}: got {back}");
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
            assert!((v - back).abs() < 1e-5, "ACEScct roundtrip failed at {v}: got {back}");
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
}
