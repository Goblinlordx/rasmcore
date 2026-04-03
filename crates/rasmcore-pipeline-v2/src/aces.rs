//! ACES output transforms — RRT + ODTs.
//!
//! Implements the Academy Color Encoding System output transform chain:
//!   ACEScg (AP1 linear) → RRT → OCES → ODT → display-referred
//!
//! The RRT (Reference Rendering Transform) is a fixed transform that converts
//! scene-referred linear to display-referred via:
//!   1. Glow module — compensates for loss of perceived brightness in dark regions
//!   2. Red modifier — desaturates bright reds to prevent hue shifts
//!   3. Global tone curve — segmented spline mapping scene → display luminance
//!   4. Output to OCES (Output Color Encoding Specification)
//!
//! ODTs (Output Device Transforms) convert OCES to device-specific signals:
//!   - sRGB (D65, 100 nit peak)
//!   - Rec.709 (D65, 100 nit peak)
//!   - P3-D65 (D65, 48 nit cinema)
//!
//! Reference: aces-dev CTL (Academy S-2014-003, S-2016-001, TB-2014-012).

use crate::color_math::{mat3_mul, Mat3};

// ═══════════════════════════════════════════════════════════════════════════════
// RRT Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// RRT saturation factor (slight desaturation before tone curve).
const RRT_SAT_FACTOR: f32 = 0.96;

/// AP1 luminance weights.
const AP1_LUMA: [f32; 3] = [0.272_228_72, 0.674_081_77, 0.053_689_52];

/// ODT desaturation factor (post-tone curve for display).
const ODT_SAT_FACTOR: f32 = 0.93;

// ═══════════════════════════════════════════════════════════════════════════════
// RRT Components
// ═══════════════════════════════════════════════════════════════════════════════

/// Glow module — adds subtle brightness boost to dark regions.
fn glow_fwd(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let luma = AP1_LUMA[0] * r + AP1_LUMA[1] * g + AP1_LUMA[2] * b;
    let glow_gain = if luma <= 0.0 {
        0.05
    } else if luma < 2.0 * 0.18 {
        let s = luma / (2.0 * 0.18);
        0.05 * (1.0 - s * s)
    } else {
        0.0
    };
    (r * (1.0 + glow_gain), g * (1.0 + glow_gain), b * (1.0 + glow_gain))
}

/// Red modifier — desaturates bright reds to prevent hue shifts.
fn red_modifier_fwd(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let luma = AP1_LUMA[0] * r + AP1_LUMA[1] * g + AP1_LUMA[2] * b;
    let red_ratio = if luma > 0.001 { r / luma } else { 0.0 };
    let hue_weight = if red_ratio > 1.0 {
        let t = ((red_ratio - 1.0) / 2.0).min(1.0);
        t * t * (3.0 - 2.0 * t) // smoothstep
    } else {
        0.0
    };
    let brightness_factor = (luma * 2.0).min(1.0);
    let blend = hue_weight * brightness_factor * 0.2;
    (r * (1.0 - blend) + luma * blend, g, b)
}

/// RRT saturation adjustment (slight desaturation before tone curve).
fn rrt_saturation(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let luma = AP1_LUMA[0] * r + AP1_LUMA[1] * g + AP1_LUMA[2] * b;
    let f = RRT_SAT_FACTOR;
    let inv = 1.0 - f;
    (luma * inv + r * f, luma * inv + g * f, luma * inv + b * f)
}

/// RRT tone curve — S-curve mapping scene-referred to display-referred.
///
/// Uses the Narkowicz ACES fitted curve, a rational polynomial that closely
/// matches the full CTL segmented spline RRT output for the [0, ∞) range.
/// This is the same approximation used by Unreal Engine and many production tools.
///
/// The fit maps 18% gray (0.18) to approximately display mid-gray and compresses
/// highlights with a gentle shoulder. Monotonically increasing for all non-negative input.
///
/// Reference: Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve" (2015).
fn rrt_tone_curve(v: f32) -> f32 {
    // Rational polynomial fit to ACES RRT
    // f(x) = (x * (2.51x + 0.03)) / (x * (2.43x + 0.59) + 0.14)
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let num = v * (a * v + b);
    let den = v * (c * v + d) + e;
    (num / den).max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full RRT Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply the ACES Reference Rendering Transform to a single pixel (AP1 linear).
///
/// Input: ACEScg (AP1 linear, scene-referred).
/// Output: OCES (display-referred luminance values).
pub fn rrt_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = (r.max(0.0), g.max(0.0), b.max(0.0));
    let (r, g, b) = glow_fwd(r, g, b);
    let (r, g, b) = red_modifier_fwd(r, g, b);
    let (r, g, b) = rrt_saturation(r, g, b);
    let r = rrt_tone_curve(r);
    let g = rrt_tone_curve(g);
    let b = rrt_tone_curve(b);
    (r, g, b)
}

/// Apply RRT to f32 RGBA pixel buffer (in-place).
pub fn apply_rrt(pixels: &mut [f32]) {
    for chunk in pixels.chunks_exact_mut(4) {
        let (r, g, b) = rrt_pixel(chunk[0], chunk[1], chunk[2]);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ODT Matrices (AP1 → display gamut)
// ═══════════════════════════════════════════════════════════════════════════════

/// AP1 → sRGB linear (D65). Pre-composed: AP1→XYZ_D60 → CAT_D60→D65 → XYZ_D65→sRGB.
const AP1_TO_SRGB: Mat3 = [
    1.70505100, -0.62179212, -0.08325888,
    -0.13025700, 1.14080290, -0.01054589,
    -0.02400328, -0.12896886, 1.15297214,
];

/// AP1 → P3-D65 linear.
const AP1_TO_P3D65: Mat3 = [
    1.37792090, -0.30886400, -0.06905690,
    -0.06933190, 1.08229380, -0.01296190,
    -0.00215832, -0.04578580, 1.04794420,
];

// ═══════════════════════════════════════════════════════════════════════════════
// ODT Implementations
// ═══════════════════════════════════════════════════════════════════════════════

/// ODT desaturation (post-tone curve).
fn odt_saturation(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let luma = AP1_LUMA[0] * r + AP1_LUMA[1] * g + AP1_LUMA[2] * b;
    let f = ODT_SAT_FACTOR;
    let inv = 1.0 - f;
    (luma * inv + r * f, luma * inv + g * f, luma * inv + b * f)
}

/// sRGB ODT: post-RRT AP1 display-referred → sRGB display.
///
/// The RRT tone curve already maps to [0, ~1] display-referred.
/// ODT applies: desaturation → AP1→sRGB matrix → sRGB OETF.
pub fn odt_srgb_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = odt_saturation(r, g, b);
    let (r, g, b) = mat3_mul(&AP1_TO_SRGB, r, g, b);
    let r = crate::color_math::linear_to_srgb(r.clamp(0.0, 1.0));
    let g = crate::color_math::linear_to_srgb(g.clamp(0.0, 1.0));
    let b = crate::color_math::linear_to_srgb(b.clamp(0.0, 1.0));
    (r, g, b)
}

/// Rec.709 ODT: post-RRT → Rec.709 display (BT.1886 gamma 2.4).
pub fn odt_rec709_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = odt_saturation(r, g, b);
    let (r, g, b) = mat3_mul(&AP1_TO_SRGB, r, g, b); // same primaries as sRGB
    let r = r.clamp(0.0, 1.0).powf(1.0 / 2.4);
    let g = g.clamp(0.0, 1.0).powf(1.0 / 2.4);
    let b = b.clamp(0.0, 1.0).powf(1.0 / 2.4);
    (r, g, b)
}

/// P3-D65 ODT: post-RRT → P3-D65 display (gamma 2.6 cinema).
pub fn odt_p3d65_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = odt_saturation(r, g, b);
    let (r, g, b) = mat3_mul(&AP1_TO_P3D65, r, g, b);
    let r = r.clamp(0.0, 1.0).powf(1.0 / 2.6);
    let g = g.clamp(0.0, 1.0).powf(1.0 / 2.6);
    let b = b.clamp(0.0, 1.0).powf(1.0 / 2.6);
    (r, g, b)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Combined RRT+ODT
// ═══════════════════════════════════════════════════════════════════════════════

/// Full ACES output: RRT + sRGB ODT. Input: AP1 linear. Output: sRGB [0,1].
pub fn aces_rrt_odt_srgb_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = rrt_pixel(r, g, b);
    odt_srgb_pixel(r, g, b)
}

/// Full ACES output: RRT + Rec.709 ODT.
pub fn aces_rrt_odt_rec709_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = rrt_pixel(r, g, b);
    odt_rec709_pixel(r, g, b)
}

/// Full ACES output: RRT + P3-D65 ODT.
pub fn aces_rrt_odt_p3d65_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = rrt_pixel(r, g, b);
    odt_p3d65_pixel(r, g, b)
}

/// Apply full ACES RRT+ODT to f32 RGBA pixel buffer (in-place).
pub fn apply_aces_output_transform(
    pixels: &mut [f32],
    odt_fn: fn(f32, f32, f32) -> (f32, f32, f32),
) {
    for chunk in pixels.chunks_exact_mut(4) {
        let (r, g, b) = odt_fn(chunk[0], chunk[1], chunk[2]);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrt_tone_curve_monotonic() {
        let mut prev = 0.0f32;
        // Test across a wide range of scene-referred values
        for i in 0..100 {
            let v = 0.00001 * 10.0f32.powf(i as f32 * 0.1);
            let y = rrt_tone_curve(v);
            assert!(
                y >= prev - 1e-6,
                "RRT spline not monotonic at v={v}: y={y} < prev={prev}"
            );
            prev = y;
        }
    }

    #[test]
    fn rrt_black_near_zero() {
        let (r, g, b) = rrt_pixel(0.0, 0.0, 0.0);
        assert!(r < 0.01, "black should map near zero, got {r}");
        assert!(g < 0.01);
        assert!(b < 0.01);
    }

    #[test]
    fn rrt_midgray_reasonable() {
        let y = rrt_tone_curve(0.18);
        assert!(y > 0.1, "18% gray mapped too low: {y}");
        assert!(y < 100.0, "18% gray mapped too high: {y}");
    }

    #[test]
    fn rrt_preserves_neutral_hue() {
        let (r, g, b) = rrt_pixel(0.18, 0.18, 0.18);
        assert!((r - g).abs() < 0.01, "neutral shifted: r={r} g={g}");
        assert!((g - b).abs() < 0.01, "neutral shifted: g={g} b={b}");
    }

    #[test]
    fn rrt_bright_values_compress() {
        // Super-white should compress (not blow up)
        let (r, _, _) = rrt_pixel(100.0, 100.0, 100.0);
        assert!(r < 100000.0, "RRT should compress highlights, got {r}");
        assert!(r > 1.0, "bright values should produce significant output");
    }

    #[test]
    fn glow_brightens_darks() {
        let (r, _, _) = glow_fwd(0.01, 0.01, 0.01);
        assert!(r > 0.01, "glow should brighten dark values");
    }

    #[test]
    fn glow_no_effect_on_bright() {
        let (r, g, b) = glow_fwd(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn red_modifier_reduces_saturated_red() {
        let (r_out, _, _) = red_modifier_fwd(5.0, 0.1, 0.1);
        assert!(r_out < 5.0, "should reduce bright saturated red");
    }

    #[test]
    fn red_modifier_preserves_neutral() {
        let (r, g, b) = red_modifier_fwd(0.5, 0.5, 0.5);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }

    // ─── ODTs ────────────────────────────────────────────────────────────

    #[test]
    fn odt_srgb_output_in_range() {
        let (r, g, b) = aces_rrt_odt_srgb_pixel(0.18, 0.18, 0.18);
        assert!(r >= 0.0 && r <= 1.0, "sRGB out of range: {r}");
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn odt_srgb_midgray_reasonable() {
        let (r, _, _) = aces_rrt_odt_srgb_pixel(0.18, 0.18, 0.18);
        assert!(r > 0.1, "mid-gray too dark: {r}");
        assert!(r < 0.9, "mid-gray too bright: {r}");
    }

    #[test]
    fn odt_rec709_output_in_range() {
        let (r, g, b) = aces_rrt_odt_rec709_pixel(0.18, 0.18, 0.18);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn odt_p3d65_output_in_range() {
        let (r, g, b) = aces_rrt_odt_p3d65_pixel(0.18, 0.18, 0.18);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn odt_srgb_bright_saturates() {
        let (r, _, _) = aces_rrt_odt_srgb_pixel(10.0, 10.0, 10.0);
        assert!(r > 0.9, "bright should saturate near 1.0, got {r}");
    }

    #[test]
    fn odt_preserves_neutral() {
        let (r, g, b) = aces_rrt_odt_srgb_pixel(0.18, 0.18, 0.18);
        assert!((r - g).abs() < 0.01, "neutral shifted: r={r} g={g}");
        assert!((g - b).abs() < 0.01, "neutral shifted: g={g} b={b}");
    }

    #[test]
    fn apply_rrt_preserves_alpha() {
        let mut pixels = vec![0.18, 0.18, 0.18, 0.7, 0.5, 0.3, 0.1, 0.3];
        apply_rrt(&mut pixels);
        assert!((pixels[3] - 0.7).abs() < 1e-6);
        assert!((pixels[7] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn apply_aces_output_preserves_alpha() {
        let mut pixels = vec![0.18, 0.18, 0.18, 1.0];
        apply_aces_output_transform(&mut pixels, aces_rrt_odt_srgb_pixel);
        assert!((pixels[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn aces_full_pipeline_black_to_white_monotonic() {
        let mut prev = 0.0f32;
        for i in 0..20 {
            let v = 0.001 * 2.0f32.powi(i);
            let (r, _, _) = aces_rrt_odt_srgb_pixel(v, v, v);
            assert!(
                r >= prev - 1e-4,
                "ACES pipeline not monotonic at v={v}: r={r} < prev={prev}"
            );
            prev = r;
        }
    }
}
