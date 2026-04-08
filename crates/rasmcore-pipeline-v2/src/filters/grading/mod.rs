//! Color grading filters — f32-only tone curves, CDL, LUT application, tone mapping.
//!
//! All CLUT-compatible filters implement `ClutOp` for fusion.
//! Film grain is spatial (position-dependent) and not CLUT-compatible.

pub mod asc_cdl;
pub mod curves;
pub mod film_grain;
pub mod hue_curves;
pub mod lift_gamma_gain;
pub mod lut_apply;
pub mod parsers;
pub mod split_toning;
pub mod tonemap;

pub use asc_cdl::AscCdl;
pub use curves::{CurvesMaster, CurvesRed, CurvesGreen, CurvesBlue};
pub use film_grain::FilmGrain;
pub use hue_curves::{HueVsSat, HueVsLum, LumVsSat, SatVsSat};
pub use lift_gamma_gain::LiftGammaGain;
pub use lut_apply::{ApplyCubeLut, ApplyHaldLut};
pub use parsers::{parse_cube_lut, parse_hald_lut};
pub use split_toning::SplitToning;
pub use tonemap::{TonemapReinhard, TonemapDrago, TonemapFilmic};

use crate::registry::{ParamDescriptor, ParamType};

// ─── Monotone Cubic Hermite Spline ─────────────────────────────────────────

/// Build a f32 LUT from control points using monotone cubic Hermite interpolation.
/// `lut_size` entries, indexed by normalized [0,1] -> [0, lut_size-1].
pub(crate) fn build_curve_lut_f32(points: &[(f32, f32)], lut_size: usize) -> Vec<f32> {
    let mut lut = vec![0.0f32; lut_size];
    if points.len() < 2 {
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as f32 / (lut_size - 1).max(1) as f32;
        }
        return lut;
    }
    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let tangents = monotone_tangents(&pts);

    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f32 / (lut_size - 1).max(1) as f32;
        *entry = eval_hermite(&pts, &tangents, x);
    }
    lut
}

/// Compute Fritsch-Carlson monotone tangents for sorted control points.
fn monotone_tangents(pts: &[(f32, f32)]) -> Vec<f32> {
    let n = pts.len();
    let mut m = vec![0.0f32; n];
    if n < 2 {
        return m;
    }
    if n == 2 {
        let slope = (pts[1].1 - pts[0].1) / (pts[1].0 - pts[0].0).max(1e-6);
        m[0] = slope;
        m[1] = slope;
        return m;
    }
    let mut deltas = vec![0.0f32; n - 1];
    for i in 0..n - 1 {
        let dx = (pts[i + 1].0 - pts[i].0).max(1e-6);
        deltas[i] = (pts[i + 1].1 - pts[i].1) / dx;
    }
    m[0] = deltas[0];
    m[n - 1] = deltas[n - 2];
    for i in 1..n - 1 {
        m[i] = (deltas[i - 1] + deltas[i]) * 0.5;
    }
    // Fritsch-Carlson monotonicity constraint
    for i in 0..n - 1 {
        if deltas[i].abs() < 1e-6 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / deltas[i];
            let beta = m[i + 1] / deltas[i];
            let tau = alpha * alpha + beta * beta;
            if tau > 9.0 {
                let t = 3.0 / tau.sqrt();
                m[i] = t * alpha * deltas[i];
                m[i + 1] = t * beta * deltas[i];
            }
        }
    }
    m
}

/// Evaluate monotone cubic Hermite spline at x.
fn eval_hermite(pts: &[(f32, f32)], tangents: &[f32], x: f32) -> f32 {
    let n = pts.len();
    if n == 0 {
        return x;
    }
    if x <= pts[0].0 {
        return pts[0].1;
    }
    if x >= pts[n - 1].0 {
        return pts[n - 1].1;
    }
    // Find segment via binary search
    let seg = match pts.binary_search_by(|p| p.0.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(idx) => return pts[idx].1,
        Err(idx) => {
            if idx == 0 {
                return pts[0].1;
            }
            idx - 1
        }
    };
    let x0 = pts[seg].0;
    let x1 = pts[seg + 1].0;
    let y0 = pts[seg].1;
    let y1 = pts[seg + 1].1;
    let h = (x1 - x0).max(1e-6);
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * tangents[seg] + h01 * y1 + h11 * h * tangents[seg + 1]
}

// ─── Helper: build curve from shadow/highlight params ──────────────────────

/// Generate control points from simplified shadow/midtone/highlight params.
/// Each param bends the curve: positive = brighten that range, negative = darken.
pub(crate) fn curve_from_params(shadows: f32, midtones: f32, highlights: f32) -> Vec<(f32, f32)> {
    vec![
        (0.0, 0.0),
        (0.25, (0.25 + shadows * 0.15).clamp(0.0, 1.0)),
        (0.5, (0.5 + midtones * 0.2).clamp(0.0, 1.0)),
        (0.75, (0.75 + highlights * 0.15).clamp(0.0, 1.0)),
        (1.0, 1.0),
    ]
}

/// Generate hue-indexed curve from a center/amount/width parameterization.
pub(crate) fn hue_curve_from_params(center: f32, amount: f32, width: f32) -> Vec<(f32, f32)> {
    // Default identity: all points at y=1 (no change)
    let n = 12;
    (0..=n).map(|i| {
        let hue = i as f32 / n as f32;
        let dist = ((hue - center / 360.0).abs()).min(1.0 - (hue - center / 360.0).abs());
        let influence = (-dist * dist / (2.0 * (width / 360.0).max(0.01).powi(2))).exp();
        (hue, (1.0 + amount * influence).clamp(0.0, 2.0))
    }).collect()
}

// ─── Param descriptor arrays ───────────────────────────────────────────────

macro_rules! pd_f32 {
    ($name:expr, $min:expr, $max:expr, $step:expr, $default:expr) => {
        ParamDescriptor {
            name: $name, value_type: ParamType::F32,
            min: Some($min), max: Some($max), step: Some($step), default: Some($default),
            hint: None, description: "", constraints: &[],
        }
    };
}

pub(crate) static CURVE_PARAMS: [ParamDescriptor; 3] = [
    pd_f32!("shadows", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("midtones", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("highlights", -1.0, 1.0, 0.05, 0.0),
];

pub(crate) static HUE_CURVE_PARAMS: [ParamDescriptor; 3] = [
    pd_f32!("center", 0.0, 360.0, 5.0, 0.0),
    pd_f32!("amount", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("width", 10.0, 180.0, 5.0, 60.0),
];

pub(crate) static NORM_CURVE_PARAMS: [ParamDescriptor; 3] = [
    pd_f32!("shadows", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("midtones", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("highlights", -1.0, 1.0, 0.05, 0.0),
];

pub(crate) static ASC_CDL_PARAMS: [ParamDescriptor; 10] = [
    pd_f32!("slope_r", 0.0, 4.0, 0.05, 1.0), pd_f32!("slope_g", 0.0, 4.0, 0.05, 1.0), pd_f32!("slope_b", 0.0, 4.0, 0.05, 1.0),
    pd_f32!("offset_r", -1.0, 1.0, 0.02, 0.0), pd_f32!("offset_g", -1.0, 1.0, 0.02, 0.0), pd_f32!("offset_b", -1.0, 1.0, 0.02, 0.0),
    pd_f32!("power_r", 0.1, 4.0, 0.05, 1.0), pd_f32!("power_g", 0.1, 4.0, 0.05, 1.0), pd_f32!("power_b", 0.1, 4.0, 0.05, 1.0),
    pd_f32!("saturation", 0.0, 4.0, 0.05, 1.0),
];

pub(crate) static SPLIT_TONING_PARAMS: [ParamDescriptor; 4] = [
    pd_f32!("shadow_hue", 0.0, 360.0, 5.0, 220.0),
    pd_f32!("highlight_hue", 0.0, 360.0, 5.0, 40.0),
    pd_f32!("shadow_strength", 0.0, 1.0, 0.05, 0.0),
    pd_f32!("highlight_strength", 0.0, 1.0, 0.05, 0.0),
];

pub(crate) static LIFT_GAMMA_GAIN_PARAMS: [ParamDescriptor; 9] = [
    pd_f32!("lift_r", -0.5, 0.5, 0.02, 0.0), pd_f32!("lift_g", -0.5, 0.5, 0.02, 0.0), pd_f32!("lift_b", -0.5, 0.5, 0.02, 0.0),
    pd_f32!("gamma_r", 0.1, 4.0, 0.05, 1.0), pd_f32!("gamma_g", 0.1, 4.0, 0.05, 1.0), pd_f32!("gamma_b", 0.1, 4.0, 0.05, 1.0),
    pd_f32!("gain_r", 0.0, 4.0, 0.05, 1.0), pd_f32!("gain_g", 0.0, 4.0, 0.05, 1.0), pd_f32!("gain_b", 0.0, 4.0, 0.05, 1.0),
];

// ─── Helper: HSL for split toning ──────────────────────────────────────────

pub(crate) fn hsl_to_rgb_simple(h: f32, s: f32, l: f32) -> [f32; 3] {
    if s < 1e-6 { return [l, l, l]; }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let h = h / 360.0;
    let hue_to_rgb = |t: f32| -> f32 {
        let t = ((t % 1.0) + 1.0) % 1.0;
        if t < 1.0 / 6.0 { p + (q - p) * 6.0 * t }
        else if t < 0.5 { q }
        else if t < 2.0 / 3.0 { p + (q - p) * (2.0 / 3.0 - t) * 6.0 }
        else { p }
    };
    [hue_to_rgb(h + 1.0 / 3.0), hue_to_rgb(h), hue_to_rgb(h - 1.0 / 3.0)]
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::Clut3D;
    use crate::ops::Filter;
    use super::super::color::ClutOp;

    fn test_pixel(r: f32, g: f32, b: f32) -> Vec<f32> {
        vec![r, g, b, 1.0]
    }

    fn assert_rgb_close(actual: &[f32], expected: (f32, f32, f32), tol: f32, label: &str) {
        assert!(
            (actual[0] - expected.0).abs() < tol
                && (actual[1] - expected.1).abs() < tol
                && (actual[2] - expected.2).abs() < tol,
            "{label}: expected ({:.4}, {:.4}, {:.4}), got ({:.4}, {:.4}, {:.4})",
            expected.0,
            expected.1,
            expected.2,
            actual[0],
            actual[1],
            actual[2]
        );
    }

    // ── Spline tests ──

    #[test]
    fn curve_identity() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0)];
        let lut = build_curve_lut_f32(&pts, 256);
        for i in 0..256 {
            let expected = i as f32 / 255.0;
            assert!(
                (lut[i] - expected).abs() < 0.01,
                "identity curve failed at {i}: got {:.4}, expected {expected:.4}",
                lut[i]
            );
        }
    }

    #[test]
    fn curve_invert() {
        let pts = vec![(0.0, 1.0), (1.0, 0.0)];
        let lut = build_curve_lut_f32(&pts, 256);
        assert!((lut[0] - 1.0).abs() < 0.01);
        assert!((lut[255] - 0.0).abs() < 0.01);
        assert!((lut[128] - 0.5).abs() < 0.02);
    }

    #[test]
    fn curve_s_shaped() {
        let pts = vec![(0.0, 0.0), (0.25, 0.1), (0.75, 0.9), (1.0, 1.0)];
        let lut = build_curve_lut_f32(&pts, 256);
        // Shadows should be darker, highlights brighter than linear
        assert!(lut[64] < 64.0 / 255.0, "shadows should be compressed");
        assert!(lut[192] > 192.0 / 255.0, "highlights should be expanded");
    }

    // ── Curves filters ──

    #[test]
    fn curves_master_identity() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = CurvesMaster {
            points: vec![(0.0, 0.0), (1.0, 1.0)],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 0.02, "curves identity");
    }

    #[test]
    fn curves_red_only_affects_red() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = CurvesRed {
            points: vec![(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] > 0.5, "Red should be boosted");
        assert!((out[1] - 0.5).abs() < 0.01, "Green should be unchanged");
        assert!((out[2] - 0.5).abs() < 0.01, "Blue should be unchanged");
    }

    #[test]
    fn curves_master_clut_matches_compute() {
        let pts = vec![(0.0, 0.0), (0.25, 0.1), (0.75, 0.9), (1.0, 1.0)];
        let f = CurvesMaster { points: pts };
        let input = test_pixel(0.4, 0.6, 0.2);
        let computed = f.compute(&input, 1, 1).unwrap();
        let clut = ClutOp::build_clut(&f);
        let (cr, cg, cb) = clut.sample(0.4, 0.6, 0.2);
        assert!(
            (computed[0] - cr).abs() < 0.05
                && (computed[1] - cg).abs() < 0.05
                && (computed[2] - cb).abs() < 0.05,
            "CLUT mismatch"
        );
    }

    // ── ASC CDL ──

    #[test]
    fn asc_cdl_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = AscCdl {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 1e-5, "CDL identity");
    }

    #[test]
    fn asc_cdl_slope_doubles() {
        let input = test_pixel(0.3, 0.3, 0.3);
        let f = AscCdl {
            slope: [2.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.6, 0.6, 0.6), 1e-5, "CDL slope 2x");
    }

    #[test]
    fn asc_cdl_saturation() {
        let input = test_pixel(0.8, 0.2, 0.4);
        let f = AscCdl {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 0.0, // desaturate completely
        };
        let out = f.compute(&input, 1, 1).unwrap();
        // All channels should equal luma
        assert!(
            (out[0] - out[1]).abs() < 0.01 && (out[1] - out[2]).abs() < 0.01,
            "CDL sat=0 should produce grayscale"
        );
    }

    // ── Lift/Gamma/Gain ──

    #[test]
    fn lgg_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = LiftGammaGain {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 0.01, "LGG identity");
    }

    #[test]
    fn lgg_gain_doubles() {
        let input = test_pixel(0.3, 0.3, 0.3);
        let f = LiftGammaGain {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [2.0; 3],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.6, 0.6, 0.6), 0.01, "LGG gain 2x");
    }

    // ── Split Toning ──

    #[test]
    fn split_toning_zero_strength_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = SplitToning {
            shadow_color: [0.0, 0.0, 1.0],
            highlight_color: [1.0, 0.0, 0.0],
            balance: 0.0,
            strength: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 1e-5, "split toning 0 strength");
    }

    #[test]
    fn split_toning_dark_gets_shadow_color() {
        let input = test_pixel(0.1, 0.1, 0.1); // dark pixel
        let f = SplitToning {
            shadow_color: [0.0, 0.0, 1.0], // blue shadows
            highlight_color: [1.0, 0.0, 0.0],
            balance: 0.0,
            strength: 1.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[2] > out[0], "Dark pixel should be tinted blue");
    }

    // ── Hue-based curves ──

    #[test]
    fn hue_vs_sat_neutral_curve() {
        let input = test_pixel(0.8, 0.2, 0.4);
        let f = HueVsSat {
            points: vec![(0.0, 0.5), (1.0, 0.5)], // neutral: mult=1.0
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.8, 0.2, 0.4), 0.02, "hue_vs_sat neutral");
    }

    #[test]
    fn lum_vs_sat_neutral() {
        let input = test_pixel(0.5, 0.3, 0.7);
        let f = LumVsSat {
            points: vec![(0.0, 0.5), (1.0, 0.5)],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.3, 0.7), 0.02, "lum_vs_sat neutral");
    }

    // ── Tone Mapping ──

    #[test]
    fn reinhard_maps_midtone() {
        let input = test_pixel(1.0, 1.0, 1.0);
        let f = TonemapReinhard;
        let out = f.compute(&input, 1, 1).unwrap();
        // 1.0 / (1 + 1.0) = 0.5
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 1e-5, "reinhard at 1.0");
    }

    #[test]
    fn reinhard_preserves_black() {
        let input = test_pixel(0.0, 0.0, 0.0);
        let f = TonemapReinhard;
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.0, 0.0, 0.0), 1e-5, "reinhard at 0.0");
    }

    #[test]
    fn reinhard_clut_matches_compute() {
        let f = TonemapReinhard;
        let input = test_pixel(0.6, 0.3, 0.8);
        let computed = f.compute(&input, 1, 1).unwrap();
        let clut = ClutOp::build_clut(&f);
        let (cr, cg, cb) = clut.sample(0.6, 0.3, 0.8);
        assert!(
            (computed[0] - cr).abs() < 0.05
                && (computed[1] - cg).abs() < 0.05
                && (computed[2] - cb).abs() < 0.05,
            "Reinhard CLUT mismatch"
        );
    }

    #[test]
    fn filmic_aces_reasonable_output() {
        let f = TonemapFilmic::default();
        let input = test_pixel(0.5, 0.5, 0.5);
        let out = f.compute(&input, 1, 1).unwrap();
        // Should produce values in (0, 1)
        assert!(out[0] > 0.0 && out[0] < 1.0, "Filmic should produce reasonable values");
    }

    #[test]
    fn drago_maps_hdr() {
        let f = TonemapDrago {
            l_max: 10.0,
            bias: 0.85,
        };
        let input = test_pixel(0.5, 0.5, 0.5);
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] > 0.0 && out[0] <= 1.0, "Drago should produce valid range");
    }

    // ── LUT Application ──

    #[test]
    fn apply_cube_lut_identity() {
        let clut = Clut3D::identity(17);
        let f = ApplyCubeLut { clut };
        let input = test_pixel(0.3, 0.6, 0.9);
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.6, 0.9), 0.02, "cube lut identity");
    }

    #[test]
    fn parse_cube_lut_minimal() {
        let cube_text = "\
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";
        let clut = parse_cube_lut(cube_text).unwrap();
        assert_eq!(clut.grid_size, 2);
        // Identity LUT: sample at corners
        let (r, g, b) = clut.sample(0.0, 0.0, 0.0);
        assert!((r).abs() < 0.01 && (g).abs() < 0.01 && (b).abs() < 0.01);
        let (r, g, b) = clut.sample(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01 && (g - 1.0).abs() < 0.01 && (b - 1.0).abs() < 0.01);
    }

    // ── 1D Cube LUT ──

    #[test]
    fn parse_cube_1d_lut() {
        // 1D LUT with 4 entries: simple gamma-like curve
        let cube_text = "\
TITLE \"Test 1D\"
LUT_1D_SIZE 4
0.0 0.0 0.0
0.3 0.3 0.3
0.7 0.7 0.7
1.0 1.0 1.0
";
        let clut = parse_cube_lut(cube_text).unwrap();
        // 1D LUT with 4 entries gets clamped to min(4, 65) = 4 grid size
        assert!(clut.grid_size <= 65);
        // Identity-ish: corners should map to themselves
        let (r, g, b) = clut.sample(0.0, 0.0, 0.0);
        assert!((r).abs() < 0.01 && (g).abs() < 0.01 && (b).abs() < 0.01, "black maps to black");
        let (r, g, b) = clut.sample(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01 && (g - 1.0).abs() < 0.01 && (b - 1.0).abs() < 0.01, "white maps to white");
    }

    #[test]
    fn parse_cube_1d_lut_with_domain() {
        // 1D LUT with custom domain
        let cube_text = "\
LUT_1D_SIZE 3
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 2.0 2.0 2.0
0.0 0.0 0.0
1.0 1.0 1.0
2.0 2.0 2.0
";
        let clut = parse_cube_lut(cube_text).unwrap();
        // After domain normalization: values should be 0.0, 0.5, 1.0
        let (r, _, _) = clut.sample(0.5, 0.0, 0.0);
        assert!((r - 0.5).abs() < 0.05, "mid-domain should map to 0.5, got {r}");
    }

    #[test]
    fn parse_cube_1d_channel_independent() {
        // 1D LUT with different per-channel curves
        let cube_text = "\
LUT_1D_SIZE 3
0.0 0.0 0.0
1.0 0.5 0.25
1.0 1.0 1.0
";
        let clut = parse_cube_lut(cube_text).unwrap();
        // At input (0.5, 0.5, 0.5), each channel interpolates independently
        let (r, g, b) = clut.sample(0.5, 0.5, 0.5);
        // R channel: lerp(0->1 then 1->1) at t=0.5 -> 1.0
        // G channel: lerp(0->0.5 then 0.5->1) at t=0.5 -> 0.5
        // B channel: lerp(0->0.25 then 0.25->1) at t=0.5 -> 0.25
        // But since it's 3D CLUT conversion, the independence comes from
        // the from_fn closure applying each 1D independently
        assert!(r > g, "R should be brighter than G at midtone, r={r}, g={g}");
        assert!(g > b, "G should be brighter than B at midtone, g={g}, b={b}");
    }

    // ── Hald CLUT ──

    #[test]
    fn parse_hald_lut_identity_level2() {
        // Level 2: grid_size = 2^2 = 4, image = 2^3 x 2^3 = 8x8 = 64 pixels
        let level: usize = 2;
        let grid = level * level; // 4
        let dim = level * level * level; // 8
        let total = dim * dim; // 64 pixels

        // Build identity Hald image: pixel i maps to (r, g, b) where
        // r = i % grid, g = (i / grid) % grid, b = i / (grid * grid)
        let mut pixels = Vec::with_capacity(total * 4);
        for i in 0..total {
            let r = (i % grid) as f32 / (grid - 1) as f32;
            let g = ((i / grid) % grid) as f32 / (grid - 1) as f32;
            let b = (i / (grid * grid)) as f32 / (grid - 1) as f32;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(1.0);
        }

        let clut = parse_hald_lut(&pixels, dim as u32, dim as u32).unwrap();
        assert_eq!(clut.grid_size, grid as u32);

        // Identity: corners should pass through
        let (r, g, b) = clut.sample(0.0, 0.0, 0.0);
        assert!((r).abs() < 0.01 && (g).abs() < 0.01 && (b).abs() < 0.01, "black");
        let (r, g, b) = clut.sample(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01 && (g - 1.0).abs() < 0.01 && (b - 1.0).abs() < 0.01, "white");
    }

    #[test]
    fn parse_hald_lut_non_square_rejected() {
        let pixels = vec![0.0f32; 4 * 64];
        assert!(parse_hald_lut(&pixels, 16, 4).is_err());
    }

    #[test]
    fn parse_hald_lut_non_cube_dim_rejected() {
        // 10x10 is not a perfect cube
        let pixels = vec![0.0f32; 4 * 100];
        assert!(parse_hald_lut(&pixels, 10, 10).is_err());
    }

    // ── GPU capability ──

    #[test]
    fn cube_lut_has_gpu_shader() {
        use crate::ops::GpuFilter;
        let clut = Clut3D::identity(17);
        let f = ApplyCubeLut { clut };
        let shaders = f.gpu_shaders(100, 100);
        assert!(!shaders.is_empty(), "ApplyCubeLut should have GPU shader");
        assert!(!shaders[0].extra_buffers.is_empty(), "Should have CLUT extra buffer");
    }

    #[test]
    fn hald_lut_has_gpu_shader() {
        use crate::ops::GpuFilter;
        let clut = Clut3D::identity(4);
        let f = ApplyHaldLut { clut };
        let shaders = f.gpu_shaders(100, 100);
        assert!(!shaders.is_empty(), "ApplyHaldLut should have GPU shader");
        assert!(!shaders[0].extra_buffers.is_empty(), "Should have CLUT extra buffer");
    }

    #[test]
    fn cube_lut_gpu_params_layout() {
        use crate::ops::GpuFilter;
        let clut = Clut3D::identity(17);
        let f = ApplyCubeLut { clut };
        let params = f.params(200, 100);
        assert_eq!(params.len(), 16, "GPU params: width, height, grid_size, _pad = 16 bytes");
        let width = u32::from_le_bytes(params[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(params[4..8].try_into().unwrap());
        let grid = u32::from_le_bytes(params[8..12].try_into().unwrap());
        assert_eq!(width, 200);
        assert_eq!(height, 100);
        assert_eq!(grid, 17);
    }

    // ── Alpha preservation ──

    #[test]
    fn alpha_preserved_all_grading_filters() {
        let input = vec![0.3, 0.5, 0.7, 0.42];
        let filters: Vec<Box<dyn Filter>> = vec![
            Box::new(CurvesMaster {
                points: vec![(0.0, 0.0), (1.0, 1.0)],
            }),
            Box::new(AscCdl {
                slope: [1.2; 3],
                offset: [0.01; 3],
                power: [0.9; 3],
                saturation: 1.0,
            }),
            Box::new(LiftGammaGain {
                lift: [0.0; 3],
                gamma: [1.0; 3],
                gain: [1.0; 3],
            }),
            Box::new(SplitToning {
                shadow_color: [0.0, 0.0, 1.0],
                highlight_color: [1.0, 0.0, 0.0],
                balance: 0.0,
                strength: 0.5,
            }),
            Box::new(TonemapReinhard),
            Box::new(TonemapFilmic::default()),
            Box::new(TonemapDrago {
                l_max: 1.0,
                bias: 0.85,
            }),
        ];
        for (i, f) in filters.iter().enumerate() {
            let out = f.compute(&input, 1, 1).unwrap();
            assert_eq!(out[3], 0.42, "Filter {i} should preserve alpha");
        }
    }
}
