//! Professional color grading — lift/gamma/gain, ASC-CDL, split toning, curves.
//!
//! All operations work on normalized [0.0, 1.0] RGB values. They can be applied
//! directly to pixel buffers or baked into a [`ColorLut3D`] for O(1) per-pixel
//! evaluation via [`ColorLut3D::from_fn`].
//!
//! # Formulas
//!
//! - **ASC-CDL**: `out = clamp01((in * slope + offset) ^ power)` per ITU-R BT.1886
//! - **Lift/Gamma/Gain**: `out = gain * (in + lift * (1 - in)) ^ (1/gamma)`
//! - **Split Toning**: blend shadow_color at low luminance, highlight_color at high
//! - **Curves**: monotone cubic spline through control points → 256-entry LUT

mod asc_cdl;
mod curves;
mod lift_gamma_gain;
mod selective_color;
mod split_toning;
mod tonemap;

pub use asc_cdl::*;
pub use curves::*;
pub use lift_gamma_gain::*;
pub use selective_color::*;
pub use split_toning::*;
pub use tonemap::*;

use super::color_lut::ColorLut3D;
use super::error::ImageError;
#[cfg(test)]
use super::types::ColorSpace;
use super::types::{ImageInfo, PixelFormat};

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Apply an f32 RGB transform function to a u8 pixel buffer.
pub(crate) fn apply_rgb_transform(
    pixels: &[u8],
    info: &ImageInfo,
    f: impl Fn(f32, f32, f32) -> (f32, f32, f32),
) -> Result<Vec<u8>, ImageError> {
    let bpp = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "color grading requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let expected = info.width as usize * info.height as usize * bpp;
    if pixels.len() < expected {
        return Err(ImageError::InvalidParameters("pixel data too small".into()));
    }

    let mut out = vec![0u8; pixels.len()];
    for i in (0..expected).step_by(bpp) {
        let r = pixels[i] as f32 / 255.0;
        let g = pixels[i + 1] as f32 / 255.0;
        let b = pixels[i + 2] as f32 / 255.0;
        let (or, og, ob) = f(r, g, b);
        out[i] = (or * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i + 1] = (og * 255.0).round().clamp(0.0, 255.0) as u8;
        out[i + 2] = (ob * 255.0).round().clamp(0.0, 255.0) as u8;
        if bpp == 4 {
            out[i + 3] = pixels[i + 3]; // preserve alpha
        }
    }
    Ok(out)
}

// ─── HSL Conversion Helpers ──────────────────────────────────────────────

/// Convert normalized [0,1] RGB to HSL. H in [0,360), S and L in [0,1].
#[inline]
pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    let delta = max - min;
    if delta == 0.0 {
        return (0.0, 0.0, l);
    }
    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };
    let h = if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, l)
}

/// Convert HSL to normalized [0,1] RGB. H in [0,360), S and L in [0,1].
#[inline]
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    (r1 + m, g1 + m, b1 + m)
}

// ─── Spline Helpers ──────────────────────────────────────────────────────

/// Compute monotone cubic Hermite tangents (Fritsch-Carlson).
pub(crate) fn monotone_tangents(pts: &[(f32, f32)]) -> Vec<f32> {
    let n = pts.len();
    let mut m = vec![0.0f32; n];

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

/// Evaluate a monotone Hermite spline at position x.
pub(crate) fn eval_hermite(pts: &[(f32, f32)], m: &[f32], n: usize, x: f32) -> f32 {
    // Find segment
    let seg =
        match pts.binary_search_by(|p| p.0.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(idx) => return pts[idx].1.clamp(0.0, 1.0),
            Err(idx) => {
                if idx == 0 {
                    return pts[0].1.clamp(0.0, 1.0);
                }
                if idx >= n {
                    return pts[n - 1].1.clamp(0.0, 1.0);
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

    (h00 * y0 + h10 * h * m[seg] + h01 * y1 + h11 * h * m[seg + 1]).clamp(0.0, 1.0)
}

// ─── CLUT Baking ──────────────────────────────────────────────────────────

/// Bake an ASC-CDL transform into a 3D CLUT.
pub fn bake_asc_cdl(cdl: &AscCdl, grid_size: usize) -> ColorLut3D {
    ColorLut3D::from_fn(grid_size, |r, g, b| asc_cdl_pixel(r, g, b, cdl))
}

/// Bake lift/gamma/gain into a 3D CLUT.
pub fn bake_lift_gamma_gain(lgg: &LiftGammaGain, grid_size: usize) -> ColorLut3D {
    ColorLut3D::from_fn(grid_size, |r, g, b| lift_gamma_gain_pixel(r, g, b, lgg))
}

/// Bake split toning into a 3D CLUT.
pub fn bake_split_toning(st: &SplitToning, grid_size: usize) -> ColorLut3D {
    ColorLut3D::from_fn(grid_size, |r, g, b| split_toning_pixel(r, g, b, st))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_info() -> ImageInfo {
        ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    // ─── ASC-CDL tests ────────────────────────────────────────────────

    #[test]
    fn asc_cdl_identity() {
        let cdl = AscCdl::default();
        let (r, g, b) = asc_cdl_pixel(0.5, 0.3, 0.8, &cdl);
        assert!((r - 0.5).abs() < 1e-5);
        assert!((g - 0.3).abs() < 1e-5);
        assert!((b - 0.8).abs() < 1e-5);
    }

    #[test]
    fn asc_cdl_slope_doubles() {
        let cdl = AscCdl {
            slope: [2.0, 2.0, 2.0],
            ..Default::default()
        };
        let (r, _, _) = asc_cdl_pixel(0.3, 0.0, 0.0, &cdl);
        assert!((r - 0.6).abs() < 1e-5);
    }

    #[test]
    fn asc_cdl_offset_adds() {
        let cdl = AscCdl {
            offset: [0.1, 0.0, 0.0],
            ..Default::default()
        };
        let (r, _, _) = asc_cdl_pixel(0.5, 0.0, 0.0, &cdl);
        assert!((r - 0.6).abs() < 1e-5);
    }

    #[test]
    fn asc_cdl_power_gamma() {
        let cdl = AscCdl {
            power: [2.0, 1.0, 1.0],
            ..Default::default()
        };
        let (r, _, _) = asc_cdl_pixel(0.5, 0.0, 0.0, &cdl);
        assert!((r - 0.25).abs() < 1e-5); // 0.5^2 = 0.25
    }

    #[test]
    fn asc_cdl_clamps_output() {
        let cdl = AscCdl {
            slope: [3.0, 3.0, 3.0],
            offset: [0.5, 0.5, 0.5],
            ..Default::default()
        };
        let (r, g, b) = asc_cdl_pixel(1.0, 1.0, 1.0, &cdl);
        assert_eq!(r, 1.0);
        assert_eq!(g, 1.0);
        assert_eq!(b, 1.0);
    }

    #[test]
    fn asc_cdl_buffer_apply() {
        let px = vec![128u8, 128, 128];
        let info = test_info();
        let cdl = AscCdl::default();
        let result = asc_cdl(&px, &info, &cdl).unwrap();
        assert_eq!(result, px); // identity
    }

    // ─── Lift/Gamma/Gain tests ────────────────────────────────────────

    #[test]
    fn lgg_identity() {
        let lgg = LiftGammaGain::default();
        let (r, g, b) = lift_gamma_gain_pixel(0.5, 0.3, 0.8, &lgg);
        assert!((r - 0.5).abs() < 1e-5);
        assert!((g - 0.3).abs() < 1e-5);
        assert!((b - 0.8).abs() < 1e-5);
    }

    #[test]
    fn lgg_gain_doubles() {
        let lgg = LiftGammaGain {
            gain: [2.0, 1.0, 1.0],
            ..Default::default()
        };
        let (r, _, _) = lift_gamma_gain_pixel(0.4, 0.0, 0.0, &lgg);
        assert!((r - 0.8).abs() < 1e-5);
    }

    #[test]
    fn lgg_lift_raises_shadows() {
        let lgg = LiftGammaGain {
            lift: [0.2, 0.2, 0.2],
            ..Default::default()
        };
        // At input=0: output = lift * (1 - 0) = lift = 0.2
        let (r, _, _) = lift_gamma_gain_pixel(0.0, 0.0, 0.0, &lgg);
        assert!((r - 0.2).abs() < 1e-4);
    }

    #[test]
    fn lgg_gamma_adjusts_midtones() {
        let lgg = LiftGammaGain {
            gamma: [2.0, 2.0, 2.0],
            ..Default::default()
        };
        // gamma=2: out = in^(1/2) = sqrt(in)
        let (r, _, _) = lift_gamma_gain_pixel(0.25, 0.0, 0.0, &lgg);
        assert!((r - 0.5).abs() < 1e-4); // sqrt(0.25) = 0.5
    }

    // ─── Split Toning tests ───────────────────────────────────────────

    #[test]
    fn split_toning_dark_pixel_tints_toward_shadow() {
        let st = SplitToning {
            shadow_color: [0.0, 0.0, 1.0], // blue shadows
            highlight_color: [1.0, 1.0, 0.0],
            balance: 0.0,
            strength: 1.0,
        };
        // Very dark pixel should be tinted blue
        let (_, _, b) = split_toning_pixel(0.05, 0.05, 0.05, &st);
        assert!(b > 0.3, "dark pixel should have blue tint, got b={b}");
    }

    #[test]
    fn split_toning_bright_pixel_tints_toward_highlight() {
        let st = SplitToning {
            shadow_color: [0.0, 0.0, 1.0],
            highlight_color: [1.0, 0.8, 0.0], // warm highlights
            balance: 0.0,
            strength: 1.0,
        };
        let (r, _, _) = split_toning_pixel(0.9, 0.9, 0.9, &st);
        assert!(r > 0.9, "bright pixel should have warm tint, got r={r}");
    }

    // ─── Curves tests ─────────────────────────────────────────────────

    #[test]
    fn curves_identity() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let lut = build_curve_lut(&points);
        for i in 0..256 {
            assert_eq!(lut[i], i as u8, "identity curve at {i}");
        }
    }

    #[test]
    fn curves_invert() {
        let points = vec![(0.0, 1.0), (1.0, 0.0)];
        let lut = build_curve_lut(&points);
        assert_eq!(lut[0], 255);
        assert_eq!(lut[255], 0);
        assert!((lut[128] as i16 - 127).abs() <= 1);
    }

    #[test]
    fn curves_s_curve() {
        let points = vec![(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)];
        let lut = build_curve_lut(&points);
        // S-curve should darken shadows and brighten highlights
        assert!(lut[64] < 64, "S-curve should darken shadows");
        assert!(lut[192] > 192, "S-curve should brighten highlights");
        // Midpoint should be roughly preserved
        assert!((lut[128] as i16 - 128).abs() < 20);
    }

    #[test]
    fn curves_monotone() {
        let points = vec![(0.0, 0.0), (0.3, 0.2), (0.7, 0.9), (1.0, 1.0)];
        let lut = build_curve_lut(&points);
        // Should be monotonically non-decreasing
        for i in 1..256 {
            assert!(
                lut[i] >= lut[i - 1],
                "curve not monotone at {i}: {} < {}",
                lut[i],
                lut[i - 1]
            );
        }
    }

    // ─── CLUT Baking tests ────────────────────────────────────────────

    #[test]
    fn bake_asc_cdl_identity_is_identity_lut() {
        let cdl = AscCdl::default();
        let lut = bake_asc_cdl(&cdl, 17);
        // Check corners
        let (r, g, b) = lut.lookup(0.0, 0.0, 0.0);
        assert!(r.abs() < 0.01 && g.abs() < 0.01 && b.abs() < 0.01);
        let (r, g, b) = lut.lookup(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01 && (g - 1.0).abs() < 0.01 && (b - 1.0).abs() < 0.01);
    }

    #[test]
    fn bake_lgg_matches_direct() {
        let lgg = LiftGammaGain {
            lift: [0.1, 0.0, 0.0],
            gamma: [1.2, 1.0, 1.0],
            gain: [0.9, 1.0, 1.0],
            ..Default::default()
        };
        let lut = bake_lift_gamma_gain(&lgg, 33);
        // Compare LUT vs direct
        let (dr, dg, db) = lift_gamma_gain_pixel(0.5, 0.3, 0.8, &lgg);
        let (lr, lg, lb) = lut.lookup(0.5, 0.3, 0.8);
        assert!((dr - lr).abs() < 0.02, "r: direct={dr} lut={lr}");
        assert!((dg - lg).abs() < 0.02, "g: direct={dg} lut={lg}");
        assert!((db - lb).abs() < 0.02, "b: direct={db} lut={lb}");
    }

    // ─── Tone Mapping tests ───────────────────────────────────────────

    #[test]
    fn reinhard_identity_at_low_values() {
        let (r, _, _) = tonemap_reinhard_pixel(0.1, 0.1, 0.1);
        assert!((r - 0.0909).abs() < 0.01);
    }

    #[test]
    fn reinhard_compresses_highlights() {
        let (r, _, _) = tonemap_reinhard_pixel(1.0, 0.0, 0.0);
        assert!((r - 0.5).abs() < 0.01);
        let (r2, _, _) = tonemap_reinhard_pixel(10.0, 0.0, 0.0);
        assert!((r2 - 0.909).abs() < 0.01);
    }

    #[test]
    fn reinhard_buffer_apply() {
        let px = vec![128u8, 128, 128];
        let info = test_info();
        let result = tonemap_reinhard(&px, &info).unwrap();
        assert!((result[0] as i16 - 85).abs() <= 1);
    }

    #[test]
    fn drago_identity_at_default() {
        let params = DragoParams::default();
        let (r, _, _) = tonemap_drago_pixel(0.5, 0.0, 0.0, &params);
        assert!(r > 0.0 && r <= 1.0);
    }

    #[test]
    fn drago_zero_is_zero() {
        let params = DragoParams::default();
        let (r, _, _) = tonemap_drago_pixel(0.0, 0.0, 0.0, &params);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn filmic_aces_s_curve() {
        let params = FilmicParams::default();
        let (low, _, _) = tonemap_filmic_pixel(0.1, 0.0, 0.0, &params);
        let (mid, _, _) = tonemap_filmic_pixel(0.5, 0.0, 0.0, &params);
        let (high, _, _) = tonemap_filmic_pixel(1.0, 0.0, 0.0, &params);
        assert!(low < mid, "low={low} should be < mid={mid}");
        assert!(mid < high, "mid={mid} should be < high={high}");
        assert!(high < 1.0, "ACES should compress highlights: got {high}");
    }

    #[test]
    fn filmic_zero_is_near_zero() {
        let params = FilmicParams::default();
        let (r, _, _) = tonemap_filmic_pixel(0.0, 0.0, 0.0, &params);
        assert!(r.abs() < 0.01);
    }

    // ─── Film Grain tests ─────────────────────────────────────────────

    #[test]
    fn grain_zero_amount_is_identity() {
        let px = vec![128u8; 64 * 64 * 3];
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let params = FilmGrainParams {
            amount: 0.0,
            ..Default::default()
        };
        let result = film_grain(&px, &info, &params).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn grain_deterministic() {
        let px = vec![128u8; 32 * 32 * 3];
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let params = FilmGrainParams::default();
        let a = film_grain(&px, &info, &params).unwrap();
        let b = film_grain(&px, &info, &params).unwrap();
        assert_eq!(a, b, "grain must be deterministic for same seed");
    }

    #[test]
    fn grain_different_seeds_differ() {
        let px = vec![128u8; 32 * 32 * 3];
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let a = film_grain(
            &px,
            &info,
            &FilmGrainParams {
                seed: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let b = film_grain(
            &px,
            &info,
            &FilmGrainParams {
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn grain_modifies_midtones_more_than_extremes() {
        let dark = vec![10u8; 3];
        let mid = vec![128u8; 3];
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let params = FilmGrainParams {
            amount: 1.0,
            size: 1.0,
            ..Default::default()
        };
        let dark_out = film_grain(&dark, &info, &params).unwrap();
        let mid_out = film_grain(&mid, &info, &params).unwrap();

        let dark_diff: u32 = dark
            .iter()
            .zip(dark_out.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();
        let mid_diff: u32 = mid
            .iter()
            .zip(mid_out.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();

        assert!(
            mid_diff >= dark_diff,
            "midtones should have more grain: mid_diff={mid_diff}, dark_diff={dark_diff}"
        );
    }

    #[test]
    fn grain_mono_vs_color() {
        let px = vec![128u8; 16 * 16 * 3];
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let mono = film_grain(
            &px,
            &info,
            &FilmGrainParams {
                color: false,
                amount: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        let color = film_grain(
            &px,
            &info,
            &FilmGrainParams {
                color: true,
                amount: 0.5,
                ..Default::default()
            },
        )
        .unwrap();

        // Mono grain: R delta == G delta == B delta for each pixel
        let mut mono_uniform = true;
        for i in (0..mono.len()).step_by(3) {
            let dr = mono[i] as i16 - 128;
            let dg = mono[i + 1] as i16 - 128;
            let db = mono[i + 2] as i16 - 128;
            if dr != dg || dg != db {
                mono_uniform = false;
                break;
            }
        }
        assert!(
            mono_uniform,
            "mono grain should affect all channels equally"
        );

        // Color grain: channels should differ for some pixels
        let mut color_varied = false;
        for i in (0..color.len()).step_by(3) {
            let dr = color[i] as i16 - 128;
            let dg = color[i + 1] as i16 - 128;
            if dr != dg {
                color_varied = true;
                break;
            }
        }
        assert!(color_varied, "color grain should vary between channels");
    }

    // ─── Color Balance tests ──────────────────────────────────────────

    #[test]
    fn color_balance_identity() {
        let cb = ColorBalance::default();
        let (r, g, b) = color_balance_pixel(0.5, 0.3, 0.8, &cb);
        assert!((r - 0.5).abs() < 1e-4, "identity r: {r}");
        assert!((g - 0.3).abs() < 1e-4, "identity g: {g}");
        assert!((b - 0.8).abs() < 1e-4, "identity b: {b}");
    }

    #[test]
    fn color_balance_shadow_shift() {
        let cb = ColorBalance {
            shadow: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let (r, _, _) = color_balance_pixel(0.1, 0.1, 0.1, &cb);
        assert!(r > 0.15, "shadow red shift on dark pixel: r={r}");

        let dark_shift = {
            let cb_no_lum = ColorBalance {
                shadow: [1.0, 0.0, 0.0],
                preserve_luminosity: false,
                ..Default::default()
            };
            let (r_d, _, _) = color_balance_pixel(0.1, 0.1, 0.1, &cb_no_lum);
            let (r_b, _, _) = color_balance_pixel(0.9, 0.9, 0.9, &cb_no_lum);
            (r_d - 0.1, r_b - 0.9)
        };
        assert!(
            dark_shift.0.abs() > dark_shift.1.abs(),
            "shadows should affect dark more than bright: dark={}, bright={}",
            dark_shift.0,
            dark_shift.1
        );
    }

    #[test]
    fn color_balance_midtone_shift() {
        let cb = ColorBalance {
            midtone: [0.0, 0.5, 0.0],
            preserve_luminosity: false,
            ..Default::default()
        };
        let (_, g, _) = color_balance_pixel(0.5, 0.5, 0.5, &cb);
        assert!(g > 0.55, "midtone green shift: g={g}");
    }

    #[test]
    fn color_balance_preserve_luminosity() {
        let cb = ColorBalance {
            midtone: [0.5, 0.0, 0.0],
            preserve_luminosity: true,
            ..Default::default()
        };
        let (r, g, b) = color_balance_pixel(0.5, 0.5, 0.5, &cb);
        let luma_before = 0.2126 * 0.5 + 0.7152 * 0.5 + 0.0722 * 0.5;
        let luma_after = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        assert!(
            (luma_before - luma_after).abs() < 0.01,
            "luminosity should be preserved: before={luma_before}, after={luma_after}"
        );
    }

    #[test]
    fn color_balance_buffer_apply() {
        let px = vec![128u8, 128, 128];
        let info = test_info();
        let cb = ColorBalance::default();
        let result = color_balance(&px, &info, &cb).unwrap();
        assert_eq!(result, px, "identity should not change pixels");
    }

    // ─── HSL roundtrip tests ──────────────────────────────────────────────

    #[test]
    fn hsl_roundtrip() {
        for &(r, g, b) in &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.3, 0.8),
        ] {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let (r2, g2, b2) = hsl_to_rgb(h, s, l);
            assert!(
                (r - r2).abs() < 0.01 && (g - g2).abs() < 0.01 && (b - b2).abs() < 0.01,
                "HSL roundtrip failed for ({r},{g},{b}): got ({r2},{g2},{b2})"
            );
        }
    }

    // ─── Hue-vs-X curve grading tests ─────────────────────────────────

    #[test]
    fn hue_vs_sat_identity_curve() {
        let identity = build_hue_curve_lut(&[(0.0, 0.5), (1.0, 0.5)]);
        let px = vec![200u8, 100, 50];
        let info = test_info();
        let result = hue_vs_sat(&px, &info, &identity).unwrap();
        assert_eq!(result, px, "identity hue_vs_sat should not change pixels");
    }

    #[test]
    fn hue_vs_sat_zero_desaturates() {
        let zero = build_hue_curve_lut(&[(0.0, 0.0), (1.0, 0.0)]);
        let px = vec![200u8, 100, 50];
        let info = test_info();
        let result = hue_vs_sat(&px, &info, &zero).unwrap();
        assert_eq!(result[0], result[1], "desaturated R should equal G");
        assert_eq!(result[1], result[2], "desaturated G should equal B");
    }

    #[test]
    fn hue_vs_lum_identity_curve() {
        let identity = build_hue_curve_lut(&[(0.0, 0.5), (1.0, 0.5)]);
        let px = vec![200u8, 100, 50];
        let info = test_info();
        let result = hue_vs_lum(&px, &info, &identity).unwrap();
        assert_eq!(result, px, "identity hue_vs_lum should not change pixels");
    }

    #[test]
    fn lum_vs_sat_identity_curve() {
        let identity = build_norm_curve_lut(&[(0.0, 0.5), (1.0, 0.5)]);
        let px = vec![200u8, 100, 50];
        let info = test_info();
        let result = lum_vs_sat(&px, &info, &identity).unwrap();
        assert_eq!(result, px, "identity lum_vs_sat should not change pixels");
    }

    #[test]
    fn sat_vs_sat_identity_curve() {
        let identity = build_norm_curve_lut(&[(0.0, 0.5), (1.0, 0.5)]);
        let px = vec![200u8, 100, 50];
        let info = test_info();
        let result = sat_vs_sat(&px, &info, &identity).unwrap();
        assert_eq!(result, px, "identity sat_vs_sat should not change pixels");
    }

    #[test]
    fn hue_vs_sat_selective_desaturation() {
        let pts = [
            (0.0, 0.5),
            (0.5, 0.5),
            (0.667, 0.0),
            (0.833, 0.5),
            (1.0, 0.5),
        ];
        let curve = build_hue_curve_lut(&pts);

        let blue = vec![0u8, 0, 255];
        let info = test_info();
        let result = hue_vs_sat(&blue, &info, &curve).unwrap();
        let max_ch = result[0].max(result[1]).max(result[2]);
        let min_ch = result[0].min(result[1]).min(result[2]);
        assert!(
            max_ch - min_ch < 30,
            "blue should be desaturated: got {:?}",
            &result[..3]
        );

        let red = vec![255u8, 0, 0];
        let result_red = hue_vs_sat(&red, &info, &curve).unwrap();
        assert_eq!(
            result_red,
            red,
            "red should be preserved: got {:?}",
            &result_red[..3]
        );
    }
}
