use super::{
    ImageError, ImageInfo, apply_rgb_transform, eval_hermite, hsl_to_rgb, monotone_tangents,
    rgb_to_hsl,
};

/// Per-channel tone curve from control points.
///
/// Each channel has a set of control points (x, y) in [0, 1].
/// A monotone cubic spline interpolates between them.
/// The result is a 256-entry LUT for O(1) evaluation.
#[derive(Debug, Clone)]
pub struct ToneCurves {
    pub r: Vec<(f32, f32)>,
    pub g: Vec<(f32, f32)>,
    pub b: Vec<(f32, f32)>,
}

impl Default for ToneCurves {
    fn default() -> Self {
        Self {
            r: vec![(0.0, 0.0), (1.0, 1.0)],
            g: vec![(0.0, 0.0), (1.0, 1.0)],
            b: vec![(0.0, 0.0), (1.0, 1.0)],
        }
    }
}

/// Build a 256-entry LUT from control points using monotone cubic interpolation.
pub fn build_curve_lut(points: &[(f32, f32)]) -> [u8; 256] {
    let mut lut = [0u8; 256];

    if points.len() < 2 {
        // Identity
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        return lut;
    }

    // Sort by x
    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute monotone cubic Hermite spline (Fritsch-Carlson method)
    let n = pts.len();
    let mut m = vec![0.0f32; n]; // tangents

    if n == 2 {
        // Linear
        let slope = (pts[1].1 - pts[0].1) / (pts[1].0 - pts[0].0).max(1e-6);
        m[0] = slope;
        m[1] = slope;
    } else {
        // Compute slopes
        let mut deltas = vec![0.0f32; n - 1];
        for i in 0..n - 1 {
            let dx = (pts[i + 1].0 - pts[i].0).max(1e-6);
            deltas[i] = (pts[i + 1].1 - pts[i].1) / dx;
        }

        // Interior tangents: average of adjacent slopes
        m[0] = deltas[0];
        m[n - 1] = deltas[n - 2];
        for i in 1..n - 1 {
            m[i] = (deltas[i - 1] + deltas[i]) * 0.5;
        }

        // Monotonicity constraint (Fritsch-Carlson)
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
    }

    // Evaluate at each of 256 positions
    #[allow(clippy::needless_range_loop)]
    for i in 0..256 {
        let x = i as f32 / 255.0;

        // Find segment
        let seg = match pts
            .binary_search_by(|p| p.0.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => {
                lut[i] = (pts[idx].1 * 255.0).round().clamp(0.0, 255.0) as u8;
                continue;
            }
            Err(idx) => {
                if idx == 0 {
                    lut[i] = (pts[0].1 * 255.0).round().clamp(0.0, 255.0) as u8;
                    continue;
                }
                if idx >= n {
                    lut[i] = (pts[n - 1].1 * 255.0).round().clamp(0.0, 255.0) as u8;
                    continue;
                }
                idx - 1
            }
        };

        // Hermite interpolation
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

        let y = h00 * y0 + h10 * h * m[seg] + h01 * y1 + h11 * h * m[seg + 1];
        lut[i] = (y * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    lut
}

/// Apply per-channel tone curves to a single pixel.
#[inline]
pub fn curves_pixel(
    r: f32,
    g: f32,
    b: f32,
    r_lut: &[u8; 256],
    g_lut: &[u8; 256],
    b_lut: &[u8; 256],
) -> (f32, f32, f32) {
    let ri = (r * 255.0).round().clamp(0.0, 255.0) as u8;
    let gi = (g * 255.0).round().clamp(0.0, 255.0) as u8;
    let bi = (b * 255.0).round().clamp(0.0, 255.0) as u8;
    (
        r_lut[ri as usize] as f32 / 255.0,
        g_lut[gi as usize] as f32 / 255.0,
        b_lut[bi as usize] as f32 / 255.0,
    )
}

/// Apply per-channel tone curves to an image pixel buffer.
pub fn curves(pixels: &[u8], info: &ImageInfo, tc: &ToneCurves) -> Result<Vec<u8>, ImageError> {
    let r_lut = build_curve_lut(&tc.r);
    let g_lut = build_curve_lut(&tc.g);
    let b_lut = build_curve_lut(&tc.b);
    apply_rgb_transform(pixels, info, |r, g, b| {
        curves_pixel(r, g, b, &r_lut, &g_lut, &b_lut)
    })
}

// ─── Hue-vs-X Curve Grading ─────────────────────────────────────────────

/// Build a 360-entry f32 LUT for hue-indexed curves.
pub fn build_hue_curve_lut(points: &[(f32, f32)]) -> [f32; 360] {
    let mut lut = [0.0f32; 360];

    if points.len() < 2 {
        lut.fill(0.5);
        return lut;
    }

    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = pts.len();
    let m = monotone_tangents(&pts);

    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f32 / 359.0;
        *entry = eval_hermite(&pts, &m, n, x);
    }

    lut
}

/// Build a 256-entry f32 LUT for normalized-value-indexed curves (lum/sat).
pub fn build_norm_curve_lut(points: &[(f32, f32)]) -> [f32; 256] {
    let mut lut = [0.0f32; 256];

    if points.len() < 2 {
        lut.fill(0.5);
        return lut;
    }

    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = pts.len();
    let m = monotone_tangents(&pts);

    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f32 / 255.0;
        *entry = eval_hermite(&pts, &m, n, x);
    }

    lut
}

/// Hue vs Saturation: adjust saturation based on hue.
pub fn hue_vs_sat(
    pixels: &[u8],
    info: &ImageInfo,
    curve: &[f32; 360],
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| {
        let (h, s, l) = rgb_to_hsl(r, g, b);
        let idx = (h.round() as usize).min(359);
        let mult = curve[idx] * 2.0;
        hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
    })
}

/// Hue vs Luminance: adjust luminance based on hue.
pub fn hue_vs_lum(
    pixels: &[u8],
    info: &ImageInfo,
    curve: &[f32; 360],
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| {
        let (h, s, l) = rgb_to_hsl(r, g, b);
        let idx = (h.round() as usize).min(359);
        let offset = (curve[idx] - 0.5) * 2.0;
        hsl_to_rgb(h, s, (l + offset).clamp(0.0, 1.0))
    })
}

/// Luminance vs Saturation: adjust saturation based on luminance.
pub fn lum_vs_sat(
    pixels: &[u8],
    info: &ImageInfo,
    curve: &[f32; 256],
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| {
        let (h, s, l) = rgb_to_hsl(r, g, b);
        let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
        let mult = curve[idx] * 2.0;
        hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
    })
}

/// Saturation vs Saturation: remap saturation based on current saturation.
pub fn sat_vs_sat(
    pixels: &[u8],
    info: &ImageInfo,
    curve: &[f32; 256],
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| {
        let (h, s, l) = rgb_to_hsl(r, g, b);
        let idx = (s * 255.0).round().clamp(0.0, 255.0) as usize;
        let mult = curve[idx] * 2.0;
        hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
    })
}
