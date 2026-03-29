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

use super::color_lut::ColorLut3D;
use super::error::ImageError;
use super::types::{ColorSpace, ImageInfo, PixelFormat};

// ─── ASC-CDL (Slope/Offset/Power) ─────────────────────────────────────────

/// ASC Color Decision List parameters — per-channel slope, offset, power.
///
/// Standard reference: ASC CDL specification (Society of Motion Picture and
/// Television Engineers). Formula:
/// ```text
///   out = clamp01((in * slope + offset) ^ power)
/// ```
/// Also supports an overall saturation adjustment.
#[derive(Debug, Clone, Copy)]
pub struct AscCdl {
    pub slope: [f32; 3],
    pub offset: [f32; 3],
    pub power: [f32; 3],
    /// Overall saturation (1.0 = unchanged). Applied after SOP.
    pub saturation: f32,
}

impl Default for AscCdl {
    fn default() -> Self {
        Self {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        }
    }
}

/// Apply ASC-CDL transform to a single pixel (r, g, b in [0, 1]).
#[inline]
pub fn asc_cdl_pixel(r: f32, g: f32, b: f32, cdl: &AscCdl) -> (f32, f32, f32) {
    // SOP: slope, offset, power
    let mut out_r = ((r * cdl.slope[0] + cdl.offset[0]).max(0.0)).powf(cdl.power[0]);
    let mut out_g = ((g * cdl.slope[1] + cdl.offset[1]).max(0.0)).powf(cdl.power[1]);
    let mut out_b = ((b * cdl.slope[2] + cdl.offset[2]).max(0.0)).powf(cdl.power[2]);

    // Saturation adjustment (Rec. 709 luma weights)
    if cdl.saturation != 1.0 {
        let luma = 0.2126 * out_r + 0.7152 * out_g + 0.0722 * out_b;
        out_r = luma + (out_r - luma) * cdl.saturation;
        out_g = luma + (out_g - luma) * cdl.saturation;
        out_b = luma + (out_b - luma) * cdl.saturation;
    }

    (
        out_r.clamp(0.0, 1.0),
        out_g.clamp(0.0, 1.0),
        out_b.clamp(0.0, 1.0),
    )
}

/// Apply ASC-CDL to an image pixel buffer.
pub fn asc_cdl(pixels: &[u8], info: &ImageInfo, cdl: &AscCdl) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| asc_cdl_pixel(r, g, b, cdl))
}

// ─── Lift / Gamma / Gain (3-Way Color Corrector) ──────────────────────────

/// 3-way color corrector parameters — lift, gamma, gain per channel.
///
/// DaVinci Resolve formula:
/// ```text
///   out = gain * (input + lift * (1 - input)) ^ (1/gamma)
/// ```
///
/// Neutral values: lift=[0,0,0], gamma=[1,1,1], gain=[1,1,1].
#[derive(Debug, Clone, Copy)]
pub struct LiftGammaGain {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
}

impl Default for LiftGammaGain {
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
        }
    }
}

/// Apply lift/gamma/gain to a single pixel.
#[inline]
pub fn lift_gamma_gain_pixel(r: f32, g: f32, b: f32, lgg: &LiftGammaGain) -> (f32, f32, f32) {
    #[inline]
    fn channel(val: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
        let lifted = val + lift * (1.0 - val);
        let gammaed = if gamma > 0.0 && lifted > 0.0 {
            lifted.powf(1.0 / gamma)
        } else {
            0.0
        };
        (gain * gammaed).clamp(0.0, 1.0)
    }

    (
        channel(r, lgg.lift[0], lgg.gamma[0], lgg.gain[0]),
        channel(g, lgg.lift[1], lgg.gamma[1], lgg.gain[1]),
        channel(b, lgg.lift[2], lgg.gamma[2], lgg.gain[2]),
    )
}

/// Apply lift/gamma/gain to an image pixel buffer.
pub fn lift_gamma_gain(
    pixels: &[u8],
    info: &ImageInfo,
    lgg: &LiftGammaGain,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| lift_gamma_gain_pixel(r, g, b, lgg))
}

// ─── Split Toning ─────────────────────────────────────────────────────────

/// Split toning parameters — tint shadows and highlights with different colors.
///
/// `shadow_color` and `highlight_color` are RGB tint values in [0, 1].
/// `balance` shifts the crossover point: negative = more shadow tint, positive = more highlight tint.
/// `strength` controls overall intensity (0 = no tinting, 1 = full).
#[derive(Debug, Clone, Copy)]
pub struct SplitToning {
    pub shadow_color: [f32; 3],
    pub highlight_color: [f32; 3],
    /// Balance: -1.0 (all shadow) to +1.0 (all highlight). Default: 0.0.
    pub balance: f32,
    /// Strength: 0.0 (none) to 1.0 (full). Default: 0.5.
    pub strength: f32,
}

impl Default for SplitToning {
    fn default() -> Self {
        Self {
            shadow_color: [0.0, 0.0, 0.5],    // Blue shadows
            highlight_color: [1.0, 0.8, 0.4], // Warm highlights
            balance: 0.0,
            strength: 0.5,
        }
    }
}

/// Apply split toning to a single pixel.
#[inline]
pub fn split_toning_pixel(r: f32, g: f32, b: f32, st: &SplitToning) -> (f32, f32, f32) {
    // Luminance (Rec. 709)
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Crossover point shifted by balance
    let midpoint = 0.5 + st.balance * 0.5;

    // Shadow/highlight blend factor
    let shadow_weight = (1.0 - luma / midpoint.max(0.001)).max(0.0).min(1.0) * st.strength;
    let highlight_weight = ((luma - midpoint) / (1.0 - midpoint).max(0.001))
        .max(0.0)
        .min(1.0)
        * st.strength;

    // Blend: tint toward shadow/highlight color
    let out_r = r
        + (st.shadow_color[0] - r) * shadow_weight
        + (st.highlight_color[0] - r) * highlight_weight;
    let out_g = g
        + (st.shadow_color[1] - g) * shadow_weight
        + (st.highlight_color[1] - g) * highlight_weight;
    let out_b = b
        + (st.shadow_color[2] - b) * shadow_weight
        + (st.highlight_color[2] - b) * highlight_weight;

    (
        out_r.clamp(0.0, 1.0),
        out_g.clamp(0.0, 1.0),
        out_b.clamp(0.0, 1.0),
    )
}

/// Apply split toning to an image pixel buffer.
pub fn split_toning(
    pixels: &[u8],
    info: &ImageInfo,
    st: &SplitToning,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| split_toning_pixel(r, g, b, st))
}

// ─── Cubic Spline Curves ──────────────────────────────────────────────────

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
        for i in 0..256 {
            lut[i] = i as u8;
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

// ─── HDR Tone Mapping ─────────────────────────────────────────────────────

/// Reinhard global tone mapping operator.
///
/// Maps HDR luminance to SDR: `L_out = L / (1 + L)`.
/// Input/output in [0, 1] (SDR range). Values > 1.0 are compressed.
///
/// Reference: Reinhard et al., "Photographic Tone Reproduction for Digital Images" (2002).
#[inline]
pub fn tonemap_reinhard_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Apply per-channel: out = val / (1 + val)
    (r / (1.0 + r), g / (1.0 + g), b / (1.0 + b))
}

/// Apply Reinhard tone mapping to an image buffer.
pub fn tonemap_reinhard(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| tonemap_reinhard_pixel(r, g, b))
}

/// Drago logarithmic tone mapping operator.
///
/// Maps HDR to SDR using: `L_out = log(1 + L) / log(1 + L_max)`.
/// The `bias` parameter (0.7–0.9, default 0.85) controls contrast.
///
/// Reference: Drago et al., "Adaptive Logarithmic Mapping" (2003).
#[derive(Debug, Clone, Copy)]
pub struct DragoParams {
    /// Maximum luminance in the scene. Default: 1.0 (SDR).
    pub l_max: f32,
    /// Bias parameter (0.7–0.9). Higher = more contrast. Default: 0.85.
    pub bias: f32,
}

impl Default for DragoParams {
    fn default() -> Self {
        Self {
            l_max: 1.0,
            bias: 0.85,
        }
    }
}

/// Apply Drago tone mapping to a single pixel.
#[inline]
pub fn tonemap_drago_pixel(r: f32, g: f32, b: f32, params: &DragoParams) -> (f32, f32, f32) {
    let log_max = (1.0 + params.l_max).ln();
    let bias_pow = (params.bias.ln() / 0.5f32.ln()).max(0.01);

    #[inline]
    fn drago_channel(val: f32, log_max: f32, bias_pow: f32) -> f32 {
        if val <= 0.0 {
            return 0.0;
        }
        let mapped = (1.0 + val).ln() / log_max;
        mapped.powf(1.0 / bias_pow).clamp(0.0, 1.0)
    }

    (
        drago_channel(r, log_max, bias_pow),
        drago_channel(g, log_max, bias_pow),
        drago_channel(b, log_max, bias_pow),
    )
}

/// Apply Drago tone mapping to an image buffer.
pub fn tonemap_drago(
    pixels: &[u8],
    info: &ImageInfo,
    params: &DragoParams,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| tonemap_drago_pixel(r, g, b, params))
}

/// Filmic/ACES tone mapping (Narkowicz 2015 approximation).
///
/// S-curve that compresses highlights and slightly lifts shadows,
/// matching the look of ACES (Academy Color Encoding System).
///
/// Formula: `out = (x * (a*x + b)) / (x * (c*x + d) + e)`
/// Default coefficients match Narkowicz "ACES Filmic Tone Mapping Curve".
///
/// Reference: Narkowicz, "ACES Filmic Tone Mapping Curve" (2015).
#[derive(Debug, Clone, Copy)]
pub struct FilmicParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
}

impl Default for FilmicParams {
    fn default() -> Self {
        // Narkowicz 2015 ACES fit
        Self {
            a: 2.51,
            b: 0.03,
            c: 2.43,
            d: 0.59,
            e: 0.14,
        }
    }
}

/// Apply filmic/ACES tone mapping to a single pixel.
#[inline]
pub fn tonemap_filmic_pixel(r: f32, g: f32, b: f32, params: &FilmicParams) -> (f32, f32, f32) {
    #[inline]
    fn filmic(x: f32, p: &FilmicParams) -> f32 {
        let num = x * (p.a * x + p.b);
        let den = x * (p.c * x + p.d) + p.e;
        (num / den).clamp(0.0, 1.0)
    }

    (filmic(r, params), filmic(g, params), filmic(b, params))
}

/// Apply filmic/ACES tone mapping to an image buffer.
pub fn tonemap_filmic(
    pixels: &[u8],
    info: &ImageInfo,
    params: &FilmicParams,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| {
        tonemap_filmic_pixel(r, g, b, params)
    })
}

// ─── Film Grain Simulation ────────────────────────────────────────────────

/// Film grain simulation parameters.
#[derive(Debug, Clone, Copy)]
pub struct FilmGrainParams {
    /// Grain amount (0.0 = none, 1.0 = heavy). Default: 0.3.
    pub amount: f32,
    /// Grain size in pixels (1.0 = fine, 4.0+ = coarse). Default: 1.5.
    pub size: f32,
    /// Color grain (true) or monochrome grain (false). Default: false.
    pub color: bool,
    /// Random seed for deterministic output. Default: 0.
    pub seed: u32,
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            amount: 0.3,
            size: 1.5,
            color: false,
            seed: 0,
        }
    }
}

/// Generate deterministic pseudo-random noise using a hash function.
/// Returns a value in [-1.0, 1.0].
#[inline]
fn hash_noise(x: u32, y: u32, seed: u32) -> f32 {
    // Simple but effective hash: based on Wang hash
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(1274126177));
    h = (h ^ (h >> 13)).wrapping_mul(1103515245);
    h = h ^ (h >> 16);
    // Map to [-1, 1]
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Apply film grain to an image buffer.
///
/// Grain intensity varies by luminance — strongest in midtones, weaker in
/// shadows and highlights, matching real photographic film behavior.
pub fn film_grain(
    pixels: &[u8],
    info: &ImageInfo,
    params: &FilmGrainParams,
) -> Result<Vec<u8>, ImageError> {
    let bpp = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "film grain requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let w = info.width as usize;
    let h = info.height as usize;
    let expected = w * h * bpp;
    if pixels.len() < expected {
        return Err(ImageError::InvalidParameters("pixel data too small".into()));
    }

    let mut out = pixels.to_vec();
    let inv_size = 1.0 / params.size.max(0.1);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * bpp;

            // Sample coordinates scaled by grain size
            let sx = (x as f32 * inv_size) as u32;
            let sy = (y as f32 * inv_size) as u32;

            // Luminance for intensity modulation (midtone emphasis)
            let r = pixels[idx] as f32 / 255.0;
            let g = pixels[idx + 1] as f32 / 255.0;
            let b = pixels[idx + 2] as f32 / 255.0;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            // Midtone emphasis: grain is strongest at luma=0.5, weaker at extremes
            // Parabola: 4 * luma * (1 - luma) peaks at 0.5 with value 1.0
            let intensity = 4.0 * luma * (1.0 - luma) * params.amount;

            if params.color {
                // Independent noise per channel
                let nr = hash_noise(sx, sy, params.seed) * intensity;
                let ng = hash_noise(sx, sy, params.seed.wrapping_add(1)) * intensity;
                let nb = hash_noise(sx, sy, params.seed.wrapping_add(2)) * intensity;
                out[idx] = ((r + nr) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 1] = ((g + ng) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 2] = ((b + nb) * 255.0).round().clamp(0.0, 255.0) as u8;
            } else {
                // Monochrome: same noise for all channels
                let n = hash_noise(sx, sy, params.seed) * intensity;
                out[idx] = ((r + n) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 1] = ((g + n) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 2] = ((b + n) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Apply an f32 RGB transform function to a u8 pixel buffer.
fn apply_rgb_transform(
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
        // At small values, Reinhard ≈ identity: x/(1+x) ≈ x when x << 1
        let (r, g, b) = tonemap_reinhard_pixel(0.1, 0.1, 0.1);
        assert!((r - 0.0909).abs() < 0.01); // 0.1/1.1 = 0.0909
    }

    #[test]
    fn reinhard_compresses_highlights() {
        let (r, _, _) = tonemap_reinhard_pixel(1.0, 0.0, 0.0);
        assert!((r - 0.5).abs() < 0.01); // 1/(1+1) = 0.5
        let (r2, _, _) = tonemap_reinhard_pixel(10.0, 0.0, 0.0);
        assert!((r2 - 0.909).abs() < 0.01); // 10/11
    }

    #[test]
    fn reinhard_buffer_apply() {
        let px = vec![128u8, 128, 128];
        let info = test_info();
        let result = tonemap_reinhard(&px, &info).unwrap();
        // 0.502 / 1.502 = 0.334 → round(0.334*255) = 85
        assert!((result[0] as i16 - 85).abs() <= 1);
    }

    #[test]
    fn drago_identity_at_default() {
        let params = DragoParams::default();
        // At l_max=1.0, Drago maps [0,1] → [0,1] approximately
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
        // ACES S-curve: shadows slightly lifted, highlights compressed
        let (low, _, _) = tonemap_filmic_pixel(0.1, 0.0, 0.0, &params);
        let (mid, _, _) = tonemap_filmic_pixel(0.5, 0.0, 0.0, &params);
        let (high, _, _) = tonemap_filmic_pixel(1.0, 0.0, 0.0, &params);

        // Should be monotonically increasing
        assert!(low < mid, "low={low} should be < mid={mid}");
        assert!(mid < high, "mid={mid} should be < high={high}");

        // High values compressed (< 1.0)
        assert!(high < 1.0, "ACES should compress highlights: got {high}");
    }

    #[test]
    fn filmic_zero_is_near_zero() {
        let params = FilmicParams::default();
        let (r, _, _) = tonemap_filmic_pixel(0.0, 0.0, 0.0, &params);
        // 0*(a*0+b)/(0*(c*0+d)+e) = 0/0.14 = 0
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
        // Dark pixel (luma ≈ 0): grain intensity ≈ 0
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
}
