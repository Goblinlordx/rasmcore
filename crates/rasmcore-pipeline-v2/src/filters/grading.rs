//! Color grading filters — f32-only tone curves, CDL, LUT application, tone mapping.
//!
//! All CLUT-compatible filters implement `ClutOp` for fusion.
//! Film grain is spatial (position-dependent) and not CLUT-compatible.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use super::color::ClutOp;

// ─── HSL helpers (re-exported from color module internals) ─────────────────
// We duplicate these small helpers here to avoid making color module internals pub.

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < 1e-7 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if (max - r).abs() < 1e-7 {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h * 60.0
    } else if (max - g).abs() < 1e-7 {
        ((b - r) / d + 2.0) * 60.0
    } else {
        ((r - g) / d + 4.0) * 60.0
    };
    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-7 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;
    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

fn bt709_luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ─── Monotone Cubic Hermite Spline ─────────────────────────────────────────

/// Build a f32 LUT from control points using monotone cubic Hermite interpolation.
/// `lut_size` entries, indexed by normalized [0,1] → [0, lut_size-1].
fn build_curve_lut_f32(points: &[(f32, f32)], lut_size: usize) -> Vec<f32> {
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

// ─── Curves Filters ────────────────────────────────────────────────────────

/// Per-channel tone curves — master (same curve for R, G, B).
#[derive(Clone)]
pub struct CurvesMaster {
    /// Control points as (x, y) pairs in [0,1].
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesMaster {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = lut[(pixel[0] * 4095.0).round().clamp(0.0, 4095.0) as usize];
            pixel[1] = lut[(pixel[1] * 4095.0).round().clamp(0.0, 4095.0) as usize];
            pixel[2] = lut[(pixel[2] * 4095.0).round().clamp(0.0, 4095.0) as usize];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for CurvesMaster {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let ri = (r * 4095.0).round().clamp(0.0, 4095.0) as usize;
            let gi = (g * 4095.0).round().clamp(0.0, 4095.0) as usize;
            let bi = (b * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (lut[ri], lut[gi], lut[bi])
        })
    }
}

/// Per-channel tone curve — red channel only.
#[derive(Clone)]
pub struct CurvesRed {
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesRed {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let idx = (pixel[0] * 4095.0).round().clamp(0.0, 4095.0) as usize;
            pixel[0] = lut[idx];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for CurvesRed {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let ri = (r * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (lut[ri], g, b)
        })
    }
}

/// Per-channel tone curve — green channel only.
#[derive(Clone)]
pub struct CurvesGreen {
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesGreen {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let idx = (pixel[1] * 4095.0).round().clamp(0.0, 4095.0) as usize;
            pixel[1] = lut[idx];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for CurvesGreen {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let gi = (g * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (r, lut[gi], b)
        })
    }
}

/// Per-channel tone curve — blue channel only.
#[derive(Clone)]
pub struct CurvesBlue {
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesBlue {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let idx = (pixel[2] * 4095.0).round().clamp(0.0, 4095.0) as usize;
            pixel[2] = lut[idx];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for CurvesBlue {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let bi = (b * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (r, g, lut[bi])
        })
    }
}

// ─── ASC CDL ───────────────────────────────────────────────────────────────

/// ASC Color Decision List — per-channel slope, offset, power with optional saturation.
///
/// Formula: `out = clamp01((in * slope + offset) ^ power)`
/// Optional saturation adjustment via Rec. 709 luma.
#[derive(Clone)]
pub struct AscCdl {
    pub slope: [f32; 3],
    pub offset: [f32; 3],
    pub power: [f32; 3],
    /// Overall saturation (1.0 = unchanged).
    pub saturation: f32,
}

impl Filter for AscCdl {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = asc_cdl_pixel(pixel[0], pixel[1], pixel[2], self);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

fn asc_cdl_pixel(r: f32, g: f32, b: f32, cdl: &AscCdl) -> (f32, f32, f32) {
    let mut or = ((r * cdl.slope[0] + cdl.offset[0]).max(0.0)).powf(cdl.power[0]);
    let mut og = ((g * cdl.slope[1] + cdl.offset[1]).max(0.0)).powf(cdl.power[1]);
    let mut ob = ((b * cdl.slope[2] + cdl.offset[2]).max(0.0)).powf(cdl.power[2]);
    if cdl.saturation != 1.0 {
        let luma = bt709_luma(or, og, ob);
        or = luma + (or - luma) * cdl.saturation;
        og = luma + (og - luma) * cdl.saturation;
        ob = luma + (ob - luma) * cdl.saturation;
    }
    (or.clamp(0.0, 1.0), og.clamp(0.0, 1.0), ob.clamp(0.0, 1.0))
}

impl ClutOp for AscCdl {
    fn build_clut(&self) -> Clut3D {
        let cdl = self.clone();
        Clut3D::from_fn(33, move |r, g, b| asc_cdl_pixel(r, g, b, &cdl))
    }
}

// ─── Lift/Gamma/Gain ───────────────────────────────────────────────────────

/// 3-way color corrector — DaVinci Resolve style lift/gamma/gain per channel.
///
/// Formula: `out = gain * (input + lift * (1 - input)) ^ (1/gamma)`
#[derive(Clone)]
pub struct LiftGammaGain {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
}

impl Filter for LiftGammaGain {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = lgg_pixel(pixel[0], pixel[1], pixel[2], self);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

fn lgg_channel(val: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
    let lifted = val + lift * (1.0 - val);
    let gammaed = if gamma > 0.0 && lifted > 0.0 {
        lifted.powf(1.0 / gamma)
    } else {
        0.0
    };
    (gain * gammaed).clamp(0.0, 1.0)
}

fn lgg_pixel(r: f32, g: f32, b: f32, lgg: &LiftGammaGain) -> (f32, f32, f32) {
    (
        lgg_channel(r, lgg.lift[0], lgg.gamma[0], lgg.gain[0]),
        lgg_channel(g, lgg.lift[1], lgg.gamma[1], lgg.gain[1]),
        lgg_channel(b, lgg.lift[2], lgg.gamma[2], lgg.gain[2]),
    )
}

impl ClutOp for LiftGammaGain {
    fn build_clut(&self) -> Clut3D {
        let lgg = self.clone();
        Clut3D::from_fn(33, move |r, g, b| lgg_pixel(r, g, b, &lgg))
    }
}

// ─── Split Toning ──────────────────────────────────────────────────────────

/// Split toning — tint shadows and highlights with different colors.
#[derive(Clone)]
pub struct SplitToning {
    pub shadow_color: [f32; 3],
    pub highlight_color: [f32; 3],
    /// Balance: -1.0 (all shadow) to +1.0 (all highlight).
    pub balance: f32,
    /// Strength: 0.0 (none) to 1.0 (full).
    pub strength: f32,
}

impl Filter for SplitToning {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = split_toning_pixel(pixel[0], pixel[1], pixel[2], self);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

fn split_toning_pixel(r: f32, g: f32, b: f32, st: &SplitToning) -> (f32, f32, f32) {
    let luma = bt709_luma(r, g, b);
    let midpoint = 0.5 + st.balance * 0.5;
    let shadow_w = (1.0 - luma / midpoint.max(0.001)).clamp(0.0, 1.0) * st.strength;
    let highlight_w = ((luma - midpoint) / (1.0 - midpoint).max(0.001)).clamp(0.0, 1.0) * st.strength;
    let or = r + (st.shadow_color[0] - r) * shadow_w + (st.highlight_color[0] - r) * highlight_w;
    let og = g + (st.shadow_color[1] - g) * shadow_w + (st.highlight_color[1] - g) * highlight_w;
    let ob = b + (st.shadow_color[2] - b) * shadow_w + (st.highlight_color[2] - b) * highlight_w;
    (or.clamp(0.0, 1.0), og.clamp(0.0, 1.0), ob.clamp(0.0, 1.0))
}

impl ClutOp for SplitToning {
    fn build_clut(&self) -> Clut3D {
        let st = self.clone();
        Clut3D::from_fn(33, move |r, g, b| split_toning_pixel(r, g, b, &st))
    }
}

// ─── Hue-Based Curves ──────────────────────────────────────────────────────

/// Hue vs Saturation — adjust saturation based on hue position.
#[derive(Clone)]
pub struct HueVsSat {
    /// Control points (x in [0,1] mapping to hue 0-360, y in [0,1]).
    /// y=0.5 is neutral, y>0.5 boosts, y<0.5 reduces.
    pub points: Vec<(f32, f32)>,
}

impl Filter for HueVsSat {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 360);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (h.round() as usize).min(359);
            let mult = lut[idx] * 2.0;
            let (r, g, b) = hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for HueVsSat {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 360);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (h.round() as usize).min(359);
            let mult = lut[idx] * 2.0;
            hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
        })
    }
}

/// Hue vs Luminance — adjust luminance based on hue position.
#[derive(Clone)]
pub struct HueVsLum {
    pub points: Vec<(f32, f32)>,
}

impl Filter for HueVsLum {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 360);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (h.round() as usize).min(359);
            let offset = (lut[idx] - 0.5) * 2.0;
            let (r, g, b) = hsl_to_rgb(h, s, (l + offset).clamp(0.0, 1.0));
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for HueVsLum {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 360);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (h.round() as usize).min(359);
            let offset = (lut[idx] - 0.5) * 2.0;
            hsl_to_rgb(h, s, (l + offset).clamp(0.0, 1.0))
        })
    }
}

/// Luminance vs Saturation — adjust saturation based on luminance.
#[derive(Clone)]
pub struct LumVsSat {
    pub points: Vec<(f32, f32)>,
}

impl Filter for LumVsSat {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 256);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            let (r, g, b) = hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for LumVsSat {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 256);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
        })
    }
}

/// Saturation vs Saturation — remap saturation based on current saturation.
#[derive(Clone)]
pub struct SatVsSat {
    pub points: Vec<(f32, f32)>,
}

impl Filter for SatVsSat {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 256);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (s * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            let (r, g, b) = hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for SatVsSat {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 256);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (s * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
        })
    }
}

// ─── LUT Application ───────────────────────────────────────────────────────

/// Apply a .cube format 3D LUT.
///
/// The Clut3D is pre-built from parsed .cube data. This filter wraps it
/// as a standard Filter + ClutOp for pipeline integration.
#[derive(Clone)]
pub struct ApplyCubeLut {
    pub clut: Clut3D,
}

impl Filter for ApplyCubeLut {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.clut.apply(input))
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for ApplyCubeLut {
    fn build_clut(&self) -> Clut3D {
        self.clut.clone()
    }
}

/// Apply a Hald CLUT image as a 3D LUT.
///
/// The Clut3D is pre-built from parsed Hald image data.
#[derive(Clone)]
pub struct ApplyHaldLut {
    pub clut: Clut3D,
}

impl Filter for ApplyHaldLut {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.clut.apply(input))
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for ApplyHaldLut {
    fn build_clut(&self) -> Clut3D {
        self.clut.clone()
    }
}

// ─── Tone Mapping ──────────────────────────────────────────────────────────

/// Reinhard global tone mapping: `out = v / (1 + v)`.
/// Maps HDR [0, ∞) to [0, 1).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tonemap_reinhard", category = "grading", cost = "O(n)")]
pub struct TonemapReinhard;

impl Filter for TonemapReinhard {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0] / (1.0 + pixel[0]);
            pixel[1] = pixel[1] / (1.0 + pixel[1]);
            pixel[2] = pixel[2] / (1.0 + pixel[2]);
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for TonemapReinhard {
    fn build_clut(&self) -> Clut3D {
        Clut3D::from_fn(33, |r, g, b| {
            (r / (1.0 + r), g / (1.0 + g), b / (1.0 + b))
        })
    }
}

/// Drago logarithmic tone mapping.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tonemap_drago", category = "grading", cost = "O(n)")]
pub struct TonemapDrago {
    /// Maximum luminance in scene (default 1.0 for SDR).
    #[param(min = 0.0, max = 100.0, default = 1.0)]
    pub l_max: f32,
    /// Bias parameter (0.7-0.9, default 0.85).
    #[param(min = 0.5, max = 1.0, default = 0.85)]
    pub bias: f32,
}

impl Filter for TonemapDrago {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let log_max = (1.0 + self.l_max).ln();
        let bias_pow = (self.bias.ln() / 0.5f32.ln()).max(0.01);
        let mut out = input.to_vec();
        let drago = |v: f32| -> f32 {
            if v <= 0.0 { 0.0 } else { ((1.0 + v).ln() / log_max).powf(1.0 / bias_pow).clamp(0.0, 1.0) }
        };
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = drago(pixel[0]);
            pixel[1] = drago(pixel[1]);
            pixel[2] = drago(pixel[2]);
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for TonemapDrago {
    fn build_clut(&self) -> Clut3D {
        let log_max = (1.0 + self.l_max).ln();
        let bias_pow = (self.bias.ln() / 0.5f32.ln()).max(0.01);
        Clut3D::from_fn(33, move |r, g, b| {
            let drago = |v: f32| -> f32 {
                if v <= 0.0 {
                    0.0
                } else {
                    ((1.0 + v).ln() / log_max).powf(1.0 / bias_pow).clamp(0.0, 1.0)
                }
            };
            (drago(r), drago(g), drago(b))
        })
    }
}

/// Filmic/ACES tone mapping (Narkowicz 2015 approximation).
///
/// Formula: `out = (x*(a*x+b)) / (x*(c*x+d) + e)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tonemap_filmic", category = "grading", cost = "O(n)")]
pub struct TonemapFilmic {
    #[param(min = 0.0, max = 10.0, default = 2.51)]
    pub a: f32,
    #[param(min = 0.0, max = 1.0, default = 0.03)]
    pub b: f32,
    #[param(min = 0.0, max = 10.0, default = 2.43)]
    pub c: f32,
    #[param(min = 0.0, max = 2.0, default = 0.59)]
    pub d: f32,
    #[param(min = 0.0, max = 1.0, default = 0.14)]
    pub e: f32,
}

impl Filter for TonemapFilmic {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let (a, b, c, d, e) = (self.a, self.b, self.c, self.d, self.e);
        let filmic = |x: f32| -> f32 {
            (x * (a * x + b) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
        };
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = filmic(pixel[0]);
            pixel[1] = filmic(pixel[1]);
            pixel[2] = filmic(pixel[2]);
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for TonemapFilmic {
    fn build_clut(&self) -> Clut3D {
        let (a, b, c, d, e) = (self.a, self.b, self.c, self.d, self.e);
        Clut3D::from_fn(33, move |r, g, bi| {
            let filmic = |x: f32| -> f32 {
                let num = x * (a * x + b);
                let den = x * (c * x + d) + e;
                (num / den).clamp(0.0, 1.0)
            };
            (filmic(r), filmic(g), filmic(bi))
        })
    }
}

// ─── Film Grain ────────────────────────────────────────────────────────────

/// Film grain simulation — position-dependent, NOT CLUT-compatible.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "film_grain_grading", category = "grading", cost = "O(n)")]
pub struct FilmGrain {
    /// Grain amount (0.0 = none, 1.0 = heavy).
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub amount: f32,
    /// Grain size in pixels (1.0 = fine, 4.0+ = coarse).
    #[param(min = 0.1, max = 10.0, default = 1.0)]
    pub size: f32,
    /// Color grain (true) or monochrome (false).
    #[param(default = false)]
    pub color: bool,
    /// Random seed for deterministic output.
    #[param(min = 0, max = 100, default = 42)]
    pub seed: u32,
}

impl Filter for FilmGrain {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let inv_size = 1.0 / self.size.max(0.1);
        let seed = self.seed as u64 ^ noise::SEED_FILM_GRAIN;
        let mut out = input.to_vec();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * 4;
                let sx = (x as f32 * inv_size) as u32;
                let sy = (y as f32 * inv_size) as u32;
                let (r, g, b) = (out[idx], out[idx + 1], out[idx + 2]);
                let luma = bt709_luma(r, g, b);
                let intensity = 4.0 * luma * (1.0 - luma) * self.amount;
                if self.color {
                    out[idx] = r + noise::noise_2d(sx, sy, seed) * intensity;
                    out[idx + 1] = g + noise::noise_2d(sx, sy, seed.wrapping_add(1)) * intensity;
                    out[idx + 2] = b + noise::noise_2d(sx, sy, seed.wrapping_add(2)) * intensity;
                } else {
                    let n = noise::noise_2d(sx, sy, seed) * intensity;
                    out[idx] = r + n;
                    out[idx + 1] = g + n;
                    out[idx + 2] = b + n;
                }
            }
        }
        Ok(out)
    }
}

/// WGSL compute shader body for film grain (without noise functions).
///
/// Uses SplitMix64 noise from `noise::NOISE_WGSL` (composed at runtime).
const FILM_GRAIN_WGSL_BODY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  amount: f32,
  inv_size: f32,
  seed_lo: u32,
  seed_hi: u32,
  color_grain: u32,
  _pad: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) {
    return;
  }
  let idx = y * params.width + x;
  let pixel = load_pixel(idx);
  let luma = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  let intensity = 4.0 * luma * (1.0 - luma) * params.amount;
  let sx = u32(f32(x) * params.inv_size);
  let sy = u32(f32(y) * params.inv_size);
  var result = pixel;
  if (params.color_grain != 0u) {
    result.x = pixel.x + noise_2d(sx, sy, params.seed_lo, params.seed_hi) * intensity;
    result.y = pixel.y + noise_2d(sx, sy, params.seed_lo + 1u, params.seed_hi) * intensity;
    result.z = pixel.z + noise_2d(sx, sy, params.seed_lo + 2u, params.seed_hi) * intensity;
  } else {
    let n = noise_2d(sx, sy, params.seed_lo, params.seed_hi) * intensity;
    result.x = pixel.x + n;
    result.y = pixel.y + n;
    result.z = pixel.z + n;
  }
  store_pixel(idx, result);
}
"#;

/// Compose the full film grain shader: NOISE_WGSL + FILM_GRAIN_WGSL_BODY.
fn film_grain_shader() -> String {
    let mut s = String::with_capacity(noise::NOISE_WGSL.len() + FILM_GRAIN_WGSL_BODY.len() + 1);
    s.push_str(noise::NOISE_WGSL);
    s.push('\n');
    s.push_str(FILM_GRAIN_WGSL_BODY);
    s
}

impl GpuFilter for FilmGrain {
    fn shader_body(&self) -> &str {
        // Return the body portion only — the noise functions are composed
        // via the full shader in gpu_shader() override below.
        FILM_GRAIN_WGSL_BODY
    }

    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }

    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let inv_size = 1.0 / self.size.max(0.1);
        let seed = self.seed as u64 ^ noise::SEED_FILM_GRAIN;
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;
        let color_grain: u32 = if self.color { 1 } else { 0 };
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.amount.to_le_bytes());
        buf.extend_from_slice(&inv_size.to_le_bytes());
        buf.extend_from_slice(&seed_lo.to_le_bytes());
        buf.extend_from_slice(&seed_hi.to_le_bytes());
        buf.extend_from_slice(&color_grain.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // _pad
        buf
    }

    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: film_grain_shader(),
            entry_point: self.entry_point(),
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: self.extra_buffers(),
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

// ─── .cube LUT Parser ──────────────────────────────────────────────────────

/// Parse a .cube format 3D LUT from text content into a Clut3D.
///
/// Supports TITLE, DOMAIN_MIN, DOMAIN_MAX, LUT_3D_SIZE directives.
pub fn parse_cube_lut(content: &str) -> Result<Clut3D, PipelineError> {
    let mut grid_size: Option<u32> = None;
    let mut data: Vec<f32> = Vec::new();
    let mut domain_min = [0.0f32; 3];
    let mut domain_max = [1.0f32; 3];

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("TITLE") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
            grid_size = Some(
                rest.trim()
                    .parse::<u32>()
                    .map_err(|_| PipelineError::InvalidParams("invalid LUT_3D_SIZE".into()))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
            let vals: Vec<f32> = rest
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() == 3 {
                domain_min = [vals[0], vals[1], vals[2]];
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
            let vals: Vec<f32> = rest
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() == 3 {
                domain_max = [vals[0], vals[1], vals[2]];
            }
            continue;
        }
        // Data line: three floats
        let vals: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() == 3 {
            // Normalize to [0,1] from domain
            for i in 0..3 {
                let range = (domain_max[i] - domain_min[i]).max(1e-6);
                data.push((vals[i] - domain_min[i]) / range);
            }
        }
    }

    let n = grid_size.ok_or_else(|| PipelineError::InvalidParams("missing LUT_3D_SIZE".into()))?;
    let expected = (n * n * n * 3) as usize;
    if data.len() != expected {
        return Err(PipelineError::InvalidParams(format!(
            "expected {expected} values for {n}^3 LUT, got {}",
            data.len()
        )));
    }

    Ok(Clut3D {
        grid_size: n,
        data,
    })
}

// All grading filters are auto-registered via #[derive(V2Filter)] on their structs.

// ─── Manual Registrations (complex types not supported by V2Filter derive) ──

use crate::filter_node::FilterNode;
#[allow(unused_imports)]
use crate::registry::{
    FilterFactoryRegistration, OperationRegistration, OperationKind,
    OperationCapabilities, ParamDescriptor, ParamMap, ParamType,
};

// ─── Helper: build curve from shadow/highlight params ──────────────────────

/// Generate control points from simplified shadow/midtone/highlight params.
/// Each param bends the curve: positive = brighten that range, negative = darken.
fn curve_from_params(shadows: f32, midtones: f32, highlights: f32) -> Vec<(f32, f32)> {
    vec![
        (0.0, 0.0),
        (0.25, (0.25 + shadows * 0.15).clamp(0.0, 1.0)),
        (0.5, (0.5 + midtones * 0.2).clamp(0.0, 1.0)),
        (0.75, (0.75 + highlights * 0.15).clamp(0.0, 1.0)),
        (1.0, 1.0),
    ]
}

/// Generate hue-indexed curve from a center/amount/width parameterization.
fn hue_curve_from_params(center: f32, amount: f32, width: f32) -> Vec<(f32, f32)> {
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

static CURVE_PARAMS: [ParamDescriptor; 3] = [
    pd_f32!("shadows", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("midtones", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("highlights", -1.0, 1.0, 0.05, 0.0),
];

static HUE_CURVE_PARAMS: [ParamDescriptor; 3] = [
    pd_f32!("center", 0.0, 360.0, 5.0, 0.0),
    pd_f32!("amount", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("width", 10.0, 180.0, 5.0, 60.0),
];

static NORM_CURVE_PARAMS: [ParamDescriptor; 3] = [
    pd_f32!("shadows", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("midtones", -1.0, 1.0, 0.05, 0.0),
    pd_f32!("highlights", -1.0, 1.0, 0.05, 0.0),
];

static ASC_CDL_PARAMS: [ParamDescriptor; 10] = [
    pd_f32!("slope_r", 0.0, 4.0, 0.05, 1.0), pd_f32!("slope_g", 0.0, 4.0, 0.05, 1.0), pd_f32!("slope_b", 0.0, 4.0, 0.05, 1.0),
    pd_f32!("offset_r", -1.0, 1.0, 0.02, 0.0), pd_f32!("offset_g", -1.0, 1.0, 0.02, 0.0), pd_f32!("offset_b", -1.0, 1.0, 0.02, 0.0),
    pd_f32!("power_r", 0.1, 4.0, 0.05, 1.0), pd_f32!("power_g", 0.1, 4.0, 0.05, 1.0), pd_f32!("power_b", 0.1, 4.0, 0.05, 1.0),
    pd_f32!("saturation", 0.0, 4.0, 0.05, 1.0),
];

static SPLIT_TONING_PARAMS: [ParamDescriptor; 4] = [
    pd_f32!("shadow_hue", 0.0, 360.0, 5.0, 220.0),
    pd_f32!("highlight_hue", 0.0, 360.0, 5.0, 40.0),
    pd_f32!("shadow_strength", 0.0, 1.0, 0.05, 0.0),
    pd_f32!("highlight_strength", 0.0, 1.0, 0.05, 0.0),
];

static LIFT_GAMMA_GAIN_PARAMS: [ParamDescriptor; 9] = [
    pd_f32!("lift_r", -0.5, 0.5, 0.02, 0.0), pd_f32!("lift_g", -0.5, 0.5, 0.02, 0.0), pd_f32!("lift_b", -0.5, 0.5, 0.02, 0.0),
    pd_f32!("gamma_r", 0.1, 4.0, 0.05, 1.0), pd_f32!("gamma_g", 0.1, 4.0, 0.05, 1.0), pd_f32!("gamma_b", 0.1, 4.0, 0.05, 1.0),
    pd_f32!("gain_r", 0.0, 4.0, 0.05, 1.0), pd_f32!("gain_g", 0.0, 4.0, 0.05, 1.0), pd_f32!("gain_b", 0.0, 4.0, 0.05, 1.0),
];

// ─── Helper: HSL for split toning ──────────────────────────────────────────

fn hsl_to_rgb_simple(h: f32, s: f32, l: f32) -> [f32; 3] {
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

// ─── Registrations ─────────────────────────────────────────────────────────

// Curves Master
inventory::submit! { &FilterFactoryRegistration { name: "curves_master",
    display_name: "Curves (Master)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesMaster { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_master", display_name: "Curves (Master)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Curves Red
inventory::submit! { &FilterFactoryRegistration { name: "curves_red",
    display_name: "Curves (Red)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesRed { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_red", display_name: "Curves (Red)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Curves Green
inventory::submit! { &FilterFactoryRegistration { name: "curves_green",
    display_name: "Curves (Green)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesGreen { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_green", display_name: "Curves (Green)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Curves Blue
inventory::submit! { &FilterFactoryRegistration { name: "curves_blue",
    display_name: "Curves (Blue)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesBlue { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_blue", display_name: "Curves (Blue)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Hue vs Saturation
inventory::submit! { &FilterFactoryRegistration { name: "hue_vs_sat",
    display_name: "Hue vs Sat", category: "grading", params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = hue_curve_from_params(params.get_f32("center"), params.get_f32("amount"), params.get_f32("width"));
        Box::new(FilterNode::point_op(upstream, info, HueVsSat { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "hue_vs_sat", display_name: "Hue vs Sat", category: "grading",
    kind: OperationKind::Filter, params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Hue vs Luminance
inventory::submit! { &FilterFactoryRegistration { name: "hue_vs_lum",
    display_name: "Hue vs Lum", category: "grading", params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = hue_curve_from_params(params.get_f32("center"), params.get_f32("amount"), params.get_f32("width"));
        Box::new(FilterNode::point_op(upstream, info, HueVsLum { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "hue_vs_lum", display_name: "Hue vs Lum", category: "grading",
    kind: OperationKind::Filter, params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Lum vs Saturation
inventory::submit! { &FilterFactoryRegistration { name: "lum_vs_sat",
    display_name: "Lum vs Sat", category: "grading", params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, LumVsSat { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "lum_vs_sat", display_name: "Lum vs Sat", category: "grading",
    kind: OperationKind::Filter, params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Sat vs Saturation
inventory::submit! { &FilterFactoryRegistration { name: "sat_vs_sat",
    display_name: "Sat vs Sat", category: "grading", params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, SatVsSat { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "sat_vs_sat", display_name: "Sat vs Sat", category: "grading",
    kind: OperationKind::Filter, params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ASC CDL (with proper params)
inventory::submit! { &FilterFactoryRegistration { name: "asc_cdl",
    display_name: "ASC CDL", category: "grading", params: &ASC_CDL_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        Box::new(FilterNode::point_op(upstream, info, AscCdl {
            slope: [params.get_f32("slope_r"), params.get_f32("slope_g"), params.get_f32("slope_b")],
            offset: [params.get_f32("offset_r"), params.get_f32("offset_g"), params.get_f32("offset_b")],
            power: [params.get_f32("power_r"), params.get_f32("power_g"), params.get_f32("power_b")],
            saturation: params.get_f32("saturation"),
        }))
    },
} }
inventory::submit! { &OperationRegistration { name: "asc_cdl", display_name: "ASC CDL", category: "grading",
    kind: OperationKind::Filter, params: &ASC_CDL_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Split Toning (with hue-based params instead of raw RGB)
inventory::submit! { &FilterFactoryRegistration { name: "split_toning",
    display_name: "Split Toning", category: "grading", params: &SPLIT_TONING_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let sh = hsl_to_rgb_simple(params.get_f32("shadow_hue"), 0.7, 0.3);
        let hh = hsl_to_rgb_simple(params.get_f32("highlight_hue"), 0.7, 0.7);
        Box::new(FilterNode::point_op(upstream, info, SplitToning {
            shadow_color: sh, highlight_color: hh,
            balance: 0.5, strength: (params.get_f32("shadow_strength") + params.get_f32("highlight_strength")) * 0.5,
        }))
    },
} }
inventory::submit! { &OperationRegistration { name: "split_toning", display_name: "Split Toning", category: "grading",
    kind: OperationKind::Filter, params: &SPLIT_TONING_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// Lift/Gamma/Gain (with proper params)
inventory::submit! { &FilterFactoryRegistration { name: "lift_gamma_gain",
    display_name: "Lift/Gamma/Gain", category: "grading", params: &LIFT_GAMMA_GAIN_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        Box::new(FilterNode::point_op(upstream, info, LiftGammaGain {
            lift: [params.get_f32("lift_r"), params.get_f32("lift_g"), params.get_f32("lift_b")],
            gamma: [params.get_f32("gamma_r"), params.get_f32("gamma_g"), params.get_f32("gamma_b")],
            gain: [params.get_f32("gain_r"), params.get_f32("gain_g"), params.get_f32("gain_b")],
        }))
    },
} }
inventory::submit! { &OperationRegistration { name: "lift_gamma_gain", display_name: "Lift/Gamma/Gain", category: "grading",
    kind: OperationKind::Filter, params: &LIFT_GAMMA_GAIN_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        let clut = f.build_clut();
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
        let clut = f.build_clut();
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

    // ── Film Grain ──

    #[test]
    fn film_grain_deterministic() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = FilmGrain {
            amount: 0.3,
            size: 1.0,
            color: false,
            seed: 42,
        };
        let out1 = f.compute(&input, 1, 1).unwrap();
        let out2 = f.compute(&input, 1, 1).unwrap();
        assert_eq!(out1, out2, "Film grain should be deterministic with same seed");
    }

    #[test]
    fn film_grain_zero_amount_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = FilmGrain {
            amount: 0.0,
            size: 1.0,
            color: false,
            seed: 42,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 1e-5, "grain amount=0 identity");
    }

    #[test]
    fn film_grain_preserves_alpha() {
        let input = vec![0.5, 0.5, 0.5, 0.42];
        let f = FilmGrain {
            amount: 0.5,
            size: 1.0,
            color: true,
            seed: 7,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_eq!(out[3], 0.42, "Grain should preserve alpha");
    }

    #[test]
    fn film_grain_gpu_shader_valid() {
        let shader = film_grain_shader();
        assert!(shader.contains("fn splitmix64("), "Shader should contain splitmix64");
        assert!(shader.contains("fn noise_2d("), "Shader should contain noise_2d");
        assert!(shader.contains("@compute @workgroup_size(16, 16, 1)"), "Shader should have workgroup_size");
        assert!(shader.contains("load_pixel"), "Shader should use load_pixel");
        assert!(shader.contains("store_pixel"), "Shader should use store_pixel");
        assert!(shader.contains("params.amount"), "Shader should reference params.amount");
        assert!(shader.contains("params.color_grain"), "Shader should reference color_grain flag");
        assert!(shader.contains("params.seed_lo"), "Shader should use seed_lo/seed_hi");
    }

    #[test]
    fn film_grain_gpu_params_layout() {
        let f = FilmGrain {
            amount: 0.3,
            size: 2.0,
            color: true,
            seed: 99,
        };
        let params = f.params(100, 50);
        assert_eq!(params.len(), 32, "Params should be 8 u32s = 32 bytes");
        let width = u32::from_le_bytes(params[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(params[4..8].try_into().unwrap());
        let amount = f32::from_le_bytes(params[8..12].try_into().unwrap());
        let color_grain = u32::from_le_bytes(params[24..28].try_into().unwrap());
        assert_eq!(width, 100);
        assert_eq!(height, 50);
        assert!((amount - 0.3).abs() < 1e-6);
        assert_eq!(color_grain, 1);
        // seed_lo and seed_hi should be non-zero (XOR'd with SEED_FILM_GRAIN)
        let seed_lo = u32::from_le_bytes(params[16..20].try_into().unwrap());
        let seed_hi = u32::from_le_bytes(params[20..24].try_into().unwrap());
        assert!(seed_lo != 0 || seed_hi != 0, "Seed should be mixed with offset");
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
