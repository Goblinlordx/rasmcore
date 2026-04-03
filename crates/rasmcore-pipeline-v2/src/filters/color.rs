//! Color filters — f32-only color operations.
//!
//! Three categories:
//! 1. **CLUT-compatible**: Per-pixel RGB→RGB transforms that can be baked into
//!    a Clut3D for fusion (hue_rotate, saturate, channel_mixer, etc.)
//! 2. **Image-dependent**: Need full-image statistics (gray world WB, gradient map,
//!    quantize, dither) — not fusable via CLUT.
//! 3. **Spatial**: Need pixel neighborhoods (lab_sharpen, sparse_color).
//!
//! CLUT-compatible filters implement `ClutOp` to expose their Clut3D for the
//! fusion optimizer. The optimizer composes consecutive ClutOps into a single
//! fused CLUT pass.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::noise;
use crate::ops::Filter;

// ─── Color Space Helpers ───────────────────────────────────────────────────

/// RGB [0,1] → HSL (H in [0,360], S in [0,1], L in [0,1]).
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

/// HSL → RGB [0,1].
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

/// RGB [0,1] → CIE Lab (L in [0,100], a,b in ~[-128,127]).
/// Assumes sRGB input → linearize → D65 XYZ → Lab.
fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB → linear
    let rl = srgb_comp_to_linear(r);
    let gl = srgb_comp_to_linear(g);
    let bl = srgb_comp_to_linear(b);
    // Linear RGB → XYZ (D65)
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.119192 * gl + 0.9503041 * bl;
    // XYZ → Lab (D65 white point)
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;
    let fx = lab_f(x / xn);
    let fy = lab_f(y / yn);
    let fz = lab_f(z / zn);
    let l_star = 116.0 * fy - 16.0;
    let a_star = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);
    (l_star, a_star, b_star)
}

/// CIE Lab → RGB [0,1] (sRGB output).
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let xn = 0.95047_f32;
    let yn = 1.0_f32;
    let zn = 1.08883_f32;
    let x = xn * lab_f_inv(fx);
    let y = yn * lab_f_inv(fy);
    let z = zn * lab_f_inv(fz);
    // XYZ → linear RGB
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (linear_to_srgb_comp(rl), linear_to_srgb_comp(gl), linear_to_srgb_comp(bl))
}

fn lab_f(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta * delta * delta {
        t.cbrt()
    } else {
        t / (3.0 * delta * delta) + 4.0 / 29.0
    }
}

fn lab_f_inv(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta {
        t * t * t
    } else {
        3.0 * delta * delta * (t - 4.0 / 29.0)
    }
}

fn srgb_comp_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_comp(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// BT.709 luminance from linear RGB.
fn bt709_luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ─── ClutOp Trait ──────────────────────────────────────────────────────────

/// Color operation that can be represented as a 3D CLUT for fusion.
///
/// Filters implementing this trait can be composed by the fusion optimizer
/// into a single Clut3D pass via `compose_cluts()`.
pub trait ClutOp {
    /// Build a Clut3D representing this color operation.
    /// Grid size 33 is standard (33^3 ≈ 36K entries, good quality/size tradeoff).
    fn build_clut(&self) -> Clut3D;
}

// ─── CLUT-Compatible Filters ───────────────────────────────────────────────

/// Hue rotation in HSL space.
#[derive(Clone)]
pub struct HueRotate {
    /// Rotation in degrees (0-360).
    pub degrees: f32,
}

impl Filter for HueRotate {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let deg = self.degrees;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let nh = (h + deg) % 360.0;
            let (r, g, b) = hsl_to_rgb(if nh < 0.0 { nh + 360.0 } else { nh }, s, l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

impl ClutOp for HueRotate {
    fn build_clut(&self) -> Clut3D {
        let deg = self.degrees;
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let nh = (h + deg) % 360.0;
            hsl_to_rgb(if nh < 0.0 { nh + 360.0 } else { nh }, s, l)
        })
    }
}

/// Saturation adjustment in HSL space.
#[derive(Clone)]
pub struct Saturate {
    /// Factor: 0=grayscale, 1=unchanged, 2=double.
    pub factor: f32,
}

impl Filter for Saturate {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let factor = self.factor;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let (r, g, b) = hsl_to_rgb(h, (s * factor).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

impl ClutOp for Saturate {
    fn build_clut(&self) -> Clut3D {
        let factor = self.factor;
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            hsl_to_rgb(h, (s * factor).clamp(0.0, 1.0), l)
        })
    }
}

/// 3x3 channel mixing matrix in RGB space.
#[derive(Clone)]
pub struct ChannelMixer {
    /// Row-major 3x3: [rr, rg, rb, gr, gg, gb, br, bg, bb].
    pub matrix: [f32; 9],
}

impl Filter for ChannelMixer {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let m = &self.matrix;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            pixel[0] = m[0] * r + m[1] * g + m[2] * b;
            pixel[1] = m[3] * r + m[4] * g + m[5] * b;
            pixel[2] = m[6] * r + m[7] * g + m[8] * b;
        }
        Ok(out)
    }
}

impl ClutOp for ChannelMixer {
    fn build_clut(&self) -> Clut3D {
        let m = self.matrix;
        Clut3D::from_fn(33, move |r, g, b| {
            (
                m[0] * r + m[1] * g + m[2] * b,
                m[3] * r + m[4] * g + m[5] * b,
                m[6] * r + m[7] * g + m[8] * b,
            )
        })
    }
}

/// Vibrance — perceptually-weighted saturation boost.
/// Boosts less-saturated colors more than already-saturated ones.
#[derive(Clone)]
pub struct Vibrance {
    /// Amount: -100 to 100.
    pub amount: f32,
}

impl Filter for Vibrance {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let amt = self.amount / 100.0;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c > 1e-7 {
                (max_c - min_c) / max_c
            } else {
                0.0
            };
            let scale = amt * (1.0 - sat);
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let ns = (s * (1.0 + scale)).clamp(0.0, 1.0);
            let (nr, ng, nb) = hsl_to_rgb(h, ns, l);
            pixel[0] = nr;
            pixel[1] = ng;
            pixel[2] = nb;
        }
        Ok(out)
    }
}

impl ClutOp for Vibrance {
    fn build_clut(&self) -> Clut3D {
        let amt = self.amount / 100.0;
        Clut3D::from_fn(33, move |r, g, b| {
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c > 1e-7 {
                (max_c - min_c) / max_c
            } else {
                0.0
            };
            let scale = amt * (1.0 - sat);
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let ns = (s * (1.0 + scale)).clamp(0.0, 1.0);
            hsl_to_rgb(h, ns, l)
        })
    }
}

/// Sepia tone — warm brownish tint via standard matrix blend.
#[derive(Clone)]
pub struct Sepia {
    /// Intensity: 0=none, 1=full sepia.
    pub intensity: f32,
}

impl Filter for Sepia {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let t = self.intensity;
        let inv = 1.0 - t;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(1.0);
            let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(1.0);
            let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(1.0);
            pixel[0] = inv * r + t * sr;
            pixel[1] = inv * g + t * sg;
            pixel[2] = inv * b + t * sb;
        }
        Ok(out)
    }
}

impl ClutOp for Sepia {
    fn build_clut(&self) -> Clut3D {
        let t = self.intensity;
        let inv = 1.0 - t;
        Clut3D::from_fn(33, move |r, g, b| {
            let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(1.0);
            let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(1.0);
            let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(1.0);
            (inv * r + t * sr, inv * g + t * sg, inv * b + t * sb)
        })
    }
}

/// Colorize — tint image with a target color using W3C luma blend.
#[derive(Clone)]
pub struct Colorize {
    /// Target color RGB [0,1].
    pub target_r: f32,
    pub target_g: f32,
    pub target_b: f32,
    /// Blend amount: 0=none, 1=full.
    pub amount: f32,
}

impl Filter for Colorize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (tr, tg, tb) = (self.target_r, self.target_g, self.target_b);
        let amt = self.amount;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let luma = bt709_luma(pixel[0], pixel[1], pixel[2]);
            pixel[0] = pixel[0] + (luma * tr - pixel[0]) * amt;
            pixel[1] = pixel[1] + (luma * tg - pixel[1]) * amt;
            pixel[2] = pixel[2] + (luma * tb - pixel[2]) * amt;
        }
        Ok(out)
    }
}

impl ClutOp for Colorize {
    fn build_clut(&self) -> Clut3D {
        let (tr, tg, tb, amt) = (self.target_r, self.target_g, self.target_b, self.amount);
        Clut3D::from_fn(33, move |r, g, b| {
            let luma = bt709_luma(r, g, b);
            (
                r + (luma * tr - r) * amt,
                g + (luma * tg - g) * amt,
                b + (luma * tb - b) * amt,
            )
        })
    }
}

/// Modulate — combined brightness/saturation/hue in HSL space.
#[derive(Clone)]
pub struct Modulate {
    /// Brightness factor (1.0=unchanged, 0=black, 2=double).
    pub brightness: f32,
    /// Saturation factor (1.0=unchanged, 0=grayscale).
    pub saturation: f32,
    /// Hue rotation in degrees.
    pub hue: f32,
}

impl Filter for Modulate {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (bri, sat, hue) = (self.brightness, self.saturation, self.hue);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let nl = (l * bri).clamp(0.0, 1.0);
            let ns = (s * sat).clamp(0.0, 1.0);
            let mut nh = (h + hue) % 360.0;
            if nh < 0.0 {
                nh += 360.0;
            }
            let (r, g, b) = hsl_to_rgb(nh, ns, nl);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

impl ClutOp for Modulate {
    fn build_clut(&self) -> Clut3D {
        let (bri, sat, hue) = (self.brightness, self.saturation, self.hue);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let nl = (l * bri).clamp(0.0, 1.0);
            let ns = (s * sat).clamp(0.0, 1.0);
            let mut nh = (h + hue) % 360.0;
            if nh < 0.0 {
                nh += 360.0;
            }
            hsl_to_rgb(nh, ns, nl)
        })
    }
}

/// Photo filter — color overlay with optional luminosity preservation.
#[derive(Clone)]
pub struct PhotoFilter {
    /// Filter color RGB [0,1].
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    /// Density: 0=none, 1=full overlay.
    pub density: f32,
    /// Preserve original luminosity.
    pub preserve_luminosity: bool,
}

impl Filter for PhotoFilter {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (fr, fg, fb) = (self.color_r, self.color_g, self.color_b);
        let d = self.density;
        let preserve = self.preserve_luminosity;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            let mut nr = r + (fr - r) * d;
            let mut ng = g + (fg - g) * d;
            let mut nb = b + (fb - b) * d;
            if preserve {
                let orig_luma = bt709_luma(r, g, b);
                let new_luma = bt709_luma(nr, ng, nb);
                if new_luma > 1e-7 {
                    let scale = orig_luma / new_luma;
                    nr *= scale;
                    ng *= scale;
                    nb *= scale;
                }
            }
            pixel[0] = nr;
            pixel[1] = ng;
            pixel[2] = nb;
        }
        Ok(out)
    }
}

impl ClutOp for PhotoFilter {
    fn build_clut(&self) -> Clut3D {
        let (fr, fg, fb, d, preserve) = (
            self.color_r,
            self.color_g,
            self.color_b,
            self.density,
            self.preserve_luminosity,
        );
        Clut3D::from_fn(33, move |r, g, b| {
            let mut nr = r + (fr - r) * d;
            let mut ng = g + (fg - g) * d;
            let mut nb = b + (fb - b) * d;
            if preserve {
                let orig_luma = bt709_luma(r, g, b);
                let new_luma = bt709_luma(nr, ng, nb);
                if new_luma > 1e-7 {
                    let scale = orig_luma / new_luma;
                    nr *= scale;
                    ng *= scale;
                    nb *= scale;
                }
            }
            (nr, ng, nb)
        })
    }
}

/// Selective color — adjust pixels matching a hue range in HSL space.
#[derive(Clone)]
pub struct SelectiveColor {
    /// Target hue center (0-360).
    pub target_hue: f32,
    /// Hue range (1-180).
    pub hue_range: f32,
    /// Hue shift (-180 to 180).
    pub hue_shift: f32,
    /// Saturation factor.
    pub saturation: f32,
    /// Lightness offset (-1 to 1).
    pub lightness: f32,
}

impl Filter for SelectiveColor {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = selective_color_pixel(
                pixel[0],
                pixel[1],
                pixel[2],
                self.target_hue,
                self.hue_range,
                self.hue_shift,
                self.saturation,
                self.lightness,
            );
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

fn selective_color_pixel(
    r: f32,
    g: f32,
    b: f32,
    target_hue: f32,
    hue_range: f32,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
) -> (f32, f32, f32) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    let half = hue_range * 0.5;
    let mut diff = (h - target_hue).abs();
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    if diff > half {
        return (r, g, b);
    }
    // Cosine taper for smooth falloff
    let weight = 0.5 * (1.0 + (std::f32::consts::PI * diff / half).cos());
    let mut nh = h + hue_shift * weight;
    if nh < 0.0 {
        nh += 360.0;
    }
    if nh >= 360.0 {
        nh -= 360.0;
    }
    let ns = (s * (1.0 + (saturation - 1.0) * weight)).clamp(0.0, 1.0);
    let nl = (l + lightness * weight).clamp(0.0, 1.0);
    hsl_to_rgb(nh, ns, nl)
}

impl ClutOp for SelectiveColor {
    fn build_clut(&self) -> Clut3D {
        let (th, hr, hs, sat, lig) = (
            self.target_hue,
            self.hue_range,
            self.hue_shift,
            self.saturation,
            self.lightness,
        );
        Clut3D::from_fn(33, move |r, g, b| selective_color_pixel(r, g, b, th, hr, hs, sat, lig))
    }
}

/// White balance via color temperature (Kelvin).
#[derive(Clone)]
pub struct WhiteBalanceTemperature {
    /// Temperature in Kelvin (2000-12000). 6500 = daylight neutral.
    pub temperature: f32,
    /// Green-magenta tint (-1 to 1).
    pub tint: f32,
}

impl Filter for WhiteBalanceTemperature {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (sr, sg, sb) = wb_temp_scales(self.temperature, self.tint);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] *= sr;
            pixel[1] *= sg;
            pixel[2] *= sb;
        }
        Ok(out)
    }
}

fn wb_temp_scales(temperature: f32, tint: f32) -> (f32, f32, f32) {
    // Shift relative to D65 (6500K neutral)
    let temp_norm = (temperature - 6500.0) / 6500.0;
    let scale_r = 1.0 + temp_norm * 0.1;
    let scale_b = 1.0 - temp_norm * 0.1;
    let scale_g = 1.0 - tint * 0.1;
    (scale_r, scale_g, scale_b)
}

impl ClutOp for WhiteBalanceTemperature {
    fn build_clut(&self) -> Clut3D {
        let (sr, sg, sb) = wb_temp_scales(self.temperature, self.tint);
        Clut3D::from_fn(33, move |r, g, b| (r * sr, g * sg, b * sb))
    }
}

/// Replace color — select pixels by HSL ranges and shift them.
#[derive(Clone)]
pub struct ReplaceColor {
    /// Center hue (0-360).
    pub center_hue: f32,
    /// Hue range (1-180).
    pub hue_range: f32,
    /// Saturation range [min, max].
    pub sat_min: f32,
    pub sat_max: f32,
    /// Lightness range [min, max].
    pub lum_min: f32,
    pub lum_max: f32,
    /// Shift amounts.
    pub hue_shift: f32,
    pub sat_shift: f32,
    pub lum_shift: f32,
}

impl Filter for ReplaceColor {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = replace_color_pixel(pixel[0], pixel[1], pixel[2], self);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

fn replace_color_pixel(r: f32, g: f32, b: f32, p: &ReplaceColor) -> (f32, f32, f32) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    // Check saturation and lightness ranges
    if s < p.sat_min || s > p.sat_max || l < p.lum_min || l > p.lum_max {
        return (r, g, b);
    }
    // Check hue range with cosine falloff
    let half = p.hue_range * 0.5;
    let mut diff = (h - p.center_hue).abs();
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    if diff > half {
        return (r, g, b);
    }
    let weight = 0.5 * (1.0 + (std::f32::consts::PI * diff / half).cos());
    let mut nh = h + p.hue_shift * weight;
    if nh < 0.0 {
        nh += 360.0;
    }
    if nh >= 360.0 {
        nh -= 360.0;
    }
    let ns = (s + p.sat_shift * weight).clamp(0.0, 1.0);
    let nl = (l + p.lum_shift * weight).clamp(0.0, 1.0);
    hsl_to_rgb(nh, ns, nl)
}

impl ClutOp for ReplaceColor {
    fn build_clut(&self) -> Clut3D {
        let p = self.clone();
        Clut3D::from_fn(33, move |r, g, b| replace_color_pixel(r, g, b, &p))
    }
}

/// Lab adjust — shift a* and b* channels in CIE Lab space.
#[derive(Clone)]
pub struct LabAdjust {
    /// Green-red shift (-128 to 127).
    pub a_offset: f32,
    /// Blue-yellow shift (-128 to 127).
    pub b_offset: f32,
}

impl Filter for LabAdjust {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let (ao, bo) = (self.a_offset, self.b_offset);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (l, a, b) = rgb_to_lab(pixel[0], pixel[1], pixel[2]);
            let na = (a + ao).clamp(-128.0, 127.0);
            let nb = (b + bo).clamp(-128.0, 127.0);
            let (r, g, bi) = lab_to_rgb(l, na, nb);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = bi;
        }
        Ok(out)
    }
}

impl ClutOp for LabAdjust {
    fn build_clut(&self) -> Clut3D {
        let (ao, bo) = (self.a_offset, self.b_offset);
        Clut3D::from_fn(33, move |r, g, b| {
            let (l, a, bi) = rgb_to_lab(r, g, b);
            let na = (a + ao).clamp(-128.0, 127.0);
            let nb = (bi + bo).clamp(-128.0, 127.0);
            lab_to_rgb(l, na, nb)
        })
    }
}

// ─── Image-Dependent Filters ───────────────────────────────────────────────

/// White balance via gray world assumption (automatic).
/// Computes per-channel means and normalizes so all channels have equal mean.
#[derive(Clone)]
pub struct WhiteBalanceGrayWorld;

impl Filter for WhiteBalanceGrayWorld {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let pixel_count = input.len() / 4;
        if pixel_count == 0 {
            return Ok(input.to_vec());
        }
        let (mut sum_r, mut sum_g, mut sum_b) = (0.0f64, 0.0f64, 0.0f64);
        for pixel in input.chunks_exact(4) {
            sum_r += pixel[0] as f64;
            sum_g += pixel[1] as f64;
            sum_b += pixel[2] as f64;
        }
        let n = pixel_count as f64;
        let avg_r = sum_r / n;
        let avg_g = sum_g / n;
        let avg_b = sum_b / n;
        let avg_all = (avg_r + avg_g + avg_b) / 3.0;
        let sr = if avg_r > 1e-10 { (avg_all / avg_r) as f32 } else { 1.0 };
        let sg = if avg_g > 1e-10 { (avg_all / avg_g) as f32 } else { 1.0 };
        let sb = if avg_b > 1e-10 { (avg_all / avg_b) as f32 } else { 1.0 };
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] *= sr;
            pixel[1] *= sg;
            pixel[2] *= sb;
        }
        Ok(out)
    }
}

/// Gradient map — map luminance to a color gradient.
#[derive(Clone)]
pub struct GradientMap {
    /// Gradient stops: (position [0,1], r, g, b) sorted by position.
    pub stops: Vec<(f32, f32, f32, f32)>,
}

impl Filter for GradientMap {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.stops.is_empty() {
            return Ok(input.to_vec());
        }
        // Build 256-entry LUT for fast lookup
        let lut = build_gradient_lut(&self.stops);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let luma = bt709_luma(pixel[0], pixel[1], pixel[2]);
            let idx = (luma * 255.0).round().clamp(0.0, 255.0) as usize;
            pixel[0] = lut[idx * 3];
            pixel[1] = lut[idx * 3 + 1];
            pixel[2] = lut[idx * 3 + 2];
        }
        Ok(out)
    }
}

fn build_gradient_lut(stops: &[(f32, f32, f32, f32)]) -> Vec<f32> {
    let mut lut = vec![0.0f32; 256 * 3];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        let (r, g, b) = interpolate_gradient(stops, t);
        lut[i * 3] = r;
        lut[i * 3 + 1] = g;
        lut[i * 3 + 2] = b;
    }
    lut
}

fn interpolate_gradient(stops: &[(f32, f32, f32, f32)], t: f32) -> (f32, f32, f32) {
    if stops.len() == 1 {
        return (stops[0].1, stops[0].2, stops[0].3);
    }
    if t <= stops[0].0 {
        return (stops[0].1, stops[0].2, stops[0].3);
    }
    let last = stops.len() - 1;
    if t >= stops[last].0 {
        return (stops[last].1, stops[last].2, stops[last].3);
    }
    for i in 0..last {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            let range = stops[i + 1].0 - stops[i].0;
            let frac = if range > 1e-7 {
                (t - stops[i].0) / range
            } else {
                0.0
            };
            return (
                stops[i].1 + frac * (stops[i + 1].1 - stops[i].1),
                stops[i].2 + frac * (stops[i + 1].2 - stops[i].2),
                stops[i].3 + frac * (stops[i + 1].3 - stops[i].3),
            );
        }
    }
    (stops[last].1, stops[last].2, stops[last].3)
}

// ─── Quantization / Dithering Filters ──────────────────────────────────────

/// Median-cut color quantization (no dithering).
#[derive(Clone)]
pub struct Quantize {
    /// Max palette size (2-256).
    pub max_colors: u32,
}

impl Filter for Quantize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = median_cut_palette(input, self.max_colors as usize);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = nearest_color(&palette, pixel[0], pixel[1], pixel[2]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

/// K-means color quantization.
#[derive(Clone)]
pub struct KmeansQuantize {
    /// Number of clusters (2-256).
    pub k: u32,
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Random seed for deterministic initialization.
    pub seed: u32,
}

impl Filter for KmeansQuantize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = kmeans_palette(input, self.k as usize, self.max_iterations, self.seed);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = nearest_color(&palette, pixel[0], pixel[1], pixel[2]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

/// Ordered dithering with Bayer matrix.
#[derive(Clone)]
pub struct DitherOrdered {
    /// Max palette size (2-256).
    pub max_colors: u32,
    /// Bayer matrix size (2, 4, 8, or 16).
    pub map_size: u32,
}

impl Filter for DitherOrdered {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = median_cut_palette(input, self.max_colors as usize);
        let bayer = bayer_matrix(self.map_size);
        let bayer_n = self.map_size as usize;
        let spread = 1.0 / self.max_colors as f32;
        let mut out = input.to_vec();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize * 4;
                let threshold = bayer[(y as usize % bayer_n) * bayer_n + (x as usize % bayer_n)];
                let bias = (threshold - 0.5) * spread;
                let r = out[idx] + bias;
                let g = out[idx + 1] + bias;
                let b = out[idx + 2] + bias;
                let (nr, ng, nb) = nearest_color(&palette, r, g, b);
                out[idx] = nr;
                out[idx + 1] = ng;
                out[idx + 2] = nb;
            }
        }
        Ok(out)
    }
}

/// Floyd-Steinberg error diffusion dithering.
#[derive(Clone)]
pub struct DitherFloydSteinberg {
    /// Max palette size (2-256).
    pub max_colors: u32,
}

impl Filter for DitherFloydSteinberg {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = median_cut_palette(input, self.max_colors as usize);
        let w = width as usize;
        let h = height as usize;
        let mut buf = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let (r, g, b) = (buf[idx], buf[idx + 1], buf[idx + 2]);
                let (nr, ng, nb) = nearest_color(&palette, r, g, b);
                let er = r - nr;
                let eg = g - ng;
                let eb = b - nb;
                buf[idx] = nr;
                buf[idx + 1] = ng;
                buf[idx + 2] = nb;
                // Diffuse error
                let diffuse = |buf: &mut [f32], bx: usize, by: usize, weight: f32| {
                    let i = (by * w + bx) * 4;
                    buf[i] += er * weight;
                    buf[i + 1] += eg * weight;
                    buf[i + 2] += eb * weight;
                };
                if x + 1 < w {
                    diffuse(&mut buf, x + 1, y, 7.0 / 16.0);
                }
                if y + 1 < h {
                    if x > 0 {
                        diffuse(&mut buf, x - 1, y + 1, 3.0 / 16.0);
                    }
                    diffuse(&mut buf, x, y + 1, 5.0 / 16.0);
                    if x + 1 < w {
                        diffuse(&mut buf, x + 1, y + 1, 1.0 / 16.0);
                    }
                }
            }
        }
        Ok(buf)
    }
}

// ─── Palette helpers ───────────────────────────────────────────────────────

fn median_cut_palette(pixels: &[f32], max_colors: usize) -> Vec<(f32, f32, f32)> {
    // Collect unique-ish colors (subsample for speed on large images)
    let pixel_count = pixels.len() / 4;
    let step = (pixel_count / 4096).max(1);
    let mut colors: Vec<[f32; 3]> = Vec::with_capacity(pixel_count / step + 1);
    for i in (0..pixel_count).step_by(step) {
        let idx = i * 4;
        colors.push([pixels[idx], pixels[idx + 1], pixels[idx + 2]]);
    }
    if colors.is_empty() {
        return vec![(0.0, 0.0, 0.0)];
    }
    // Recursive median cut
    let mut buckets = vec![colors];
    while buckets.len() < max_colors {
        // Find bucket with largest range
        let (split_idx, _) = buckets
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ra = color_range(a);
                let rb = color_range(b);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let bucket = buckets.swap_remove(split_idx);
        if bucket.len() < 2 {
            buckets.push(bucket);
            break;
        }
        let channel = widest_channel(&bucket);
        let mut sorted = bucket;
        sorted.sort_by(|a, b| a[channel].partial_cmp(&b[channel]).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        let (left, right) = sorted.split_at(mid);
        buckets.push(left.to_vec());
        buckets.push(right.to_vec());
    }
    buckets
        .iter()
        .map(|b| {
            let n = b.len() as f32;
            let (sr, sg, sb) = b.iter().fold((0.0, 0.0, 0.0), |(sr, sg, sb), c| {
                (sr + c[0], sg + c[1], sb + c[2])
            });
            (sr / n, sg / n, sb / n)
        })
        .collect()
}

fn color_range(colors: &[[f32; 3]]) -> f32 {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for c in colors {
        for i in 0..3 {
            min[i] = min[i].min(c[i]);
            max[i] = max[i].max(c[i]);
        }
    }
    (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2])
}

fn widest_channel(colors: &[[f32; 3]]) -> usize {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for c in colors {
        for i in 0..3 {
            min[i] = min[i].min(c[i]);
            max[i] = max[i].max(c[i]);
        }
    }
    let ranges = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if ranges[0] >= ranges[1] && ranges[0] >= ranges[2] {
        0
    } else if ranges[1] >= ranges[2] {
        1
    } else {
        2
    }
}

fn nearest_color(palette: &[(f32, f32, f32)], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let mut best = palette[0];
    let mut best_dist = f32::MAX;
    for &(pr, pg, pb) in palette {
        let dr = r - pr;
        let dg = g - pg;
        let db = b - pb;
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best = (pr, pg, pb);
        }
    }
    best
}

fn kmeans_palette(
    pixels: &[f32],
    k: usize,
    max_iterations: u32,
    seed: u32,
) -> Vec<(f32, f32, f32)> {
    let pixel_count = pixels.len() / 4;
    if pixel_count == 0 || k == 0 {
        return vec![(0.0, 0.0, 0.0)];
    }
    // Initialize centroids via seeded selection (SplitMix64)
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let km_seed = seed as u64 ^ noise::SEED_KMEANS;
    for i in 0..k {
        let pi = noise::seeded_index(i as u32, km_seed, pixel_count as u32) as usize * 4;
        centroids.push([pixels[pi], pixels[pi + 1], pixels[pi + 2]]);
    }
    // Subsample for speed
    let step = (pixel_count / 8192).max(1);
    let samples: Vec<[f32; 3]> = (0..pixel_count)
        .step_by(step)
        .map(|i| {
            let idx = i * 4;
            [pixels[idx], pixels[idx + 1], pixels[idx + 2]]
        })
        .collect();
    for _ in 0..max_iterations {
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u32; k];
        for s in &samples {
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d = (s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2) + (s[2] - c[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best_c = ci;
                }
            }
            sums[best_c][0] += s[0] as f64;
            sums[best_c][1] += s[1] as f64;
            sums[best_c][2] += s[2] as f64;
            counts[best_c] += 1;
        }
        let mut converged = true;
        for (ci, c) in centroids.iter_mut().enumerate() {
            if counts[ci] > 0 {
                let n = counts[ci] as f64;
                let new = [
                    (sums[ci][0] / n) as f32,
                    (sums[ci][1] / n) as f32,
                    (sums[ci][2] / n) as f32,
                ];
                if (new[0] - c[0]).abs() > 1e-5
                    || (new[1] - c[1]).abs() > 1e-5
                    || (new[2] - c[2]).abs() > 1e-5
                {
                    converged = false;
                }
                *c = new;
            }
        }
        if converged {
            break;
        }
    }
    centroids.iter().map(|c| (c[0], c[1], c[2])).collect()
}

fn bayer_matrix(size: u32) -> Vec<f32> {
    let n = size as usize;
    let mut m = vec![0.0f32; n * n];
    // Recursive Bayer construction
    // Base: 2x2
    if n == 2 {
        m[0] = 0.0 / 4.0;
        m[1] = 2.0 / 4.0;
        m[2] = 3.0 / 4.0;
        m[3] = 1.0 / 4.0;
        return m;
    }
    let half = n / 2;
    let sub = bayer_matrix(half as u32);
    let nn = (n * n) as f32;
    for y in 0..n {
        for x in 0..n {
            let sy = y % half;
            let sx = x % half;
            let base = sub[sy * half + sx] * (half * half) as f32;
            let quadrant = match (y / half, x / half) {
                (0, 0) => 0.0,
                (0, 1) => 2.0,
                (1, 0) => 3.0,
                (1, 1) => 1.0,
                _ => 0.0,
            };
            m[y * n + x] = (4.0 * base + quadrant + 0.5) / nn;
        }
    }
    m
}

// ─── Spatial Color Filters ─────────────────────────────────────────────────

/// Lab sharpen — unsharp mask on L channel only (preserves chrominance).
#[derive(Clone)]
pub struct LabSharpen {
    /// Sharpening strength (0-10).
    pub amount: f32,
    /// Blur radius for unsharp mask.
    pub radius: f32,
}

impl Filter for LabSharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let pixel_count = w * h;
        // Convert to Lab
        let mut l_channel = vec![0.0f32; pixel_count];
        let mut a_channel = vec![0.0f32; pixel_count];
        let mut b_channel = vec![0.0f32; pixel_count];
        for i in 0..pixel_count {
            let idx = i * 4;
            let (l, a, b) = rgb_to_lab(input[idx], input[idx + 1], input[idx + 2]);
            l_channel[i] = l;
            a_channel[i] = a;
            b_channel[i] = b;
        }
        // Gaussian blur the L channel
        let blurred_l = gaussian_blur_1d(&l_channel, w, h, self.radius);
        // Unsharp mask: sharpened = original + amount * (original - blurred)
        for i in 0..pixel_count {
            l_channel[i] += self.amount * (l_channel[i] - blurred_l[i]);
            l_channel[i] = l_channel[i].clamp(0.0, 100.0);
        }
        // Convert back to RGB
        let mut out = input.to_vec();
        for i in 0..pixel_count {
            let idx = i * 4;
            let (r, g, b) = lab_to_rgb(l_channel[i], a_channel[i], b_channel[i]);
            out[idx] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
        }
        Ok(out)
    }
}

/// Simple separable Gaussian blur for a single channel.
fn gaussian_blur_1d(data: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let kernel_radius = (radius * 3.0).ceil() as usize;
    if kernel_radius == 0 {
        return data.to_vec();
    }
    let sigma = radius;
    let kernel_size = kernel_radius * 2 + 1;
    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0f32;
    for (i, kv) in kernel.iter_mut().enumerate() {
        let x = i as f32 - kernel_radius as f32;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        *kv = v;
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    // Horizontal pass
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + k as i32 - kernel_radius as i32)
                    .max(0)
                    .min(w as i32 - 1) as usize;
                acc += data[y * w + sx] * kv;
            }
            temp[y * w + x] = acc;
        }
    }
    // Vertical pass
    let mut result = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + k as i32 - kernel_radius as i32)
                    .max(0)
                    .min(h as i32 - 1) as usize;
                acc += temp[sy * w + x] * kv;
            }
            result[y * w + x] = acc;
        }
    }
    result
}

/// Sparse color interpolation — Shepard (inverse-distance weighted) method.
#[derive(Clone)]
pub struct SparseColor {
    /// Control points: (x, y, r, g, b).
    pub points: Vec<(f32, f32, f32, f32, f32)>,
    /// Inverse-distance power (default 2.0).
    pub power: f32,
}

impl Filter for SparseColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.points.is_empty() {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let power = self.power;
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let px = x as f32;
                let py = y as f32;
                let mut weight_sum = 0.0f32;
                let mut r_sum = 0.0f32;
                let mut g_sum = 0.0f32;
                let mut b_sum = 0.0f32;
                let mut exact = None;
                for &(cx, cy, cr, cg, cb) in &self.points {
                    let dx = px - cx;
                    let dy = py - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < 0.001 {
                        exact = Some((cr, cg, cb));
                        break;
                    }
                    let w = 1.0 / dist.powf(power);
                    weight_sum += w;
                    r_sum += w * cr;
                    g_sum += w * cg;
                    b_sum += w * cb;
                }
                if let Some((r, g, b)) = exact {
                    out[idx] = r;
                    out[idx + 1] = g;
                    out[idx + 2] = b;
                } else if weight_sum > 1e-10 {
                    out[idx] = r_sum / weight_sum;
                    out[idx + 1] = g_sum / weight_sum;
                    out[idx + 2] = b_sum / weight_sum;
                }
            }
        }
        Ok(out)
    }
}

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

    // ── Color space roundtrip tests ──

    #[test]
    fn hsl_roundtrip() {
        let colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.2, 0.6, 0.8),
        ];
        for (r, g, b) in colors {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let (rr, rg, rb) = hsl_to_rgb(h, s, l);
            assert!(
                (r - rr).abs() < 0.01 && (g - rg).abs() < 0.01 && (b - rb).abs() < 0.01,
                "HSL roundtrip failed for ({r}, {g}, {b}): got ({rr}, {rg}, {rb})"
            );
        }
    }

    #[test]
    fn lab_roundtrip() {
        let colors = [
            (0.5, 0.5, 0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.6, 0.8),
        ];
        for (r, g, b) in colors {
            let (l, a, bi) = rgb_to_lab(r, g, b);
            let (rr, rg, rb) = lab_to_rgb(l, a, bi);
            assert!(
                (r - rr).abs() < 0.02 && (g - rg).abs() < 0.02 && (b - rb).abs() < 0.02,
                "Lab roundtrip failed for ({r}, {g}, {b}): got ({rr:.4}, {rg:.4}, {rb:.4})"
            );
        }
    }

    // ── Filter tests ──

    #[test]
    fn hue_rotate_180_inverts_hue() {
        let input = test_pixel(1.0, 0.0, 0.0); // red
        let f = HueRotate { degrees: 180.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // Red rotated 180° = cyan (0, 1, 1)
        assert_rgb_close(&out, (0.0, 1.0, 1.0), 0.02, "hue_rotate 180");
    }

    #[test]
    fn hue_rotate_clut_matches_compute() {
        let f = HueRotate { degrees: 90.0 };
        let input = test_pixel(0.8, 0.2, 0.4);
        let computed = f.compute(&input, 1, 1).unwrap();
        let clut = f.build_clut();
        let (cr, cg, cb) = clut.sample(0.8, 0.2, 0.4);
        assert!(
            (computed[0] - cr).abs() < 0.05
                && (computed[1] - cg).abs() < 0.05
                && (computed[2] - cb).abs() < 0.05,
            "CLUT mismatch: compute=({:.3},{:.3},{:.3}) clut=({cr:.3},{cg:.3},{cb:.3})",
            computed[0],
            computed[1],
            computed[2]
        );
    }

    #[test]
    fn saturate_zero_is_grayscale() {
        let input = test_pixel(0.8, 0.2, 0.4);
        let f = Saturate { factor: 0.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // All channels should be equal (grayscale)
        assert!(
            (out[0] - out[1]).abs() < 0.02 && (out[1] - out[2]).abs() < 0.02,
            "Expected grayscale, got ({:.3}, {:.3}, {:.3})",
            out[0],
            out[1],
            out[2]
        );
    }

    #[test]
    fn channel_mixer_identity() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = ChannelMixer {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 1e-6, "identity mixer");
    }

    #[test]
    fn channel_mixer_swap_rb() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = ChannelMixer {
            matrix: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.7, 0.5, 0.3), 1e-6, "swap R<->B");
    }

    #[test]
    fn vibrance_neutral_on_saturated() {
        let input = test_pixel(1.0, 0.0, 0.0); // fully saturated red
        let f = Vibrance { amount: 50.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // Should change minimally (already saturated)
        assert!(
            (out[0] - 1.0).abs() < 0.1,
            "Vibrance should barely affect saturated color"
        );
    }

    #[test]
    fn sepia_full_intensity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = Sepia { intensity: 1.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        // Sepia: R > G > B
        assert!(out[0] > out[1] && out[1] > out[2], "Sepia should be warm-toned");
    }

    #[test]
    fn sepia_zero_is_identity() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = Sepia { intensity: 0.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 1e-6, "sepia 0");
    }

    #[test]
    fn colorize_preserves_structure() {
        let input = test_pixel(0.2, 0.5, 0.8);
        let f = Colorize {
            target_r: 1.0,
            target_g: 0.8,
            target_b: 0.2,
            amount: 0.5,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[3] == 1.0, "Alpha preserved");
    }

    #[test]
    fn modulate_identity() {
        let input = test_pixel(0.5, 0.3, 0.7);
        let f = Modulate {
            brightness: 1.0,
            saturation: 1.0,
            hue: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.3, 0.7), 0.02, "modulate identity");
    }

    #[test]
    fn photo_filter_zero_density_is_identity() {
        let input = test_pixel(0.3, 0.5, 0.7);
        let f = PhotoFilter {
            color_r: 1.0,
            color_g: 0.0,
            color_b: 0.0,
            density: 0.0,
            preserve_luminosity: false,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.3, 0.5, 0.7), 1e-6, "photo_filter 0 density");
    }

    #[test]
    fn wb_temp_neutral_is_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = WhiteBalanceTemperature {
            temperature: 6500.0,
            tint: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 1e-6, "wb 6500K neutral");
    }

    #[test]
    fn wb_warm_increases_red() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = WhiteBalanceTemperature {
            temperature: 8000.0,
            tint: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] > 0.5, "Warm WB should increase red");
        assert!(out[2] < 0.5, "Warm WB should decrease blue");
    }

    #[test]
    fn gray_world_equalizes_channels() {
        // Image with blue cast: R low, G medium, B high
        let input = vec![
            0.2, 0.4, 0.8, 1.0, 0.3, 0.5, 0.9, 1.0, 0.1, 0.3, 0.7, 1.0, 0.2, 0.4, 0.8, 1.0,
        ];
        let f = WhiteBalanceGrayWorld;
        let out = f.compute(&input, 2, 2).unwrap();
        // After gray world, channel means should be closer
        let avg_r: f32 = (0..4).map(|i| out[i * 4]).sum::<f32>() / 4.0;
        let avg_g: f32 = (0..4).map(|i| out[i * 4 + 1]).sum::<f32>() / 4.0;
        let avg_b: f32 = (0..4).map(|i| out[i * 4 + 2]).sum::<f32>() / 4.0;
        let spread = (avg_r - avg_g).abs().max((avg_g - avg_b).abs());
        assert!(spread < 0.01, "Gray world should equalize channels, spread={spread}");
    }

    #[test]
    fn gradient_map_bw() {
        // Black → white gradient: should map luma to position
        let stops = vec![(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)];
        let f = GradientMap { stops };
        let input = test_pixel(0.5, 0.5, 0.5);
        let out = f.compute(&input, 1, 1).unwrap();
        let luma = bt709_luma(0.5, 0.5, 0.5);
        assert_rgb_close(&out, (luma, luma, luma), 0.02, "gradient bw");
    }

    #[test]
    fn quantize_reduces_colors() {
        let mut input = Vec::new();
        for i in 0..100 {
            let v = i as f32 / 99.0;
            input.extend_from_slice(&[v, v * 0.5, 1.0 - v, 1.0]);
        }
        let f = Quantize { max_colors: 4 };
        let out = f.compute(&input, 10, 10).unwrap();
        // Count unique colors
        let mut unique = std::collections::HashSet::new();
        for pixel in out.chunks_exact(4) {
            let key = (
                (pixel[0] * 1000.0) as i32,
                (pixel[1] * 1000.0) as i32,
                (pixel[2] * 1000.0) as i32,
            );
            unique.insert(key);
        }
        assert!(unique.len() <= 4, "Should have <=4 colors, got {}", unique.len());
    }

    #[test]
    fn dither_floyd_steinberg_reduces_colors() {
        let mut input = Vec::new();
        for i in 0..64 {
            let v = i as f32 / 63.0;
            input.extend_from_slice(&[v, v, v, 1.0]);
        }
        let f = DitherFloydSteinberg { max_colors: 2 };
        let out = f.compute(&input, 8, 8).unwrap();
        let mut unique = std::collections::HashSet::new();
        for pixel in out.chunks_exact(4) {
            let key = (
                (pixel[0] * 1000.0) as i32,
                (pixel[1] * 1000.0) as i32,
                (pixel[2] * 1000.0) as i32,
            );
            unique.insert(key);
        }
        assert!(unique.len() <= 2, "Should have <=2 colors, got {}", unique.len());
    }

    #[test]
    fn lab_adjust_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = LabAdjust {
            a_offset: 0.0,
            b_offset: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 0.02, "lab_adjust identity");
    }

    #[test]
    fn lab_sharpen_preserves_flat() {
        // Flat gray image — sharpening should have no effect
        let input: Vec<f32> = (0..16).flat_map(|_| vec![0.5, 0.5, 0.5, 1.0]).collect();
        let f = LabSharpen {
            amount: 2.0,
            radius: 1.0,
        };
        let out = f.compute(&input, 4, 4).unwrap();
        for pixel in out.chunks_exact(4) {
            assert!(
                (pixel[0] - 0.5).abs() < 0.02
                    && (pixel[1] - 0.5).abs() < 0.02
                    && (pixel[2] - 0.5).abs() < 0.02,
                "Flat image should be unchanged after sharpening"
            );
        }
    }

    #[test]
    fn sparse_color_exact_point() {
        let points = vec![(0.0, 0.0, 1.0, 0.0, 0.0)]; // top-left = red
        let f = SparseColor {
            points,
            power: 2.0,
        };
        let input = test_pixel(0.5, 0.5, 0.5);
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (1.0, 0.0, 0.0), 0.01, "sparse exact point");
    }

    #[test]
    fn replace_color_no_match() {
        // Gray pixel (no hue) should not be affected by hue-targeted replacement
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = ReplaceColor {
            center_hue: 0.0,
            hue_range: 30.0,
            sat_min: 0.5,
            sat_max: 1.0,
            lum_min: 0.0,
            lum_max: 1.0,
            hue_shift: 90.0,
            sat_shift: 0.0,
            lum_shift: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 0.01, "replace_color no match");
    }

    #[test]
    fn selective_color_out_of_range() {
        // Blue pixel, targeting red hue — should be unchanged
        let input = test_pixel(0.0, 0.0, 1.0);
        let f = SelectiveColor {
            target_hue: 0.0,
            hue_range: 30.0,
            hue_shift: 90.0,
            saturation: 2.0,
            lightness: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert_rgb_close(&out, (0.0, 0.0, 1.0), 0.02, "selective_color no match");
    }

    #[test]
    fn alpha_preserved_all_filters() {
        let input = vec![0.3, 0.5, 0.7, 0.42]; // weird alpha
        let filters: Vec<Box<dyn Filter>> = vec![
            Box::new(HueRotate { degrees: 45.0 }),
            Box::new(Saturate { factor: 0.5 }),
            Box::new(ChannelMixer {
                matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            }),
            Box::new(Vibrance { amount: 50.0 }),
            Box::new(Sepia { intensity: 0.5 }),
            Box::new(Colorize {
                target_r: 1.0,
                target_g: 0.8,
                target_b: 0.2,
                amount: 0.5,
            }),
            Box::new(Modulate {
                brightness: 1.0,
                saturation: 1.0,
                hue: 0.0,
            }),
            Box::new(PhotoFilter {
                color_r: 1.0,
                color_g: 0.0,
                color_b: 0.0,
                density: 0.5,
                preserve_luminosity: true,
            }),
            Box::new(WhiteBalanceTemperature {
                temperature: 7000.0,
                tint: 0.0,
            }),
            Box::new(LabAdjust {
                a_offset: 10.0,
                b_offset: -5.0,
            }),
            Box::new(Invert),
        ];
        for (i, f) in filters.iter().enumerate() {
            let out = f.compute(&input, 1, 1).unwrap();
            assert_eq!(out[3], 0.42, "Filter {i} should preserve alpha");
        }
    }

    // Use Invert from adjustment module for the alpha test
    use crate::filters::adjustment::Invert;

    #[test]
    fn bayer_matrix_2x2() {
        let m = bayer_matrix(2);
        assert_eq!(m.len(), 4);
        // Values should sum to ~2.0 (each normalized to [0,1))
        let sum: f32 = m.iter().sum();
        assert!((sum - 1.5).abs() < 0.01, "Bayer 2x2 sum: {sum}");
    }

    #[test]
    fn bayer_matrix_4x4() {
        let m = bayer_matrix(4);
        assert_eq!(m.len(), 16);
        // All values should be in [0, 1)
        for v in &m {
            assert!(*v >= 0.0 && *v < 1.0, "Bayer value out of range: {v}");
        }
    }
}
