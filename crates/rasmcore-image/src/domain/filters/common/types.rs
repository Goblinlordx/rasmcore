//! Types helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Kernel shape for bokeh (lens) blur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BokehShape {
    /// Uniform circular disc kernel.
    Disc,
    /// Uniform regular hexagonal kernel (simulates 6-blade aperture).
    Hexagon,
}

/// Supported blend modes for compositing operations.
///
/// All formulas operate on normalized \[0, 1\] channel values where
/// `a` is the foreground (top) layer and `b` is the background (bottom).
/// Results are clamped to \[0, 1\] before conversion back to u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// `a * b` — darkens; validated vs vips (+/-1).
    Multiply,
    /// `1 - (1-a)(1-b)` — lightens; validated vs vips (+/-1).
    Screen,
    /// Multiply if b < 0.5, Screen otherwise; validated vs vips (+/-1).
    Overlay,
    /// `min(a, b)`; validated vs vips (exact).
    Darken,
    /// `max(a, b)`; validated vs vips (exact).
    Lighten,
    /// W3C spec formula (not Photoshop legacy); validated vs vips (+/-1).
    SoftLight,
    /// Overlay with layers swapped; validated vs vips (+/-1).
    HardLight,
    /// `|a - b|`; validated vs vips (+/-1).
    Difference,
    /// `a + b - 2ab`; validated vs vips (+/-1).
    Exclusion,
    /// `b / (1 - a)`, clamped; validated vs vips + ImageMagick (+/-1).
    ColorDodge,
    /// `1 - (1-b) / a`, clamped; validated vs vips + ImageMagick (+/-1).
    ColorBurn,
    /// ColorBurn for a < 0.5, ColorDodge for a >= 0.5; validated vs ImageMagick (exact).
    VividLight,
    /// `a + b` (Add), clamped; validated vs ImageMagick (exact).
    LinearDodge,
    /// `a + b - 1`, clamped; validated vs ImageMagick (exact).
    LinearBurn,
    /// `b + 2a - 1`, clamped; validated vs ImageMagick (exact).
    LinearLight,
    /// `min(b, 2a)` if a <= 0.5, `max(b, 2a-1)` otherwise; validated vs ImageMagick (exact).
    PinLight,
    /// Threshold VividLight result at 0.5; validated vs ImageMagick (exact).
    HardMix,
    /// `b - a`, clamped; validated vs ImageMagick (exact).
    Subtract,
    /// `b / a`, clamped; validated vs ImageMagick (exact).
    Divide,
    /// Probabilistic pixel selection based on alpha/hash — NOT per-channel.
    Dissolve,
    /// Select entire pixel with lower luminance; NOT per-channel.
    DarkerColor,
    /// Select entire pixel with higher luminance; NOT per-channel.
    LighterColor,
    /// Take hue from top layer, sat+lum from bottom; NOT per-channel (HSL).
    Hue,
    /// Take saturation from top layer, hue+lum from bottom; NOT per-channel (HSL).
    Saturation,
    /// Take hue+saturation from top layer, luminosity from bottom; NOT per-channel (HSL).
    Color,
    /// Take luminosity from top layer, hue+sat from bottom; NOT per-channel (HSL).
    Luminosity,
}

/// Structuring element shape for morphological operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphShape {
    /// Rectangle (all ones).
    Rect,
    /// Ellipse inscribed in the kernel rectangle.
    Ellipse,
    /// Cross (horizontal + vertical lines through center).
    Cross,
}

/// NLM algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NlmAlgorithm {
    /// Match OpenCV's `fastNlMeansDenoising` exactly — integer SSD, bit-shift
    /// average, precomputed weight LUT, fixed-point accumulation. Default.
    #[default]
    OpenCv,
    /// Classic Buades et al. 2005 — float SSD, exp() weights.
    Classic,
}

/// Adaptive threshold modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMethod {
    /// Use mean of the block neighborhood.
    Mean,
    /// Use Gaussian-weighted mean of the block neighborhood.
    Gaussian,
}

/// Overlap mode for distortion source rect computation.
pub enum DistortionOverlap {
    /// Constant overlap in pixels (wave, ripple -- max displacement is known).
    Uniform(u32),
    /// Full image needed (polar/depolar/swirl/spherize/barrel -- any pixel can
    /// map far from its source position).
    FullImage,
}

/// Sampling mode for the distortion engine.
pub enum DistortionSampling {
    /// EWA resampling with Robidoux filter (most distortions).
    Ewa,
    /// EWA with edge-clamped border (barrel distortion).
    EwaClamp,
    /// Bilinear interpolation (wave -- matches IM's effect.c).
    Bilinear,
}

pub const INTER_BITS: i32 = 5;

pub const INTER_TAB_SIZE: i32 = 1 << INTER_BITS; // 32
pub const INTER_REMAP_COEF_BITS: i32 = 15;

pub const INTER_REMAP_COEF_SCALE: i32 = 1 << INTER_REMAP_COEF_BITS; // 32768

/// Build the 2D bilinear weight table (1024 entries of 4 i32 weights).
///
/// Matches OpenCV's initInterTab2D for ksize=2 (bilinear). OpenCV stores
/// weights as i16 but we use i32 to avoid overflow at the (0,0) entry
/// where the weight is exactly INTER_REMAP_COEF_SCALE (32768). The
/// interpolation math is identical — only the storage type differs.
/// Public wrapper for integration tests.
pub fn build_bilinear_tab_public() -> Vec<[i32; 4]> {
    build_bilinear_tab()
}

pub fn build_bilinear_tab() -> Vec<[i32; 4]> {
    let sz = (INTER_TAB_SIZE * INTER_TAB_SIZE) as usize; // 1024
    let mut itab = vec![[0i32; 4]; sz];

    for iy in 0..INTER_TAB_SIZE {
        let fy = iy as f32 / INTER_TAB_SIZE as f32;
        for ix in 0..INTER_TAB_SIZE {
            let fx = ix as f32 / INTER_TAB_SIZE as f32;
            let idx = (iy * INTER_TAB_SIZE + ix) as usize;

            let fw = [
                (1.0 - fy) * (1.0 - fx),
                (1.0 - fy) * fx,
                fy * (1.0 - fx),
                fy * fx,
            ];

            for k in 0..4 {
                itab[idx][k] = (fw[k] * INTER_REMAP_COEF_SCALE as f32).round() as i32;
            }

            // Fix rounding: ensure weights sum to exactly INTER_REMAP_COEF_SCALE
            let isum: i32 = itab[idx].iter().sum();
            let diff = isum - INTER_REMAP_COEF_SCALE;
            if diff != 0 {
                if diff < 0 {
                    let max_idx = itab[idx]
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &w)| w)
                        .unwrap()
                        .0;
                    itab[idx][max_idx] -= diff;
                } else {
                    let min_idx = itab[idx]
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, &w)| w)
                        .unwrap()
                        .0;
                    itab[idx][min_idx] -= diff;
                }
            }
        }
    }
    itab
}

/// Gradient map parameters (stops passed as string, not via ConfigParams).
pub struct GradientMapParams {
    /// Gradient color stops as "pos:RRGGBB,pos:RRGGBB,...".
    pub stops: String,
}

/// A detected line segment from Hough transform: (x1, y1, x2, y2).
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// Multiply-With-Carry PRNG matching OpenCV's cv::RNG.
/// Seeded with `0xFFFF_FFFF_FFFF_FFFF` by default (OpenCV convention).
pub struct CvRng {
    state: u64,
}

impl CvRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { u64::MAX } else { seed },
        }
    }

    /// Advance state and return next u32.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(4164903690)
            .wrapping_add(self.state >> 32);
        self.state as u32
    }

    /// Uniform random in [0, upper). Matches OpenCV RNG::uniform(0, upper).
    pub fn uniform(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            return 0;
        }
        self.next_u32() % upper
    }
}

/// Non-local means denoising parameters.
#[derive(Debug, Clone, Copy)]
pub struct NlmParams {
    /// Filter strength (h). Higher = more denoising. Default: 10.0.
    pub h: f32,
    /// Patch size (must be odd). Default: 7.
    pub patch_size: u32,
    /// Search window size (must be odd). Default: 21.
    pub search_size: u32,
    /// Algorithm variant. Default: OpenCv.
    pub algorithm: NlmAlgorithm,
}

impl Default for NlmParams {
    fn default() -> Self {
        Self {
            h: 10.0,
            patch_size: 7,
            search_size: 21,
            algorithm: NlmAlgorithm::OpenCv,
        }
    }
}

/// Parameters for Mertens exposure fusion.
#[derive(Debug, Clone)]
pub struct MertensParams {
    /// Weight for contrast metric (default 1.0).
    pub contrast_weight: f32,
    /// Weight for saturation metric (default 1.0).
    pub saturation_weight: f32,
    /// Weight for well-exposedness metric (default 1.0).
    pub exposure_weight: f32,
}

impl Default for MertensParams {
    fn default() -> Self {
        Self {
            contrast_weight: 1.0,
            saturation_weight: 1.0,
            exposure_weight: 1.0,
        }
    }
}

/// Parameters for Debevec HDR merge.
#[derive(Debug, Clone)]
pub struct DebevecParams {
    /// Number of sample pixels (default 70).
    pub samples: usize,
    /// Smoothness regularization lambda (default 10.0).
    pub lambda: f32,
}

impl Default for DebevecParams {
    fn default() -> Self {
        Self {
            samples: 70,
            lambda: 10.0,
        }
    }
}

/// cvRound: round to nearest, ties to even (matches C lrint/rint).
#[inline]
pub fn cv_round(v: f64) -> i32 {
    // Rust's f64::round() rounds ties away from zero (0.5 → 1), but OpenCV
    // uses C lrint which rounds ties to even. For the coordinate ranges we
    // deal with, the difference is negligible on half-integer values. We use
    // round-half-to-even for exact parity.
    let r = v.round();
    // Check if exactly at a .5 boundary
    if (v - (v.floor() + 0.5)).abs() < 1e-15 {
        let fl = v.floor() as i64;
        if fl & 1 == 0 {
            fl as i32 // even floor → round down
        } else {
            (fl + 1) as i32 // odd floor → round up
        }
    } else {
        r as i32
    }
}

#[inline]
pub fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn lerp_f32(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}

/// Compute BT.601 luminance from 0–255 RGB values.
#[inline]
pub fn pixel_luminance(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// sRGB gamma to linear (f32 version for inline use).
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert normalized [0,1] RGB to (u8, u8, u8).
#[inline]
pub fn to_u8_triple(r: f32, g: f32, b: f32) -> (u8, u8, u8) {
    (
        (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    )
}

