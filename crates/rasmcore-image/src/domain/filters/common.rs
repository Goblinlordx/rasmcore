//! Shared helper functions for filters.
//!
//! Pure utility functions with no ConfigParams dependencies.
//! Used by individual filter files via `use crate::domain::filters::common::*;`

pub use crate::domain::error::ImageError;
pub use crate::domain::types::{ImageInfo, PixelFormat, ColorSpace, DecodedImage};
pub use rasmcore_pipeline::Rect;
pub use crate::domain::point_ops::LutPointOp;
pub use crate::domain::color_lut::{ColorLut3D, ColorLutOp, ColorOp, DEFAULT_CLUT_GRID};

/// Upstream pixel request function.
pub type UpstreamFn<'a> = dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + 'a;

pub fn validate_format(format: PixelFormat) -> Result<(), ImageError> {
    match format {
        PixelFormat::Rgb8
        | PixelFormat::Rgba8
        | PixelFormat::Gray8
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Gray16 => Ok(()),
        other => Err(ImageError::UnsupportedFormat(format!(
            "filter on {other:?} not supported"
        ))),
    }
}

/// Check if a pixel format is 16-bit.
pub fn is_16bit(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
    )
}

/// Number of channels for a pixel format (not bytes — channels).
pub fn channels(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 | PixelFormat::Gray16 => 1,
        PixelFormat::Rgb8 | PixelFormat::Rgb16 => 3,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => 4,
        _ => 3,
    }
}

// ── 16-bit I/O helpers ─────────────────────────────────────────────────────

/// Read u16 samples from a byte buffer (little-endian).
pub fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Write u16 samples to a byte buffer (little-endian).
pub fn u16_to_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Convert 16-bit pixel buffer to f32 normalized [0.0, 1.0] per sample.
pub fn u16_pixels_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes_to_u16(bytes)
        .into_iter()
        .map(|v| v as f32 / 65535.0)
        .collect()
}

/// Convert f32 normalized [0.0, 1.0] samples back to 16-bit pixel buffer.
pub fn f32_to_u16_pixels(values: &[f32]) -> Vec<u8> {
    let u16s: Vec<u16> = values
        .iter()
        .map(|&v| (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16)
        .collect();
    u16_to_bytes(&u16s)
}

/// Convert 16-bit pixel buffer to 8-bit for processing, then back to 16-bit.
/// Used when an operation only supports 8-bit internally (e.g., convolve).
pub fn process_via_8bit<F>(pixels: &[u8], info: &ImageInfo, f: F) -> Result<Vec<u8>, ImageError>
where
    F: FnOnce(&[u8], &ImageInfo) -> Result<Vec<u8>, ImageError>,
{
    let _ch = channels(info.format);
    let samples = bytes_to_u16(pixels);

    // Downscale to 8-bit
    let pixels_8: Vec<u8> = samples.iter().map(|&v| (v >> 8) as u8).collect();

    let info_8 = ImageInfo {
        format: match info.format {
            PixelFormat::Rgb16 => PixelFormat::Rgb8,
            PixelFormat::Rgba16 => PixelFormat::Rgba8,
            PixelFormat::Gray16 => PixelFormat::Gray8,
            other => other,
        },
        ..*info
    };

    let result_8 = f(&pixels_8, &info_8)?;

    // Upscale back to 16-bit, preserving the original high bits where possible
    // Use linear interpolation: result_16 = result_8 * 257 (maps 0-255 → 0-65535)
    let result_16: Vec<u16> = result_8.iter().map(|&v| v as u16 * 257).collect();
    Ok(u16_to_bytes(&result_16))
}

// ─── Spatial overlap crop helper ─────────────────────────────────────────────

/// Crop a pixel buffer from an expanded region back to the original request.
/// Used by spatial filters that expand their upstream request for overlap.
pub fn crop_to_request(
    filtered: &[u8],
    expanded: Rect,
    request: Rect,
    format: PixelFormat,
) -> Vec<u8> {
    if expanded == request {
        return filtered.to_vec();
    }
    let bpp = crate::domain::types::bytes_per_pixel(format) as usize;
    let src_stride = expanded.width as usize * bpp;
    let dst_stride = request.width as usize * bpp;
    let x_off = (request.x - expanded.x) as usize * bpp;
    let y_off = (request.y - expanded.y) as usize;
    let mut out = Vec::with_capacity(request.height as usize * dst_stride);
    for row in 0..request.height as usize {
        let start = (y_off + row) * src_stride + x_off;
        out.extend_from_slice(&filtered[start..start + dst_stride]);
    }
    out
}

// Re-export parent module items (ConfigParams, enums, remaining helpers)
#[allow(unused_imports)]
pub use super::*;

// ─── Enums (moved from mod.rs) ───

// ─── Shared Enums ───────────────────────────────────────────────────────

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


// ─── Constants (moved from mod.rs) ───

// ─── Shared Constants ───────────────────────────────────────────────────

    pub const EMBOSS: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];

    pub const EDGE_ENHANCE: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

    pub const SHARPEN_3X3: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

    pub const BOX_BLUR_3X3: [f32; 9] = [1.0; 9];

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

pub const SIMPLEX_STRETCH: f64 = -0.211324865405187; // (1/sqrt(3) - 1) / 2
pub const SIMPLEX_SQUISH: f64 = 0.366025403784439; // (sqrt(3) - 1) / 2

/// Gradient table for OpenSimplex 2D (8 directions).
pub const SIMPLEX_GRADS: [(f64, f64); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
];

pub const PYR_KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];


// ─── Structs (moved from mod.rs) ───

// ─── Shared Structs ─────────────────────────────────────────────────────

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


// ─── Helpers (moved from mod.rs) ───

// ─── Shared Helper Functions ────────────────────────────────────────────

/// Adaptive threshold — per-pixel threshold computed from local neighborhood.
///
/// For each pixel, the threshold is the (mean or Gaussian-weighted mean) of a
/// block_size × block_size neighborhood, minus constant C.
///
/// Reference: OpenCV cv2.adaptiveThreshold.
pub fn adaptive_threshold(
    pixels: &[u8],
    info: &ImageInfo,
    max_value: u8,
    method: AdaptiveMethod,
    block_size: u32,
    c: f64,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "adaptive threshold requires Gray8 input".into(),
        ));
    }
    if block_size.is_multiple_of(2) || block_size < 3 {
        return Err(ImageError::InvalidParameters(
            "block_size must be odd and >= 3".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let r = (block_size / 2) as usize;

    match method {
        AdaptiveMethod::Mean => {
            // Integer box mean via integral image — BORDER_REPLICATE (matches OpenCV exactly).
            // OpenCV adaptiveThreshold uses: boxFilter(src, mean, CV_8U, ..., BORDER_REPLICATE)
            // then: pixel > (mean - C) in integer arithmetic.
            let box_mean = box_mean_u8_replicate(pixels, w, h, r);
            let c_int = c.round() as i16;
            let mut result = vec![0u8; w * h];
            for i in 0..(w * h) {
                // Signed comparison — threshold can be negative (OpenCV does NOT clamp to 0)
                let thresh = box_mean[i] as i16 - c_int;
                result[i] = if (pixels[i] as i16) > thresh {
                    max_value
                } else {
                    0
                };
            }
            Ok(result)
        }
        AdaptiveMethod::Gaussian => {
            // Gaussian-weighted mean — use separable Gaussian blur
            let sigma = 0.3 * ((block_size as f64 - 1.0) * 0.5 - 1.0) + 0.8;
            let local_mean = gaussian_blur_f64(pixels, w, h, block_size as usize, sigma);
            let mut result = vec![0u8; w * h];
            for i in 0..(w * h) {
                let thresh = local_mean[i] - c;
                result[i] = if (pixels[i] as f64) > thresh {
                    max_value
                } else {
                    0
                };
            }
            Ok(result)
        }
    }
}

/// Apply a ColorOp to a pixel buffer via direct per-pixel evaluation.
///
/// No CLUT allocation — evaluates ColorOp::apply() on each pixel's
/// normalized (R,G,B). For pipeline use, ColorOpNode builds a CLUT instead.
pub fn apply_color_op(pixels: &[u8], info: &ImageInfo, op: &ColorOp) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if info.format == PixelFormat::Gray8 || info.format == PixelFormat::Gray16 {
        return Ok(pixels.to_vec());
    }

    // 16-bit color operations: work in f32 [0,1] range
    if is_16bit(info.format) {
        let ch = channels(info.format);
        let samples = bytes_to_u16(pixels);
        let mut result_u16 = samples.clone();
        for chunk in result_u16.chunks_exact_mut(ch) {
            let (r, g, b) = op.apply(
                chunk[0] as f32 / 65535.0,
                chunk[1] as f32 / 65535.0,
                chunk[2] as f32 / 65535.0,
            );
            chunk[0] = (r * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            chunk[1] = (g * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            chunk[2] = (b * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        }
        return Ok(u16_to_bytes(&result_u16));
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let (r, g, b) = op.apply(
            chunk[0] as f32 / 255.0,
            chunk[1] as f32 / 255.0,
            chunk[2] as f32 / 255.0,
        );
        chunk[0] = (r * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        chunk[1] = (g * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        chunk[2] = (b * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    Ok(result)
}

/// Unified distortion engine. Each filter provides its inverse transform
/// and Jacobian as closures. This function handles:
/// - Computing required source rect (bounding box or uniform overlap)
/// - Requesting expanded upstream pixels
/// - Running the sampling loop in image-space coordinates (not tile-local)
/// - Producing output for the requested tile
pub fn apply_distortion(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo, // FULL image dimensions -- never overwrite with tile dims
    overlap: DistortionOverlap,
    sampling: DistortionSampling,
    // inverse_fn: (output_x, output_y) -> (source_x, source_y) in image space
    inverse_fn: &dyn Fn(f32, f32) -> (f32, f32),
    // jacobian_fn: (output_x, output_y) -> 2x2 Jacobian matrix
    // Ignored for Bilinear sampling mode.
    jacobian_fn: &dyn Fn(f32, f32) -> crate::domain::ewa::Jacobian,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width;
    let h = info.height;

    // Compute required source rect
    let source_rect = match overlap {
        DistortionOverlap::Uniform(px) => request.expand_uniform(px, w, h),
        DistortionOverlap::FullImage => Rect::new(0, 0, w, h),
    };

    let pixels = upstream(source_rect)?;
    let ch = channels(info.format);
    let src_w = source_rect.width as usize;
    let src_h = source_rect.height as usize;

    let sampler = crate::domain::ewa::EwaSampler::new(&pixels, src_w, src_h, ch);

    // Output buffer for the requested tile only
    let out_w = request.width as usize;
    let out_h = request.height as usize;
    let mut out = vec![0u8; out_w * out_h * ch];

    // Iterate in IMAGE-SPACE coordinates (not tile-local)
    for oy in 0..out_h {
        let img_y = (request.y as usize + oy) as f32;
        for ox in 0..out_w {
            let img_x = (request.x as usize + ox) as f32;

            // Inverse transform in image space
            let (sx, sy) = inverse_fn(img_x, img_y);

            // Adjust source coords to be relative to source_rect
            let local_sx = sx - source_rect.x as f32;
            let local_sy = sy - source_rect.y as f32;

            let off = (oy * out_w + ox) * ch;
            match sampling {
                DistortionSampling::Bilinear => {
                    for c in 0..ch {
                        out[off + c] = sampler
                            .bilinear_pub(local_sx, local_sy, c)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                }
                DistortionSampling::Ewa => {
                    let j = jacobian_fn(img_x, img_y);
                    for c in 0..ch {
                        out[off + c] = sampler
                            .sample(local_sx, local_sy, &j, c)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                }
                DistortionSampling::EwaClamp => {
                    let j = jacobian_fn(img_x, img_y);
                    for c in 0..ch {
                        out[off + c] = sampler
                            .sample_clamp(local_sx, local_sy, &j, c)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Approximate a contour polygon using the Douglas-Peucker algorithm.
///
/// Simplifies the contour by removing points within `epsilon` distance
/// of the line between endpoints. Larger epsilon = fewer points.
///
/// Reference: Douglas & Peucker (1973)
pub fn approx_poly(contour: &[(i32, i32)], epsilon: f64) -> Vec<(i32, i32)> {
    if contour.len() <= 2 || epsilon <= 0.0 {
        return contour.to_vec();
    }
    let mut keep = vec![false; contour.len()];
    keep[0] = true;
    keep[contour.len() - 1] = true;
    douglas_peucker(contour, 0, contour.len() - 1, epsilon, &mut keep);
    contour
        .iter()
        .zip(keep.iter())
        .filter_map(|(&pt, &k)| if k { Some(pt) } else { None })
        .collect()
}

/// Apply per-pixel blend formula in normalized [0, 1] space.
///
/// `a` = foreground channel, `b` = background channel.
/// See [`BlendMode`] variants for individual formulas and validation status.
#[inline]
pub fn blend_channel(a: u8, b: u8, mode: BlendMode) -> u8 {
    let af = a as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let result = match mode {
        BlendMode::Multiply => af * bf,
        BlendMode::Screen => 1.0 - (1.0 - af) * (1.0 - bf),
        BlendMode::Overlay => {
            if bf < 0.5 {
                2.0 * af * bf
            } else {
                1.0 - 2.0 * (1.0 - af) * (1.0 - bf)
            }
        }
        BlendMode::Darken => af.min(bf),
        BlendMode::Lighten => af.max(bf),
        BlendMode::SoftLight => {
            if af < 0.5 {
                bf - (1.0 - 2.0 * af) * bf * (1.0 - bf)
            } else {
                let d = if bf <= 0.25 {
                    ((16.0 * bf - 12.0) * bf + 4.0) * bf
                } else {
                    bf.sqrt()
                };
                bf + (2.0 * af - 1.0) * (d - bf)
            }
        }
        BlendMode::HardLight => {
            if af < 0.5 {
                2.0 * af * bf
            } else {
                1.0 - 2.0 * (1.0 - af) * (1.0 - bf)
            }
        }
        BlendMode::Difference => (af - bf).abs(),
        BlendMode::Exclusion => af + bf - 2.0 * af * bf,
        BlendMode::ColorDodge => {
            if af >= 1.0 {
                1.0
            } else if bf == 0.0 {
                0.0
            } else {
                (bf / (1.0 - af)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if af <= 0.0 {
                0.0
            } else if bf >= 1.0 {
                1.0
            } else {
                1.0 - ((1.0 - bf) / af).min(1.0)
            }
        }
        BlendMode::VividLight => {
            // ColorBurn for a < 0.5, ColorDodge for a >= 0.5
            if af <= 0.5 {
                let a2 = 2.0 * af;
                if a2 <= 0.0 {
                    0.0
                } else if bf >= 1.0 {
                    1.0
                } else {
                    1.0 - ((1.0 - bf) / a2).min(1.0)
                }
            } else {
                let a2 = 2.0 * (af - 0.5);
                if a2 >= 1.0 {
                    1.0
                } else if bf == 0.0 {
                    0.0
                } else {
                    (bf / (1.0 - a2)).min(1.0)
                }
            }
        }
        BlendMode::LinearDodge => af + bf,      // clamped below
        BlendMode::LinearBurn => af + bf - 1.0, // clamped below
        BlendMode::LinearLight => {
            // LinearBurn for a < 0.5, LinearDodge for a >= 0.5
            bf + 2.0 * af - 1.0
        }
        BlendMode::PinLight => {
            if af <= 0.5 {
                bf.min(2.0 * af)
            } else {
                bf.max(2.0 * af - 1.0)
            }
        }
        BlendMode::HardMix => {
            // Threshold the VividLight result at 0.5.
            // Note: the simplified `a + b >= 1` is wrong at fg=0, bg=255
            // where VividLight(0,1) = ColorBurn(0,1) = 0, so threshold → 0.
            let vl = if af <= 0.5 {
                let a2 = 2.0 * af;
                if a2 <= 0.0 {
                    0.0
                } else if bf >= 1.0 {
                    1.0
                } else {
                    1.0 - ((1.0 - bf) / a2).min(1.0)
                }
            } else {
                let a2 = 2.0 * (af - 0.5);
                if a2 >= 1.0 {
                    1.0
                } else if bf == 0.0 {
                    0.0
                } else {
                    (bf / (1.0 - a2)).min(1.0)
                }
            };
            if vl >= 0.5 { 1.0 } else { 0.0 }
        }
        BlendMode::Subtract => bf - af, // clamped below
        BlendMode::Divide => {
            if af <= 0.0 {
                1.0
            } else {
                (bf / af).min(1.0)
            }
        }
        // Pixel-level modes — handled in blend(), not here.
        BlendMode::Dissolve
        | BlendMode::DarkerColor
        | BlendMode::LighterColor
        | BlendMode::Hue
        | BlendMode::Saturation
        | BlendMode::Color
        | BlendMode::Luminosity => unreachable!("pixel-level mode in blend_channel"),
    };
    (result.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Smoothstep function for blend-if feathering.
pub fn blend_if_smoothstep(x: f32, edge0: f32, edge1: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Blend a single pixel using a pixel-level (non-per-channel) mode.
///
/// `fg` and `bg` are RGB slices (3 bytes). `px_idx` is the pixel index
/// used for Dissolve's deterministic hash.
#[inline]
pub fn blend_pixel(fg: &[u8], bg: &[u8], mode: BlendMode, px_idx: u32) -> (u8, u8, u8) {
    match mode {
        BlendMode::Dissolve => {
            // Deterministic hash-based dither: hash the pixel index to get
            // a pseudo-random threshold. If threshold < 128, show fg; else bg.
            // This matches PS behavior for fully-opaque layers (alpha=1.0 → always fg).
            // For the general compositing path (RGBA with partial alpha), the
            // composite node pre-multiplies, so Dissolve on fully-opaque RGB
            // always selects the foreground pixel.
            let hash = px_idx
                .wrapping_mul(2654435761) // Knuth multiplicative hash
                .wrapping_shr(16) as u8;
            if hash < 128 {
                (fg[0], fg[1], fg[2])
            } else {
                (bg[0], bg[1], bg[2])
            }
        }
        BlendMode::DarkerColor => {
            let fg_lum = pixel_luminance(fg[0], fg[1], fg[2]);
            let bg_lum = pixel_luminance(bg[0], bg[1], bg[2]);
            if fg_lum <= bg_lum {
                (fg[0], fg[1], fg[2])
            } else {
                (bg[0], bg[1], bg[2])
            }
        }
        BlendMode::LighterColor => {
            let fg_lum = pixel_luminance(fg[0], fg[1], fg[2]);
            let bg_lum = pixel_luminance(bg[0], bg[1], bg[2]);
            if fg_lum >= bg_lum {
                (fg[0], fg[1], fg[2])
            } else {
                (bg[0], bg[1], bg[2])
            }
        }
        BlendMode::Hue => {
            // W3C: SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum_sat(sr, sg, sb, sat(br, bg_g, bb), lum(br, bg_g, bb));
            to_u8_triple(r, g, b)
        }
        BlendMode::Saturation => {
            // W3C: SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum_sat(br, bg_g, bb, sat(sr, sg, sb), lum(br, bg_g, bb));
            to_u8_triple(r, g, b)
        }
        BlendMode::Color => {
            // W3C: SetLum(Cs, Lum(Cb))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum(sr, sg, sb, lum(br, bg_g, bb));
            to_u8_triple(r, g, b)
        }
        BlendMode::Luminosity => {
            // W3C: SetLum(Cb, Lum(Cs))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum(br, bg_g, bb, lum(sr, sg, sb));
            to_u8_triple(r, g, b)
        }
        _ => unreachable!("per-channel mode in blend_pixel"),
    }
}

/// Young/van Vliet IIR gaussian blur on a single-channel f32 buffer.
///
/// Exact port of GEGL's `gegl:gaussian-blur` IIR implementation from gblur-1d.c.
/// Uses the recursive (IIR) algorithm from:
///   I.T. Young, L.J. van Vliet, "Recursive implementation of the Gaussian
///   filter", Signal Processing 44 (1995) 139-151.
///
/// Properties:
/// - O(1) per pixel regardless of sigma (vs O(sigma) for FIR)
/// - Infinite support (exact gaussian frequency response)
/// - Separable: applied as H then V, each forward+backward
/// - Right boundary correction via 3x3 matrix (matches GEGL exactly)
///
/// Used by shadow_highlight, retinex, clarity, frequency separation.
pub fn blur_1ch_f32(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || w == 0 || h == 0 {
        return data.to_vec();
    }

    // For very small sigma (< 0.5), IIR is unstable — fall back to FIR
    if sigma < 0.5 {
        return blur_1ch_f32_fir(data, w, h, sigma);
    }

    let (b, m) = yvv_find_constants(sigma);

    // Horizontal pass
    let mut out = data.to_vec();
    for y in 0..h {
        let off = y * w;
        yvv_blur_1d(&mut out[off..off + w], &b, &m);
    }

    // Vertical pass (extract column, blur, write back)
    let mut col = vec![0.0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = out[y * w + x];
        }
        yvv_blur_1d(&mut col, &b, &m);
        for y in 0..h {
            out[y * w + x] = col[y];
        }
    }

    out
}

/// FIR fallback for very small sigma where IIR is unstable.
pub fn blur_1ch_f32_fir(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let ksize = ((sigma * 3.0).ceil() as usize) * 2 + 1;
    let ksize = ksize.max(3);
    let kernel = gaussian_kernel_1d(ksize, sigma);
    let half = ksize / 2;

    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                let sx =
                    (x as isize + k as isize - half as isize).clamp(0, w as isize - 1) as usize;
                sum += data[y * w + sx] * kval;
            }
            tmp[y * w + x] = sum;
        }
    }

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                let sy =
                    (y as isize + k as isize - half as isize).clamp(0, h as isize - 1) as usize;
                sum += tmp[sy * w + x] * kval;
            }
            out[y * w + x] = sum;
        }
    }
    out
}

pub fn blur_impl(
    pixels: &[u8],
    info: &ImageInfo,
    config: &BlurParams,
) -> Result<Vec<u8>, ImageError> {
    let radius = config.radius;

    if radius < 0.0 {
        return Err(ImageError::InvalidParameters(
            "blur radius must be >= 0".into(),
        ));
    }
    validate_format(info.format)?;

    if radius == 0.0 {
        return Ok(pixels.to_vec());
    }

    // 16-bit: delegate to 8-bit path via process_via_8bit (convolve only supports u8)
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| blur_impl(p8, i8, config));
    }

    // In the old libblur API, radius was effectively sigma
    let sigma = radius;

    // Large sigma: use box blur approximation (O(1) per pixel, 3 passes)
    if sigma >= 20.0 {
        return gaussian_blur_box_approx(pixels, info, sigma);
    }

    // Build separable Gaussian kernel
    let ksize = {
        let k = (sigma * 6.0 + 1.0).round() as usize;
        if k % 2 == 0 { k + 1 } else { k }
    };
    let ksize = ksize.max(3);
    let k1d = gaussian_kernel_1d(ksize, sigma);

    // Build 2D kernel as outer product (convolve auto-detects separable)
    let mut kernel_2d = vec![0.0f32; ksize * ksize];
    for y in 0..ksize {
        for x in 0..ksize {
            kernel_2d[y * ksize + x] = k1d[y] * k1d[x];
        }
    }

    // Use our own convolve with auto-separable detection + WASM SIMD
    let full_rect = Rect::new(0, 0, info.width, info.height);
    let mut u = |_: Rect| Ok(pixels.to_vec());
    convolve(
        full_rect,
        &mut u,
        info,
        &kernel_2d,
        &ConvolveParams {
            kw: ksize as u32,
            kh: ksize as u32,
            divisor: 1.0,
        },
    )
}

/// Compute the axis-aligned bounding rectangle of a contour.
///
/// Returns (x, y, width, height) where (x, y) is the top-left corner.
pub fn bounding_rect(contour: &[(i32, i32)]) -> (i32, i32, i32, i32) {
    if contour.is_empty() {
        return (0, 0, 0, 0);
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y) in contour {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
}

/// Single-pass box blur on an f32 buffer (single channel, row-major).
///
/// Uses a sliding sum for O(1) per pixel regardless of radius.
/// Border handling: extend edge pixels (clamp).
pub fn box_blur_pass_f32(data: &mut [f32], w: usize, h: usize, radius: usize) {
    if radius == 0 {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];

    // Horizontal pass
    let diameter = 2 * radius + 1;
    let inv_d = 1.0 / diameter as f32;
    for y in 0..h {
        let row = y * w;
        // Initialize running sum for first output pixel
        let mut sum = 0.0f32;
        for kx in 0..diameter {
            let sx = (kx as isize - radius as isize).clamp(0, (w - 1) as isize) as usize;
            sum += data[row + sx];
        }
        tmp[row] = sum * inv_d;

        for x in 1..w {
            // Add new right pixel, subtract old left pixel
            let add_x = (x + radius).min(w - 1);
            let sub_x = (x as isize - radius as isize - 1).max(0) as usize;
            sum += data[row + add_x] - data[row + sub_x];
            tmp[row + x] = sum * inv_d;
        }
    }

    // Vertical pass
    for x in 0..w {
        let mut sum = 0.0f32;
        for ky in 0..diameter {
            let sy = (ky as isize - radius as isize).clamp(0, (h - 1) as isize) as usize;
            sum += tmp[sy * w + x];
        }
        data[x] = sum * inv_d;

        for y in 1..h {
            let add_y = (y + radius).min(h - 1);
            let sub_y = (y as isize - radius as isize - 1).max(0) as usize;
            sum += tmp[add_y * w + x] - tmp[sub_y * w + x];
            data[y * w + x] = sum * inv_d;
        }
    }
}

/// Compute box blur radii for a 3-pass stackable approximation of Gaussian blur.
///
/// Three sequential box blur passes approximate a Gaussian via the central limit
/// theorem. Returns three radii. Based on the algorithm from:
/// "Fast Almost-Gaussian Filtering" (Kovesi, 2010).
pub fn box_blur_radii_for_gaussian(sigma: f32) -> [usize; 3] {
    // ideal box filter width: w = sqrt(12*sigma^2/n + 1), n = 3 passes
    let w_ideal = ((12.0 * sigma * sigma / 3.0) + 1.0).sqrt();
    let wl = (w_ideal.floor() as usize) | 1; // round down to odd
    let wu = wl + 2; // next odd

    // how many passes use wl vs wu to best approximate the target variance
    let m = ((12.0 * sigma * sigma - (3 * wl * wl + 12 * wl + 9) as f32) / (4 * (wl + 2)) as f32)
        .round() as usize;

    let mut radii = [0usize; 3];
    for (i, r) in radii.iter_mut().enumerate() {
        *r = if i < m { wu / 2 } else { wl / 2 };
    }
    radii
}

/// Box mean via integral image — O(1) per pixel regardless of radius.
/// Box mean matching OpenCV's boxFilter with BORDER_REFLECT.
/// Pads data with reflect border, computes f32 SAT, queries fixed-size window.
pub fn box_mean(data: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let n = w * h;
    let r = radius;
    let ksize = 2 * r + 1;

    // Pad with BORDER_REFLECT (edge pixel duplicated)
    let pw = w + 2 * r;
    let ph = h + 2 * r;
    let mut padded = vec![0.0f32; pw * ph];
    for py in 0..ph {
        // BORDER_REFLECT: idx < 0 → |idx+1|, idx >= size → 2*size - idx - 1
        let sy = if py < r {
            r - 1 - py // reflect with duplication
        } else if py >= h + r {
            2 * h - (py - r) - 1
        } else {
            py - r
        };
        for px in 0..pw {
            let sx = if px < r {
                r - 1 - px
            } else if px >= w + r {
                2 * w - (px - r) - 1
            } else {
                px - r
            };
            padded[py * pw + px] = data[sy * w + sx];
        }
    }

    // Build SAT in f32
    let mut sat = vec![0.0f32; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        for x in 0..pw {
            sat[(y + 1) * (pw + 1) + (x + 1)] =
                padded[y * pw + x] + sat[y * (pw + 1) + (x + 1)] + sat[(y + 1) * (pw + 1) + x]
                    - sat[y * (pw + 1) + x];
        }
    }

    // Query: fixed ksize window centered on each original pixel
    let count = (ksize * ksize) as f32;
    let mut result = vec![0.0f32; n];
    for y in 0..h {
        let y0 = y; // in padded coords, original pixel y is at py = y + r, window starts at y
        let y1 = y + ksize;
        for x in 0..w {
            let x0 = x;
            let x1 = x + ksize;
            let sum = sat[y1 * (pw + 1) + x1] - sat[y0 * (pw + 1) + x1] - sat[y1 * (pw + 1) + x0]
                + sat[y0 * (pw + 1) + x0];
            result[y * w + x] = sum / count;
        }
    }

    result
}

/// Box mean via integral image (f64 precision).
#[allow(dead_code)] // reserved for future adaptive-threshold modes
pub fn box_mean_f64(data: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let n = w * h;
    let r = radius;

    // Build integral image with BORDER_REFLECT_101 padding
    let pw = w + 2 * r;
    let ph = h + 2 * r;
    let mut padded = vec![0.0f64; pw * ph];
    for py in 0..ph {
        let sy = if py < r {
            r - 1 - py
        } else if py >= h + r {
            2 * h - (py - r) - 1
        } else {
            py - r
        };
        for px in 0..pw {
            let sx = if px < r {
                r - 1 - px
            } else if px >= w + r {
                2 * w - (px - r) - 1
            } else {
                px - r
            };
            padded[py * pw + px] = data[sy.min(h - 1) * w + sx.min(w - 1)];
        }
    }

    // Build SAT
    let mut sat = vec![0.0f64; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        for x in 0..pw {
            sat[(y + 1) * (pw + 1) + (x + 1)] =
                padded[y * pw + x] + sat[y * (pw + 1) + (x + 1)] + sat[(y + 1) * (pw + 1) + x]
                    - sat[y * (pw + 1) + x];
        }
    }

    // Query box means
    let ksize = 2 * r + 1;
    let area = (ksize * ksize) as f64;
    let mut result = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let y1 = y;
            let x1 = x;
            let y2 = y + ksize;
            let x2 = x + ksize;
            let sum = sat[y2 * (pw + 1) + x2] - sat[y1 * (pw + 1) + x2] - sat[y2 * (pw + 1) + x1]
                + sat[y1 * (pw + 1) + x1];
            result[y * w + x] = sum / area;
        }
    }

    result
}

/// Integer box mean via integral image with BORDER_REPLICATE padding.
/// Matches OpenCV's boxFilter(src, CV_8U, ksize, BORDER_REPLICATE) exactly.
pub fn box_mean_u8_replicate(pixels: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let r = radius;
    let ksize = 2 * r + 1;

    // Pad with BORDER_REPLICATE: clamp to edge
    let pw = w + 2 * r;
    let ph = h + 2 * r;
    let mut padded = vec![0u32; pw * ph];
    for py in 0..ph {
        let sy = if py < r {
            0
        } else if py >= h + r {
            h - 1
        } else {
            py - r
        };
        for px in 0..pw {
            let sx = if px < r {
                0
            } else if px >= w + r {
                w - 1
            } else {
                px - r
            };
            padded[py * pw + px] = pixels[sy * w + sx] as u32;
        }
    }

    // Build integral image (i64 for safe subtraction)
    let mut sat = vec![0i64; (pw + 1) * (ph + 1)];
    for y in 0..ph {
        for x in 0..pw {
            sat[(y + 1) * (pw + 1) + (x + 1)] = padded[y * pw + x] as i64
                + sat[y * (pw + 1) + (x + 1)]
                + sat[(y + 1) * (pw + 1) + x]
                - sat[y * (pw + 1) + x];
        }
    }

    // Query box means with rounded integer division (matches OpenCV boxFilter CV_8U)
    let area = (ksize * ksize) as i64;
    let half_area = area / 2;
    let n = w * h;
    let mut result = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let y1 = y;
            let x1 = x;
            let y2 = y + ksize;
            let x2 = x + ksize;
            let sum = sat[y2 * (pw + 1) + x2] - sat[y1 * (pw + 1) + x2] - sat[y2 * (pw + 1) + x1]
                + sat[y1 * (pw + 1) + x1];
            result[y * w + x] = ((sum + half_area) / area) as u8;
        }
    }

    result
}

/// Box-Muller transform: two uniform randoms → two standard-normal values.
/// Returns one value per call (discards the second for simplicity).
#[inline]
pub fn box_muller(state: &mut u64) -> f64 {
    let u1 = xorshift64_f64(state).max(1e-300); // avoid log(0)
    let u2 = xorshift64_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Build an anti-aliased elliptical mask via 8×8 supersampling at boundary pixels.
///
/// Interior pixels get 1.0, exterior get 0.0, boundary pixels (where the ellipse
/// edge crosses the pixel) get the fraction of 64 sub-pixel samples that fall
/// inside the ellipse. Sub-pixel samples span [col-0.5, col+0.5) × [row-0.5, row+0.5)
/// around the integer pixel coordinate, matching ImageMagick's rasterization
/// convention where the pixel center is at integer (col, row).
pub fn build_aa_ellipse_mask(w: usize, h: usize, cx: f64, cy: f64, rx: f64, ry: f64) -> Vec<f64> {
    const N: usize = 8; // 8×8 = 64 sub-pixel samples
    let inv_rx = 1.0 / rx;
    let inv_ry = 1.0 / ry;

    let mut mask = vec![0.0f64; w * h];
    for row in 0..h {
        for col in 0..w {
            // Check all four corners of the pixel [-0.5,+0.5] around center (col, row)
            let mut corners_inside = 0u8;
            for &(dx, dy) in &[(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                let xn = (col as f64 + dx - cx) * inv_rx;
                let yn = (row as f64 + dy - cy) * inv_ry;
                if xn * xn + yn * yn <= 1.0 {
                    corners_inside += 1;
                }
            }

            if corners_inside == 4 {
                mask[row * w + col] = 1.0;
            } else if corners_inside == 0 {
                // All corners outside — check center in case the arc passes through
                let xn = (col as f64 - cx) * inv_rx;
                let yn = (row as f64 - cy) * inv_ry;
                if xn * xn + yn * yn <= 1.0 {
                    // Center inside, corners outside — supersample
                    let mut count = 0u32;
                    for sy in 0..N {
                        let py = (row as f64 - 0.5 + (sy as f64 + 0.5) / N as f64 - cy) * inv_ry;
                        let py2 = py * py;
                        for sx in 0..N {
                            let px =
                                (col as f64 - 0.5 + (sx as f64 + 0.5) / N as f64 - cx) * inv_rx;
                            if px * px + py2 <= 1.0 {
                                count += 1;
                            }
                        }
                    }
                    mask[row * w + col] = count as f64 / (N * N) as f64;
                }
            } else {
                // Mixed corners — boundary pixel, supersample
                let mut count = 0u32;
                for sy in 0..N {
                    let py = (row as f64 - 0.5 + (sy as f64 + 0.5) / N as f64 - cy) * inv_ry;
                    let py2 = py * py;
                    for sx in 0..N {
                        let px = (col as f64 - 0.5 + (sx as f64 + 0.5) / N as f64 - cx) * inv_rx;
                        if px * px + py2 <= 1.0 {
                            count += 1;
                        }
                    }
                }
                mask[row * w + col] = count as f64 / (N * N) as f64;
            }
        }
    }
    mask
}

/// Build a seeded permutation table (256 entries, doubled for wrapping).
pub fn build_perm_table(seed: u64) -> [u8; 512] {
    let mut perm = [0u8; 256];
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i as u8;
    }
    // Fisher-Yates shuffle with a simple LCG seeded from the user seed
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..256).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    let mut table = [0u8; 512];
    for i in 0..512 {
        table[i] = perm[i & 255];
    }
    table
}

/// Canny edge detection (internal — returns raw Gray8 bytes).
pub fn canny(pixels: &[u8], info: &ImageInfo, config: &CannyParams) -> Result<Vec<u8>, ImageError> {
    let low_threshold = config.low_threshold;
    let high_threshold = config.high_threshold;

    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| canny(p8, i8, config));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    // Step 1: Convert to grayscale
    let gray = to_grayscale(pixels, channels);

    // Step 2: Sobel gradient magnitude and direction
    // Note: no internal blur — matches OpenCV cv2.Canny behavior.
    // Caller should pre-blur if desired (e.g., GaussianBlur then Canny).
    let mut magnitude = vec![0.0f32; w * h];
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;

    for y in 0..h {
        let r0 = y * pw;
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            let p00 = padded[r0 + x] as f32;
            let p01 = padded[r0 + x + 1] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p10 = padded[r1 + x] as f32;
            let p12 = padded[r1 + x + 2] as f32;
            let p20 = padded[r2 + x] as f32;
            let p21 = padded[r2 + x + 1] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
            let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

            // L1 gradient magnitude (matches OpenCV default: |gx| + |gy|)
            magnitude[y * w + x] = gx.abs() + gy.abs();
        }
    }

    // Step 4: Non-maximum suppression (matches OpenCV's tangent-ratio method)
    //
    // OpenCV uses TG22 = tan(22.5°) * 2^15 = 13573 to classify angles into
    // 3 bins without atan2. Asymmetric comparison (> on one side, >= on other)
    // ensures consistent tie-breaking.
    let mut nms = vec![0u8; w * h]; // 0=suppressed, 1=weak candidate, 2=strong edge

    // Recompute Sobel components for NMS direction (need signed gx, gy)
    let padded_nms = pad_reflect(&gray, w, h, 1, 1);
    let pw_nms = w + 2;

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let mag = magnitude[y * w + x];
            if mag <= low_threshold {
                continue;
            }

            let r0 = y * pw_nms;
            let r1 = (y + 1) * pw_nms;
            let r2 = (y + 2) * pw_nms;
            let p00 = padded_nms[r0 + x] as f32;
            let p01 = padded_nms[r0 + x + 1] as f32;
            let p02 = padded_nms[r0 + x + 2] as f32;
            let p10 = padded_nms[r1 + x] as f32;
            let p12 = padded_nms[r1 + x + 2] as f32;
            let p20 = padded_nms[r2 + x] as f32;
            let p21 = padded_nms[r2 + x + 1] as f32;
            let p22 = padded_nms[r2 + x + 2] as f32;

            let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
            let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

            // OpenCV tangent-ratio NMS: TG22 = tan(22.5°) * 2^15 = 13573
            let ax = gx.abs();
            let ay = gy.abs();
            let tg22x = ax * 13573.0;
            let y_shifted = ay * 32768.0; // ay << 15

            let is_max = if y_shifted < tg22x {
                // Near-horizontal edge: compare left/right
                mag > magnitude[y * w + x - 1] && mag >= magnitude[y * w + x + 1]
            } else {
                let tg67x = tg22x + ax * 65536.0; // tg22x + (ax << 16)
                if y_shifted > tg67x {
                    // Near-vertical edge: compare up/down
                    mag > magnitude[(y - 1) * w + x] && mag >= magnitude[(y + 1) * w + x]
                } else {
                    // Diagonal edge: compare diagonal neighbors
                    let s: i32 = if (gx < 0.0) != (gy < 0.0) { -1 } else { 1 };
                    mag > magnitude[(y - 1) * w + (x as i32 - s) as usize]
                        && mag > magnitude[(y + 1) * w + (x as i32 + s) as usize]
                }
            };

            if is_max {
                if mag >= high_threshold {
                    nms[y * w + x] = 2; // strong edge
                } else {
                    nms[y * w + x] = 1; // weak candidate
                }
            }
        }
    }

    // Step 5: Hysteresis thresholding (stack-based BFS, matches OpenCV)
    let mut out = vec![0u8; w * h];
    let mut stack: Vec<(usize, usize)> = Vec::new();

    // Seed stack with strong edges
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if nms[y * w + x] == 2 {
                out[y * w + x] = 255;
                stack.push((x, y));
            }
        }
    }

    // BFS: extend strong edges to connected weak edges
    while let Some((x, y)) = stack.pop() {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                if nx < w && ny < h && nms[ny * w + nx] == 1 && out[ny * w + nx] == 0 {
                    out[ny * w + nx] = 255;
                    nms[ny * w + nx] = 2; // mark as visited
                    stack.push((nx, ny));
                }
            }
        }
    }

    Ok(out)
}

/// Clip a color so all channels are in [0,1] while preserving luminance.
#[inline]
pub fn clip_color(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = lum(r, g, b);
    let n = r.min(g).min(b);
    let x = r.max(g).max(b);
    let (mut r, mut g, mut b) = (r, g, b);
    if n < 0.0 {
        let ln = l - n;
        r = l + (r - l) * l / ln;
        g = l + (g - l) * l / ln;
        b = l + (b - l) * l / ln;
    }
    if x > 1.0 {
        let xl = x - l;
        let one_l = 1.0 - l;
        r = l + (r - l) * one_l / xl;
        g = l + (g - l) * one_l / xl;
        b = l + (b - l) * one_l / xl;
    }
    (r, g, b)
}

/// Compute Mertens weight map for a single image.
/// Input is f32 RGB in [0,1], interleaved. Returns one weight per pixel.
pub fn compute_mertens_weight(img_f: &[f32], w: usize, h: usize, params: &MertensParams) -> Vec<f32> {
    let n = w * h;
    let _sigma = 0.2f32;

    // Convert to grayscale — matches OpenCV MergeMertens which uses COLOR_RGB2GRAY
    // on BGR data, effectively: 0.299*B + 0.587*G + 0.114*R.
    // Our input is RGB, so: 0.114*R + 0.587*G + 0.299*B
    let mut gray = vec![0.0f32; n];
    for i in 0..n {
        let r = img_f[i * 3];
        let g = img_f[i * 3 + 1];
        let b = img_f[i * 3 + 2];
        gray[i] = 0.114 * r + 0.587 * g + 0.299 * b;
    }

    // Contrast: abs(Laplacian(gray)) — standard 3×3 kernel [[0,1,0],[1,-4,1],[0,1,0]]
    // Border: BORDER_REFLECT_101 (OpenCV BORDER_DEFAULT)
    let mut contrast = vec![0.0f32; n];
    let ws = w as isize;
    let hs = h as isize;
    for y in 0..h {
        for x in 0..w {
            let yp = reflect101(y as isize - 1, hs) as usize;
            let yn = reflect101(y as isize + 1, hs) as usize;
            let xp = reflect101(x as isize - 1, ws) as usize;
            let xn = reflect101(x as isize + 1, ws) as usize;

            let center = gray[y * w + x];
            let lap = gray[yp * w + x] + gray[yn * w + x] + gray[y * w + xp] + gray[y * w + xn]
                - 4.0 * center;
            contrast[y * w + x] = lap.abs();
        }
    }

    // Saturation: sqrt(sum((ch - mean)²)) — matches OpenCV MergeMertens exactly.
    // Note: OpenCV does NOT divide by channel count before sqrt (not population std).
    let mut saturation = vec![0.0f32; n];
    for i in 0..n {
        let r = img_f[i * 3];
        let g = img_f[i * 3 + 1];
        let b = img_f[i * 3 + 2];
        let mu = (r + g + b) / 3.0;
        let sum_sq = (r - mu) * (r - mu) + (g - mu) * (g - mu) + (b - mu) * (b - mu);
        saturation[i] = sum_sq.sqrt();
    }

    // Well-exposedness: product over channels of exp(-(ch - 0.5)² / (2 * σ²))
    // OpenCV computes: expo = (ch - 0.5)²; expo = -expo / 0.08; exp(expo)
    // where 0.08 = 2 * 0.2² = 2 * σ². Match the exact operation order.
    let mut well_exp = vec![1.0f32; n];
    for i in 0..n {
        for c in 0..3 {
            let ch = img_f[i * 3 + c];
            let expo = ch - 0.5;
            let expo = expo * expo;
            let expo = -expo / 0.08;
            well_exp[i] *= expo.exp();
        }
    }

    // Combined weight
    let mut weight = vec![0.0f32; n];
    for i in 0..n {
        let mut w = 1.0f32;
        if params.contrast_weight != 0.0 {
            w *= contrast[i].powf(params.contrast_weight);
        }
        if params.saturation_weight != 0.0 {
            w *= saturation[i].powf(params.saturation_weight);
        }
        if params.exposure_weight != 0.0 {
            w *= well_exp[i].powf(params.exposure_weight);
        }
        weight[i] = w + 1e-12; // avoid zero weights
    }

    weight
}

/// Connected component labeling on a binary (thresholded) grayscale image.
///
/// Returns a label map where each pixel has the label of its connected component
/// (0 = background, 1..N = component labels). Matches `cv2.connectedComponents`.
///
/// `connectivity`: 4 or 8 (default 8).
/// Input must be binary: 0 = background, non-zero = foreground.
pub fn connected_components(
    pixels: &[u8],
    info: &ImageInfo,
    connectivity: u32,
) -> Result<(Vec<u32>, u32), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "connected_components requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;

    let mut labels = vec![0u32; w * h];
    let mut parent = vec![0u32; w * h + 1]; // union-find
    let mut next_label: u32 = 1;

    // Initialize union-find
    for (i, p) in parent.iter_mut().enumerate() {
        *p = i as u32;
    }

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize]; // path compression
            x = parent[x as usize];
        }
        x
    }

    fn union(parent: &mut [u32], a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }

    // Pass 1: assign provisional labels
    for y in 0..h {
        for x in 0..w {
            if pixels[y * w + x] == 0 {
                continue; // background
            }

            let mut neighbors = Vec::with_capacity(4);

            // Check neighbors based on connectivity
            if y > 0 && pixels[(y - 1) * w + x] != 0 {
                neighbors.push(labels[(y - 1) * w + x]); // above
            }
            if x > 0 && pixels[y * w + x - 1] != 0 {
                neighbors.push(labels[y * w + x - 1]); // left
            }
            if connectivity == 8 {
                if y > 0 && x > 0 && pixels[(y - 1) * w + x - 1] != 0 {
                    neighbors.push(labels[(y - 1) * w + x - 1]); // above-left
                }
                if y > 0 && x + 1 < w && pixels[(y - 1) * w + x + 1] != 0 {
                    neighbors.push(labels[(y - 1) * w + x + 1]); // above-right
                }
            }

            if neighbors.is_empty() {
                labels[y * w + x] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbors.iter().min().unwrap();
                labels[y * w + x] = min_label;
                for &n in &neighbors {
                    if n != min_label {
                        union(&mut parent, n, min_label);
                    }
                }
            }
        }
    }

    // Pass 2: resolve labels
    let mut label_map = vec![0u32; next_label as usize];
    let mut num_labels: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            if labels[y * w + x] > 0 {
                let root = find(&mut parent, labels[y * w + x]);
                if label_map[root as usize] == 0 {
                    num_labels += 1;
                    label_map[root as usize] = num_labels;
                }
                labels[y * w + x] = label_map[root as usize];
            }
        }
    }

    Ok((labels, num_labels))
}

/// Compute the area of a contour using the shoelace formula.
///
/// Returns the absolute area. For a closed polygon, this gives the
/// enclosed area. Matches `cv2.contourArea`.
pub fn contour_area(contour: &[(i32, i32)]) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }
    let mut area: f64 = 0.0;
    let n = contour.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += contour[i].0 as f64 * contour[j].1 as f64;
        area -= contour[j].0 as f64 * contour[i].1 as f64;
    }
    area.abs() / 2.0
}

/// Compute the perimeter (arc length) of a contour.
///
/// Sums the Euclidean distances between consecutive points.
/// If `closed` is true, also adds the distance from last to first point.
pub fn contour_perimeter(contour: &[(i32, i32)], closed: bool) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    let mut perim: f64 = 0.0;
    for i in 0..contour.len() - 1 {
        let dx = (contour[i + 1].0 - contour[i].0) as f64;
        let dy = (contour[i + 1].1 - contour[i].1) as f64;
        perim += (dx * dx + dy * dy).sqrt();
    }
    if closed {
        let dx = (contour[0].0 - contour.last().unwrap().0) as f64;
        let dy = (contour[0].1 - contour.last().unwrap().1) as f64;
        perim += (dx * dx + dy * dy).sqrt();
    }
    perim
}

/// Two-pass separable convolution: horizontal then vertical.
pub fn convolve_separable(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    row_k: &[f32],
    col_k: &[f32],
    divisor: f32,
) -> Result<Vec<u8>, ImageError> {
    let rw = row_k.len() / 2;
    let rh = col_k.len() / 2;
    let pad = rw.max(rh);
    let inv_div = 1.0 / divisor;

    // Pad input
    let padded = pad_reflect(pixels, w, h, channels, pad);
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;

    // Pass 1: horizontal convolution → intermediate f32 buffer
    let mut tmp = vec![0.0f32; ph * w * channels];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        // Process 4 output values at a time with f32x4
        let total_out = ph * w * channels;
        let simd_chunks = total_out / 4;

        for chunk in 0..simd_chunks {
            let out_base = chunk * 4;
            // Determine y, x, c for each of the 4 outputs
            // Since channels is typically 1, 3, or 4, batch across x*channels
            let mut accum = f32x4_splat(0.0);

            for kx in 0..row_k.len() {
                let k_val = f32x4_splat(row_k[kx]);
                // Compute source indices for each of the 4 outputs
                let mut vals = [0.0f32; 4];
                for lane in 0..4 {
                    let idx = out_base + lane;
                    let y = idx / (w * channels);
                    let rem = idx % (w * channels);
                    let x = rem / channels;
                    let c = rem % channels;
                    let src_idx = (y * pw + x + pad - rw + kx) * channels + c;
                    vals[lane] = padded[src_idx] as f32;
                }
                let src_vec = f32x4(vals[0], vals[1], vals[2], vals[3]);
                accum = f32x4_add(accum, f32x4_mul(k_val, src_vec));
            }

            tmp[out_base] = f32x4_extract_lane::<0>(accum);
            tmp[out_base + 1] = f32x4_extract_lane::<1>(accum);
            tmp[out_base + 2] = f32x4_extract_lane::<2>(accum);
            tmp[out_base + 3] = f32x4_extract_lane::<3>(accum);
        }
        // Remainder
        for idx in simd_chunks * 4..total_out {
            let y = idx / (w * channels);
            let rem = idx % (w * channels);
            let x = rem / channels;
            let c = rem % channels;
            let mut sum = 0.0f32;
            for kx in 0..row_k.len() {
                sum += row_k[kx] * padded[(y * pw + x + pad - rw + kx) * channels + c] as f32;
            }
            tmp[idx] = sum;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for y in 0..ph {
            for x in 0..w {
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for kx in 0..row_k.len() {
                        sum +=
                            row_k[kx] * padded[(y * pw + x + pad - rw + kx) * channels + c] as f32;
                    }
                    tmp[(y * w + x) * channels + c] = sum;
                }
            }
        }
    }

    // Pass 2: vertical convolution on intermediate buffer
    let mut out = vec![0u8; w * h * channels];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let inv_div_vec = f32x4_splat(inv_div);
        let zero = f32x4_splat(0.0);
        let max_val = f32x4_splat(255.0);
        let total_out = w * h * channels;
        let simd_chunks = total_out / 4;

        for chunk in 0..simd_chunks {
            let out_base = chunk * 4;
            let mut accum = f32x4_splat(0.0);

            for ky in 0..col_k.len() {
                let k_val = f32x4_splat(col_k[ky]);
                let mut vals = [0.0f32; 4];
                for lane in 0..4 {
                    let idx = out_base + lane;
                    let y = idx / (w * channels);
                    let rem = idx % (w * channels);
                    let x = rem / channels;
                    let c = rem % channels;
                    let src_idx = ((y + pad - rh + ky) * w + x) * channels + c;
                    vals[lane] = tmp[src_idx];
                }
                let src_vec = f32x4(vals[0], vals[1], vals[2], vals[3]);
                accum = f32x4_add(accum, f32x4_mul(k_val, src_vec));
            }

            let scaled = f32x4_mul(accum, inv_div_vec);
            let clamped = f32x4_max(zero, f32x4_min(max_val, scaled));
            out[out_base] = f32x4_extract_lane::<0>(clamped) as u8;
            out[out_base + 1] = f32x4_extract_lane::<1>(clamped) as u8;
            out[out_base + 2] = f32x4_extract_lane::<2>(clamped) as u8;
            out[out_base + 3] = f32x4_extract_lane::<3>(clamped) as u8;
        }
        // Remainder
        for idx in simd_chunks * 4..total_out {
            let y = idx / (w * channels);
            let rem = idx % (w * channels);
            let x = rem / channels;
            let c = rem % channels;
            let mut sum = 0.0f32;
            for ky in 0..col_k.len() {
                sum += col_k[ky] * tmp[((y + pad - rh + ky) * w + x) * channels + c];
            }
            out[idx] = (sum * inv_div).round().clamp(0.0, 255.0) as u8;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for ky in 0..col_k.len() {
                        sum += col_k[ky] * tmp[((y + pad - rh + ky) * w + x) * channels + c];
                    }
                    out[(y * w + x) * channels + c] =
                        (sum * inv_div).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
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

/// Debevec HDR merge — compute radiance map from bracketed exposures + response curve.
///
/// Returns f32 HDR radiance map (3-channel interleaved, linear values).
pub fn debevec_hdr_merge(
    images: &[&[u8]],
    info: &ImageInfo,
    exposure_times: &[f32],
    response: &[[f32; 256]],
) -> Result<Vec<f32>, ImageError> {
    if images.len() < 2 || images.len() != exposure_times.len() {
        return Err(ImageError::InvalidInput(
            "need matching images and exposure times".into(),
        ));
    }
    if response.len() != 3 {
        return Err(ImageError::InvalidInput(
            "response must have 3 channels".into(),
        ));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "debevec requires Rgb8 input".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;

    let mut hdr = vec![0.0f32; n * 3];

    for i in 0..n {
        for ch in 0..3 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;

            for (img, &dt) in images.iter().zip(exposure_times.iter()) {
                let z = img[i * 3 + ch] as usize;
                let wt = hat_weight(z);
                let ln_dt = (dt as f64).ln();
                num += wt * (response[ch][z] as f64 - ln_dt);
                den += wt;
            }

            hdr[i * 3 + ch] = if den > 0.0 {
                (num / den).exp() as f32
            } else {
                0.0
            };
        }
    }

    Ok(hdr)
}

/// Estimate camera response curve using Debevec & Malik's method.
///
/// Takes bracketed exposures (u8 images) and exposure times.
/// Returns 256-entry response curve per channel (natural log of exposure).
///
/// Reference: Debevec & Malik "Recovering High Dynamic Range Radiance Maps
/// from Photographs" (SIGGRAPH 1997).
/// Matches OpenCV cv2.createCalibrateDebevec.
pub fn debevec_response_curve(
    images: &[&[u8]],
    info: &ImageInfo,
    exposure_times: &[f32],
    params: &DebevecParams,
) -> Result<Vec<[f32; 256]>, ImageError> {
    if images.len() < 2 || images.len() != exposure_times.len() {
        return Err(ImageError::InvalidInput(
            "need matching images and exposure times".into(),
        ));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "debevec requires Rgb8 input".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;
    let n_images = images.len();

    // Select sample pixels (deterministic, evenly spaced)
    let n_samples = params.samples.min(n);
    let step = n / n_samples;
    let sample_indices: Vec<usize> = (0..n_samples).map(|i| i * step).collect();

    let channels = 3;
    let mut response_curves = Vec::with_capacity(channels);

    for ch in 0..channels {
        // Solve for response curve g(z) where g(z) = ln(exposure)
        // Using SVD-based least squares from Debevec paper
        let n_eq = n_samples * n_images + 256 + 1; // data + smoothness + constraint
        let n_unknowns = 256 + n_samples; // g(0..255) + ln(E_i)

        // Build overdetermined system A*x = b
        let mut a = vec![0.0f64; n_eq * n_unknowns];
        let mut b = vec![0.0f64; n_eq];

        let mut eq = 0;

        // Data equations: w(z) * [g(z) - ln(dt) - ln(E)] = 0
        for (s, &si) in sample_indices.iter().enumerate() {
            for (img, &dt) in images.iter().zip(exposure_times.iter()) {
                let z = img[si * 3 + ch] as usize;
                let wt = hat_weight(z);

                a[eq * n_unknowns + z] = wt; // g(z) coefficient
                a[eq * n_unknowns + 256 + s] = -wt; // -ln(E_i) coefficient
                b[eq] = wt * (dt as f64).ln(); // w(z) * ln(dt)
                eq += 1;
            }
        }

        // Smoothness equations: lambda * w(z) * [g(z-1) - 2*g(z) + g(z+1)] = 0
        let lam = params.lambda as f64;
        for z in 1..255 {
            let wt = hat_weight(z);
            a[eq * n_unknowns + (z - 1)] = lam * wt;
            a[eq * n_unknowns + z] = -2.0 * lam * wt;
            a[eq * n_unknowns + (z + 1)] = lam * wt;
            b[eq] = 0.0;
            eq += 1;
        }

        // Fix g(128) = 0 (constraint for midpoint)
        a[eq * n_unknowns + 128] = 1.0;
        b[eq] = 0.0;
        eq += 1;

        // Solve via normal equations: A^T A x = A^T b
        let x = solve_least_squares(&a, &b, eq, n_unknowns);

        let mut curve = [0.0f32; 256];
        for z in 0..256 {
            curve[z] = x[z] as f32;
        }
        response_curves.push(curve);
    }

    Ok(response_curves)
}

/// Dilate: output pixel = maximum over structuring element neighborhood.
pub fn dilate(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    morph_op(pixels, info, ksize, shape, false)
}

/// Euclidean distance transform — distance from each pixel to nearest zero pixel.
///
/// Input: grayscale image where 0 = background, >0 = foreground.
/// Output: grayscale image where each pixel = distance to nearest background pixel.
/// Uses two-pass Rosenfeld-Pfaltz algorithm.
/// Reference: cv2.distanceTransform (OpenCV 4.13, DIST_L2).
pub fn distance_transform(pixels: &[u8], info: &ImageInfo) -> Result<Vec<f64>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "distance transform requires Gray8".into(),
        ));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let inf = (w + h) as f64;

    // Initialize: 0 for background, infinity for foreground
    let mut dist = vec![0.0f64; w * h];
    for i in 0..w * h {
        dist[i] = if pixels[i] == 0 { 0.0 } else { inf };
    }

    // Forward pass: top-left to bottom-right
    for y in 0..h {
        for x in 0..w {
            if dist[y * w + x] == 0.0 {
                continue;
            }
            if y > 0 {
                dist[y * w + x] = dist[y * w + x].min(dist[(y - 1) * w + x] + 1.0);
            }
            if x > 0 {
                dist[y * w + x] = dist[y * w + x].min(dist[y * w + x - 1] + 1.0);
            }
            // Diagonal
            if y > 0 && x > 0 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y - 1) * w + x - 1] + std::f64::consts::SQRT_2);
            }
            if y > 0 && x < w - 1 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y - 1) * w + x + 1] + std::f64::consts::SQRT_2);
            }
        }
    }

    // Backward pass: bottom-right to top-left
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            if dist[y * w + x] == 0.0 {
                continue;
            }
            if y < h - 1 {
                dist[y * w + x] = dist[y * w + x].min(dist[(y + 1) * w + x] + 1.0);
            }
            if x < w - 1 {
                dist[y * w + x] = dist[y * w + x].min(dist[y * w + x + 1] + 1.0);
            }
            if y < h - 1 && x < w - 1 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y + 1) * w + x + 1] + std::f64::consts::SQRT_2);
            }
            if y < h - 1 && x > 0 {
                dist[y * w + x] =
                    dist[y * w + x].min(dist[(y + 1) * w + x - 1] + std::f64::consts::SQRT_2);
            }
        }
    }

    Ok(dist)
}

/// Shared implementation for dodge and burn.
pub fn dodge_burn_impl(
    pixels: &[u8],
    info: &ImageInfo,
    exposure: f32,
    range: u32,
    is_dodge: bool,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            dodge_burn_impl(p8, i8, exposure, range, is_dodge)
        });
    }

    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat(
            "dodge/burn requires RGB8 or RGBA8".into(),
        ));
    }

    // Identity fast path
    if exposure.abs() < 1e-6 {
        return Ok(pixels.to_vec());
    }

    let n = (info.width as usize) * (info.height as usize);
    let mut result = vec![0u8; pixels.len()];

    for i in 0..n {
        let pi = i * ch;
        // BT.709 luminance for range weight
        let luma = (0.2126 * pixels[pi] as f32
            + 0.7152 * pixels[pi + 1] as f32
            + 0.0722 * pixels[pi + 2] as f32)
            / 255.0;

        // Range weight based on tonal selection
        let weight = match range {
            0 => {
                // Shadows: strong at dark, fades at mid
                let t = (luma * 2.0).min(1.0);
                1.0 - t * t * (3.0 - 2.0 * t) // 1-smoothstep(0, 0.5)
            }
            2 => {
                // Highlights: strong at bright, fades at mid
                let t = ((luma - 0.5) * 2.0).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t) // smoothstep(0.5, 1.0)
            }
            _ => {
                // Midtones: bell curve peaking at 0.5
                // w = 4 * luma * (1 - luma), peaks at 1.0 for luma=0.5
                (4.0 * luma * (1.0 - luma)).min(1.0)
            }
        };

        let factor = exposure * weight;

        for c in 0..3 {
            let v = pixels[pi + c] as f32;
            let adjusted = if is_dodge {
                // Dodge: lighten — output = pixel + pixel * factor
                v + v * factor
            } else {
                // Burn: darken — output = pixel * (1 - factor)
                v * (1.0 - factor)
            };
            result[pi + c] = adjusted.round().clamp(0.0, 255.0) as u8;
        }
        if ch == 4 {
            result[pi + 3] = pixels[pi + 3]; // preserve alpha
        }
    }

    Ok(result)
}

pub fn douglas_peucker(points: &[(i32, i32)], start: usize, end: usize, epsilon: f64, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let (sx, sy) = (points[start].0 as f64, points[start].1 as f64);
    let (ex, ey) = (points[end].0 as f64, points[end].1 as f64);
    let line_len = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt();

    let mut max_dist = 0.0;
    let mut max_idx = start;

    for (i, &(ptx, pty)) in points.iter().enumerate().skip(start + 1).take(end - start - 1) {
        let (px, py) = (ptx as f64, pty as f64);
        let dist = if line_len < 1e-10 {
            ((px - sx).powi(2) + (py - sy).powi(2)).sqrt()
        } else {
            ((ey - sy) * px - (ex - sx) * py + ex * sy - ey * sx).abs() / line_len
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        keep[max_idx] = true;
        douglas_peucker(points, start, max_idx, epsilon, keep);
        douglas_peucker(points, max_idx, end, epsilon, keep);
    }
}

/// Downsample by 2x using box filter (average of 2x2 blocks).
pub fn downsample_2x(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    let nw = w.div_ceil(2);
    let nh = h.div_ceil(2);
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let x0 = x * 2;
            let y0 = y * 2;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            out[y * nw + x] =
                (data[y0 * w + x0] + data[y0 * w + x1] + data[y1 * w + x0] + data[y1 * w + x1])
                    / 4.0;
        }
    }
    out
}

pub fn emboss_impl(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    // Standard emboss kernel: directional highlight along the diagonal
    #[rustfmt::skip]
    let kernel: [f32; 9] = [
        -2.0, -1.0,  0.0,
        -1.0,  1.0,  1.0,
         0.0,  1.0,  2.0,
    ];
    {
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.to_vec());
        convolve(
            r,
            &mut u,
            info,
            &kernel,
            &ConvolveParams {
                kw: 3,
                kh: 3,
                divisor: 1.0,
            },
        )
    }
}

/// Erode: output pixel = minimum over structuring element neighborhood.
///
/// For grayscale: per-pixel minimum. For RGB: per-channel minimum.
/// Matches OpenCV `cv2.erode` with `BORDER_REFLECT_101`.
pub fn erode(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    morph_op(pixels, info, ksize, shape, true)
}

/// Estimate vanishing point from line segments using weighted median of
/// pairwise intersections, weighted by product of segment lengths.
pub fn estimate_vanishing_point(lines: &[(LineSegment, f32)]) -> Option<(f32, f32)> {
    if lines.len() < 2 {
        return None;
    }

    let mut intersections: Vec<(f32, f32, f32)> = Vec::new();

    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            let (l1, w1) = &lines[i];
            let (l2, w2) = &lines[j];
            if let Some((ix, iy)) = line_intersection(l1, l2) {
                intersections.push((ix, iy, w1 * w2));
            }
        }
    }

    if intersections.is_empty() {
        return None;
    }

    let total_weight: f32 = intersections.iter().map(|&(_, _, w)| w).sum();
    if total_weight <= 0.0 {
        return None;
    }

    let mut sorted_x = intersections.clone();
    sorted_x.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let median_x = weighted_median_val(&sorted_x, total_weight, |p| p.0);

    let mut sorted_y = intersections;
    sorted_y.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let median_y = weighted_median_val(&sorted_y, total_weight, |p| p.1);

    Some((median_x, median_y))
}

/// Fade curve: 6t^5 - 15t^4 + 10t^3 (Perlin improved noise)
#[inline]
pub fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn fade_f32(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Layer multiple octaves of noise for natural-looking results.
pub fn fbm<F>(noise_fn: F, x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let mut value = 0.0f64;
    let mut amplitude = 1.0f64;
    let mut frequency = 1.0f64;
    let mut max_amp = 0.0f64;

    for _ in 0..octaves {
        value += noise_fn(x * frequency, y * frequency) * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_amp // normalize to [-1, 1]
}

#[cfg(target_arch = "wasm32")]
pub fn fbm_f32(
    perm: &[u8; 512],
    x: f32,
    y: f32,
    octaves: u32,
    noise_fn: fn(&[u8; 512], f32, f32) -> f32,
) -> f32 {
    let (mut v, mut a, mut fr, mut ma) = (0.0f32, 1.0f32, 1.0f32, 0.0f32);
    for _ in 0..octaves {
        v += noise_fn(perm, x * fr, y * fr) * a;
        ma += a;
        a *= 0.5;
        fr *= 2.0;
    }
    v / ma
}

/// Trace external contours of foreground regions in a binary Gray8 image.
///
/// Uses a simplified Suzuki-Abe border following algorithm to extract ordered
/// boundary point sequences from a binary image (0 = background, non-zero = foreground).
///
/// Returns a list of contours, each being an ordered list of (x, y) boundary points.
/// Only external (outer) contours are returned — no hierarchy.
///
/// Reference: Suzuki & Abe (1985), "Topological Structural Analysis of Digitized
/// Binary Images by Border Following"
pub fn find_contours(pixels: &[u8], info: &ImageInfo) -> Result<Vec<Vec<(i32, i32)>>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "find_contours requires Gray8 input".into(),
        ));
    }
    let w = info.width as i32;
    let h = info.height as i32;

    // Work on a copy with 1-pixel border padding (simplifies boundary checks)
    let pw = (w + 2) as usize;
    let ph = (h + 2) as usize;
    let mut img = vec![0i32; pw * ph];
    for y in 0..h as usize {
        for x in 0..w as usize {
            if pixels[y * w as usize + x] != 0 {
                img[(y + 1) * pw + (x + 1)] = 1;
            }
        }
    }

    // 8-connectivity neighbor offsets (clockwise from right)
    // Index: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
    let dx: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    let dy: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];

    let mut contours = Vec::new();
    let mut nbd: i32 = 1; // current border sequential number

    for y in 1..ph as i32 - 1 {
        for x in 1..pw as i32 - 1 {
            let idx = y as usize * pw + x as usize;
            // Detect outer border start: pixel is foreground and left neighbor is background
            if img[idx] == 1 && img[idx - 1] == 0 {
                nbd += 1;
                let contour = trace_border(&mut img, pw, x, y, &dx, &dy, nbd);
                if !contour.is_empty() {
                    // Convert from padded coordinates back to original
                    let original: Vec<(i32, i32)> =
                        contour.iter().map(|&(cx, cy)| (cx - 1, cy - 1)).collect();
                    contours.push(original);
                }
            }
            // Mark visited foreground pixels to avoid re-tracing
            if img[idx] != 0 && img[idx].abs() <= 1 {
                // Already traced or will be traced
            }
        }
    }

    Ok(contours)
}

/// Find the median value by scanning the histogram until cumulative count
/// reaches the target position.
#[inline]
pub fn find_median_in_hist(hist: &[u32; 256], target: usize) -> u8 {
    let mut cumulative = 0u32;
    for (val, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative as usize > target {
            return val as u8;
        }
    }
    255
}

/// Find the value at the given rank position by scanning the histogram.
#[inline]
pub fn find_rank_in_hist(hist: &[u32; 256], target: usize) -> u8 {
    let mut cumulative = 0u32;
    for (val, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative as usize > target {
            return val as u8;
        }
    }
    255
}

/// Flatten RGBA8 to RGB8 by blending onto a solid background color.
pub fn flatten(
    pixels: &[u8],
    info: &ImageInfo,
    bg: [u8; 3],
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "flatten requires RGBA8 input".into(),
        ));
    }
    let npixels = (info.width * info.height) as usize;
    let mut rgb = Vec::with_capacity(npixels * 3);
    for chunk in pixels.chunks_exact(4) {
        let a = chunk[3] as f32 / 255.0;
        let inv_a = 1.0 - a;
        rgb.push((chunk[0] as f32 * a + bg[0] as f32 * inv_a + 0.5) as u8);
        rgb.push((chunk[1] as f32 * a + bg[1] as f32 * inv_a + 0.5) as u8);
        rgb.push((chunk[2] as f32 * a + bg[2] as f32 * inv_a + 0.5) as u8);
    }
    Ok((
        rgb,
        ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Rgb8,
            color_space: info.color_space,
        },
    ))
}

/// Flood fill from a seed point with configurable tolerance and connectivity.
///
/// Fills connected pixels within `tolerance` of the seed pixel's value with
/// `new_val`. Returns the modified image and the number of pixels filled.
///
/// Matches `cv2.floodFill` behavior for grayscale images.
pub fn flood_fill(
    pixels: &[u8],
    info: &ImageInfo,
    seed_x: u32,
    seed_y: u32,
    new_val: u8,
    tolerance: u8,
    connectivity: u32,
) -> Result<(Vec<u8>, u32), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "flood_fill requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let sx = seed_x as usize;
    let sy = seed_y as usize;
    if sx >= w || sy >= h {
        return Err(ImageError::InvalidParameters(
            "seed point out of bounds".into(),
        ));
    }

    let mut result = pixels.to_vec();
    let seed_val = pixels[sy * w + sx];
    let lo = seed_val.saturating_sub(tolerance);
    let hi = seed_val.saturating_add(tolerance);

    let mut visited = vec![false; w * h];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((sx, sy));
    visited[sy * w + sx] = true;
    let mut filled: u32 = 0;

    while let Some((cx, cy)) = queue.pop_front() {
        let val = pixels[cy * w + cx];
        if val < lo || val > hi {
            continue;
        }
        result[cy * w + cx] = new_val;
        filled += 1;

        // 4-connectivity neighbors
        let neighbors: &[(i32, i32)] = if connectivity == 8 {
            &[
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
        } else {
            &[(0, -1), (-1, 0), (1, 0), (0, 1)]
        };

        for &(dx, dy) in neighbors {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let ni = ny as usize * w + nx as usize;
                if !visited[ni] {
                    visited[ni] = true;
                    let nval = pixels[ni];
                    if nval >= lo && nval <= hi {
                        queue.push_back((nx as usize, ny as usize));
                    }
                }
            }
        }
    }

    Ok((result, filled))
}

/// Gaussian blur approximation for u8 images using 3-pass stackable box blur.
///
/// For large sigma (>= 20), this is dramatically faster than the exact separable
/// Gaussian: O(6*N) vs O(2*K*N) where K can be 481 for sigma=80.
/// Quality: PSNR >= 35dB compared to true Gaussian for sigma >= 20.
pub fn gaussian_blur_box_approx(
    pixels: &[u8],
    info: &ImageInfo,
    sigma: f32,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    // Process each channel independently in f32 domain
    let mut result = pixels.to_vec();
    let mut channel_buf = vec![0.0f32; w * h];

    for c in 0..channels {
        // Extract channel to f32 buffer
        for i in 0..(w * h) {
            channel_buf[i] = pixels[i * channels + c] as f32;
        }

        // Apply 3-pass box blur
        stackable_box_blur_f32(&mut channel_buf, w, h, sigma);

        // Write back to result
        for i in 0..(w * h) {
            result[i * channels + c] = channel_buf[i].round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}

/// Separable Gaussian blur operating entirely in f64 precision.
/// Matches IM's KernelRank=3 Blur kernel construction (Photoshop-derived
/// 3x oversampled Gaussian for better normalization) and edge-clamp border.
pub fn gaussian_blur_f32(
    pixels: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    krad: usize,
    sigma: f64,
) -> Vec<f32> {
    // Build 1D Gaussian kernel using IM's KernelRank=3 technique:
    // Generate a Gaussian 3x wider, accumulate 3 samples per output bin.
    const KERNEL_RANK: usize = 3;
    let ksize = 2 * krad + 1;
    let mut kernel = vec![0.0f64; ksize];
    let sigma_scaled = sigma * KERNEL_RANK as f64;
    let alpha = 1.0 / (2.0 * sigma_scaled * sigma_scaled);
    let beta = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma_scaled);
    let v = (ksize * KERNEL_RANK - 1) / 2;
    for u_i in 0..=(2 * v) {
        let u = u_i as i64 - v as i64;
        let idx = u_i / KERNEL_RANK;
        kernel[idx] += (-(u * u) as f64 * alpha).exp() * beta;
    }
    let ksum: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= ksum;
    }

    let n = w * h * ch;
    // Convert input to Q16-HDRI scale (0-65535) matching IM's ScaleCharToQuantum
    // IM stores pixels as float (Quantum) in Q16-HDRI mode: value * 257.0
    const Q16_SCALE: f32 = 257.0;
    let input: Vec<f32> = pixels.iter().map(|&v| v as f32 * Q16_SCALE).collect();

    // Horizontal pass (edge-clamp border)
    let mut tmp = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut sum = 0.0f64;
                for (ki, &kval) in kernel.iter().enumerate().take(ksize) {
                    let sx = (x as i32 + ki as i32 - krad as i32).clamp(0, w as i32 - 1) as usize;
                    sum += kval * input[(y * w + sx) * ch + c] as f64;
                }
                tmp[(y * w + x) * ch + c] = sum as f32;
            }
        }
    }

    // Vertical pass (edge-clamp border)
    let mut out = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let mut sum = 0.0f64;
                for (ki, &kval) in kernel.iter().enumerate().take(ksize) {
                    let sy = (y as i32 + ki as i32 - krad as i32).clamp(0, h as i32 - 1) as usize;
                    sum += kval * tmp[(sy * w + x) * ch + c] as f64;
                }
                out[(y * w + x) * ch + c] = sum as f32;
            }
        }
    }
    out
}

/// Separable Gaussian blur (f64 precision) for adaptive threshold Gaussian mode.
#[allow(clippy::needless_range_loop)]
pub fn gaussian_blur_f64(pixels: &[u8], w: usize, h: usize, ksize: usize, sigma: f64) -> Vec<f64> {
    let r = ksize / 2;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f64; ksize];
    let mut sum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - r as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in &mut kernel {
        *k /= sum;
    }

    let hs = h as isize;
    let ws = w as isize;

    // Horizontal pass
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f64;
            for k in 0..ksize {
                let sx = x as isize + k as isize - r as isize;
                let sx = reflect101(sx, ws) as usize;
                val += pixels[y * w + sx] as f64 * kernel[k];
            }
            tmp[y * w + x] = val;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f64;
            for k in 0..ksize {
                let sy = y as isize + k as isize - r as isize;
                let sy = reflect101(sy, hs) as usize;
                val += tmp[sy * w + x] * kernel[k];
            }
            result[y * w + x] = val;
        }
    }

    result
}

#[allow(clippy::needless_range_loop)]
/// Separable 1D Gaussian blur on a f64 single-channel buffer.
///
/// Uses zero-padding outside image bounds (matching ImageMagick's vignette
/// canvas behaviour). Kernel radius computed via IM's `GetOptimalKernelWidth2D`
/// algorithm for exact Q16-compatible truncation.
pub fn gaussian_blur_mask(data: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let radius = im_gaussian_kernel_radius(sigma);
    if radius == 0 {
        return data.to_vec();
    }

    // Build 1D Gaussian kernel with IM-matched radius
    let ksize = 2 * radius + 1;
    let mut kernel = vec![0.0f64; ksize];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x * inv_2s2).exp();
        sum += kernel[i];
    }
    for v in &mut kernel {
        *v /= sum;
    }

    // Horizontal pass
    let mut tmp = vec![0.0f64; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for ki in 0..ksize {
                let src_col = col as isize + ki as isize - radius as isize;
                if src_col >= 0 && (src_col as usize) < w {
                    acc += data[row * w + src_col as usize] * kernel[ki];
                }
            }
            tmp[row * w + col] = acc;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f64; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for ki in 0..ksize {
                let src_row = row as isize + ki as isize - radius as isize;
                if src_row >= 0 && (src_row as usize) < h {
                    acc += tmp[src_row as usize * w + col] * kernel[ki];
                }
            }
            out[row * w + col] = acc;
        }
    }

    out
}

/// Generate a 1D Gaussian kernel matching OpenCV's `getGaussianKernel`.
///
/// `k[i] = exp(-0.5 * ((i - center) / sigma)^2)`, normalized to sum=1.
pub fn gaussian_kernel_1d(ksize: usize, sigma: f32) -> Vec<f32> {
    let center = (ksize / 2) as f32;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - center;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(v);
        sum += v;
    }
    let inv_sum = 1.0 / sum;
    for v in &mut kernel {
        *v *= inv_sum;
    }
    kernel
}

/// Build Gaussian pyramid for a single-channel f32 image.
/// Returns levels+1 images: [original, level1, level2, ...].
pub fn gaussian_pyramid_gray(src: &[f32], w: u32, h: u32, levels: usize) -> Vec<Vec<f32>> {
    let mut pyr = Vec::with_capacity(levels + 1);
    pyr.push(src.to_vec());
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let (down, nw, nh) = pyr_down_gray(pyr.last().unwrap(), cw, ch);
        cw = nw;
        ch = nh;
        pyr.push(down);
    }
    pyr
}

/// Gradient function for improved Perlin noise.
/// Uses hash to select from 12 gradient directions (Perlin 2002).
#[inline]
pub fn grad_perlin(hash: u8, x: f64, y: f64) -> f64 {
    match hash & 0x3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn grad_perlin_f32(hash: u8, x: f32, y: f32) -> f32 {
    match hash & 0x3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

/// Convert to grayscale using weighted channel sum.
///
/// Uses ITU-R BT.709 weights: 0.2126R + 0.7152G + 0.0722B
pub fn grayscale(pixels: &[u8], info: &ImageInfo) -> Result<DecodedImage, ImageError> {
    validate_format(info.format)?;

    let pixel_count = info.width as usize * info.height as usize;

    let gray_pixels = match info.format {
        PixelFormat::Gray8 => pixels.to_vec(),
        PixelFormat::Rgb8 => {
            let mut gray = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(3) {
                let r = chunk[0] as f32;
                let g = chunk[1] as f32;
                let b = chunk[2] as f32;
                gray.push((0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8);
            }
            gray
        }
        PixelFormat::Rgba8 => {
            let mut gray = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(4) {
                let r = chunk[0] as f32;
                let g = chunk[1] as f32;
                let b = chunk[2] as f32;
                gray.push((0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8);
            }
            gray
        }
        _ => unreachable!(),
    };

    Ok(DecodedImage {
        pixels: gray_pixels,
        info: ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Gray8,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

pub fn guided_filter_impl(
    pixels: &[u8],
    info: &ImageInfo,
    config: &GuidedFilterParams,
) -> Result<Vec<u8>, ImageError> {
    let radius = config.radius;
    let epsilon = config.epsilon;

    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "guided filter requires Gray8 input".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let r = radius as usize;
    let eps = epsilon;

    // Convert to f32
    let input: Vec<f32> = pixels.iter().map(|&v| v as f32 / 255.0).collect();

    // For self-guided: guide = input
    let guide = &input;

    // Box mean via integral image (O(1) per pixel)
    let mean_i = box_mean(&input, w, h, r);
    let mean_p = box_mean(&input, w, h, r); // p = input for self-guided

    // mean(I*p)
    let ip: Vec<f32> = input
        .iter()
        .zip(guide.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let mean_ip = box_mean(&ip, w, h, r);

    // mean(I*I)
    let ii: Vec<f32> = guide.iter().map(|&v| v * v).collect();
    let mean_ii = box_mean(&ii, w, h, r);

    // Compute a and b for each window
    let n = w * h;
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];

    for i in 0..n {
        let cov_ip = mean_ip[i] - mean_i[i] * mean_p[i];
        let var_i = mean_ii[i] - mean_i[i] * mean_i[i];
        a[i] = cov_ip / (var_i + eps);
        b[i] = mean_p[i] - a[i] * mean_i[i];
    }

    // Average a and b over window
    let mean_a = box_mean(&a, w, h, r);
    let mean_b = box_mean(&b, w, h, r);

    // Output: mean_a * I + mean_b
    let mut result = vec![0u8; n];
    for i in 0..n {
        let v = (mean_a[i] * guide[i] + mean_b[i]) * 255.0;
        result[i] = v.round().clamp(0.0, 255.0) as u8;
    }

    Ok(result)
}

/// Hat-shaped weighting function for Debevec method.
/// w(z) = z + 1 for z <= 127, 256 - z for z >= 128.
/// Gives highest weight to mid-tone pixels.
#[inline]
pub fn hat_weight(z: usize) -> f64 {
    if z <= 127 {
        (z + 1) as f64
    } else {
        (256 - z) as f64
    }
}

/// Progressive Probabilistic Hough Transform (PPHT) on a binary edge image.
///
/// Implements the exact algorithm from OpenCV's `HoughLinesProbabilistic`
/// (Matas et al., 2000). Key properties:
/// - Random pixel processing order via seeded PRNG (deterministic given seed)
/// - Incremental accumulator with vote decrement on consumed pixels
/// - Fixed-point line walking with gap tolerance
/// - Chebyshev (L∞) length check for min_length
///
/// Parameters match cv2.HoughLinesP:
/// - `rho`: distance resolution of the accumulator in pixels
/// - `theta`: angle resolution in radians
/// - `threshold`: accumulator threshold — only lines with votes > threshold are returned
/// - `min_line_length`: minimum line segment length (Chebyshev / L∞ metric)
/// - `max_line_gap`: maximum gap between points on the same line segment
/// - `seed`: PRNG seed for deterministic output (use 0 for OpenCV default)
///
/// Reference: OpenCV 4.x modules/imgproc/src/hough.cpp HoughLinesProbabilistic
#[allow(clippy::too_many_arguments)]
pub fn hough_lines_p(
    pixels: &[u8],
    info: &ImageInfo,
    rho: f32,
    theta: f32,
    threshold: i32,
    min_line_length: i32,
    max_line_gap: i32,
    seed: u64,
) -> Result<Vec<LineSegment>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::InvalidInput(
            "hough_lines_p requires Gray8 binary edge map".into(),
        ));
    }

    let width = info.width as i32;
    let height = info.height as i32;
    let irho = 1.0f32 / rho;

    // Compute numangle: discrete theta bins from [0, pi)
    let mut numangle = ((std::f64::consts::PI / theta as f64).floor() as i32) + 1;
    if numangle > 1 {
        let last_theta = (numangle - 1) as f64 * theta as f64;
        if (std::f64::consts::PI - last_theta).abs() < theta as f64 / 2.0 {
            numangle -= 1;
        }
    }

    // numrho: rho bins covering [-(w+h), +(w+h)]
    let numrho = (((width + height) * 2 + 1) as f64 / rho as f64).round() as i32;

    // Precompute trig table: trigtab[n*2] = cos(n*theta)*irho, trigtab[n*2+1] = sin(n*theta)*irho
    let mut trigtab = vec![0.0f32; (numangle * 2) as usize];
    for n in 0..numangle {
        let ang = n as f64 * theta as f64;
        trigtab[(n * 2) as usize] = (ang.cos() * irho as f64) as f32;
        trigtab[(n * 2 + 1) as usize] = (ang.sin() * irho as f64) as f32;
    }

    // Accumulator: numangle rows × numrho cols
    let mut accum = vec![0i32; (numangle * numrho) as usize];

    // Mask: tracks live edge pixels
    let mut mask = vec![0u8; (width * height) as usize];

    // Collect non-zero pixels
    let mut nzloc: Vec<(i32, i32)> = Vec::new(); // (x, y)
    for y in 0..height {
        for x in 0..width {
            if pixels[(y * width + x) as usize] != 0 {
                mask[(y * width + x) as usize] = 1;
                nzloc.push((x, y));
            }
        }
    }
    let mut count = nzloc.len();

    let mut rng = CvRng::new(if seed == 0 { u64::MAX } else { seed });
    let mut lines = Vec::new();
    let line_walk_shift: i32 = 16;

    // Main PPHT loop: process pixels in random order
    while count > 0 {
        // Pick random pixel, swap-remove
        let idx = rng.uniform(count as u32) as usize;
        let (j, i) = nzloc[idx]; // j=x, i=y
        nzloc[idx] = nzloc[count - 1];
        count -= 1;

        // Skip if already consumed
        if mask[(i * width + j) as usize] == 0 {
            continue;
        }

        // Vote: increment accumulator for all angle bins
        let mut max_val = threshold - 1;
        let mut max_n: i32 = 0;
        for n in 0..numangle {
            let r = (j as f32 * trigtab[(n * 2) as usize]
                + i as f32 * trigtab[(n * 2 + 1) as usize])
                .round() as i32
                + (numrho - 1) / 2;
            if r >= 0 && r < numrho {
                let idx_acc = (n * numrho + r) as usize;
                accum[idx_acc] += 1;
                if accum[idx_acc] > max_val {
                    max_val = accum[idx_acc];
                    max_n = n;
                }
            }
        }

        // If no bin reached threshold, continue
        if max_val < threshold {
            continue;
        }

        // Line walking: extract segment along the peak (rho, theta)
        // Line direction perpendicular to normal: a = -sin*irho, b = cos*irho
        let a = -trigtab[(max_n * 2 + 1) as usize];
        let b = trigtab[(max_n * 2) as usize];

        let mut x0 = j as i64;
        let mut y0 = i as i64;
        let dx0: i64;
        let dy0: i64;
        let xflag: bool;

        if a.abs() > b.abs() {
            xflag = true;
            dx0 = if a > 0.0 { 1 } else { -1 };
            dy0 = (b as f64 * (1i64 << line_walk_shift) as f64 / a.abs() as f64).round() as i64;
            y0 = (y0 << line_walk_shift) + (1i64 << (line_walk_shift - 1));
        } else {
            xflag = false;
            dy0 = if b > 0.0 { 1 } else { -1 };
            dx0 = (a as f64 * (1i64 << line_walk_shift) as f64 / b.abs() as f64).round() as i64;
            x0 = (x0 << line_walk_shift) + (1i64 << (line_walk_shift - 1));
        }

        // Walk in both directions to find segment endpoints
        let mut line_end = [(0i32, 0i32); 2];
        #[allow(clippy::needless_range_loop)]
        for k in 0..2usize {
            let mut gap = 0i32;
            let (mut x, mut y) = (x0, y0);
            let (dx, dy) = if k == 0 { (dx0, dy0) } else { (-dx0, -dy0) };

            loop {
                let (j1, i1) = if xflag {
                    (x as i32, (y >> line_walk_shift) as i32)
                } else {
                    ((x >> line_walk_shift) as i32, y as i32)
                };

                if j1 < 0 || j1 >= width || i1 < 0 || i1 >= height {
                    break;
                }

                if mask[(i1 * width + j1) as usize] != 0 {
                    gap = 0;
                    line_end[k] = (j1, i1);
                } else {
                    gap += 1;
                    if gap > max_line_gap {
                        break;
                    }
                }

                x += dx;
                y += dy;
            }
        }

        // Length check: Chebyshev distance (L∞), matching OpenCV
        let good_line = (line_end[1].0 - line_end[0].0).abs() >= min_line_length
            || (line_end[1].1 - line_end[0].1).abs() >= min_line_length;

        // Second walk: consume pixels along the line, decrement votes if good
        #[allow(clippy::needless_range_loop)]
        for k in 0..2usize {
            let (mut x, mut y) = (x0, y0);
            let (dx, dy) = if k == 0 { (dx0, dy0) } else { (-dx0, -dy0) };

            loop {
                let (j1, i1) = if xflag {
                    (x as i32, (y >> line_walk_shift) as i32)
                } else {
                    ((x >> line_walk_shift) as i32, y as i32)
                };

                if j1 < 0 || j1 >= width || i1 < 0 || i1 >= height {
                    break;
                }

                let midx = (i1 * width + j1) as usize;
                if mask[midx] != 0 {
                    if good_line {
                        // Decrement accumulator for ALL angle bins this pixel voted for
                        for n in 0..numangle {
                            let r = (j1 as f32 * trigtab[(n * 2) as usize]
                                + i1 as f32 * trigtab[(n * 2 + 1) as usize])
                                .round() as i32
                                + (numrho - 1) / 2;
                            if r >= 0 && r < numrho {
                                accum[(n * numrho + r) as usize] -= 1;
                            }
                        }
                    }
                    mask[midx] = 0; // Always consume, even if not good
                }

                // Stop at the endpoint found in first walk
                if i1 == line_end[k].1 && j1 == line_end[k].0 {
                    break;
                }

                x += dx;
                y += dy;
            }
        }

        if good_line {
            lines.push(LineSegment {
                x1: line_end[0].0,
                y1: line_end[0].1,
                x2: line_end[1].0,
                y2: line_end[1].1,
            });
        }
    }

    Ok(lines)
}

/// Convert hue (degrees) to an RGB tint color at full saturation, 50% lightness.
pub fn hue_to_rgb_tint(hue_deg: f32) -> [f32; 3] {
    let h = (hue_deg % 360.0 + 360.0) % 360.0;
    let c = 1.0f32; // chroma at S=1, L=0.5
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    // Add lightness offset for L=0.5 (m = L - C/2 = 0)
    [r1, g1, b1]
}

/// Compute the optimal Gaussian kernel width for a given sigma.
///
/// Reimplements ImageMagick 7's `GetOptimalKernelWidth2D` from `gem.c`:
/// starts at width 5 and grows by 2 until the normalized edge value of
/// the 2D Gaussian drops below `1.0 / quantum_range`.
///
/// For Q16 (quantum_range = 65535) this produces kernel radii that exactly
/// match ImageMagick 7.1.x's `-vignette` and `-gaussian-blur` operators.
pub fn im_gaussian_kernel_radius(sigma: f64) -> usize {
    const QUANTUM_SCALE: f64 = 1.0 / 65535.0; // Q16
    let gamma = sigma.abs();
    if gamma < 1.0e-12 {
        return 1;
    }
    let alpha = 1.0 / (2.0 * gamma * gamma);
    let beta = 1.0 / (2.0 * std::f64::consts::PI * gamma * gamma);

    let mut width: usize = 5;
    loop {
        let j = (width - 1) / 2;
        let ji = j as isize;
        let mut normalize = 0.0f64;
        for v in -ji..=ji {
            for u in -ji..=ji {
                normalize += (-(((u * u + v * v) as f64) * alpha)).exp() * beta;
            }
        }
        let value = (-((j * j) as f64 * alpha)).exp() * beta / normalize;
        if value < QUANTUM_SCALE || value < 1.0e-12 {
            break;
        }
        width += 2;
    }
    (width - 2 - 1) / 2 // convert width to radius
}

/// Interpolate a color from sorted gradient stops at the given position.
pub fn interpolate_gradient(stops: &[(f32, [u8; 3])], t: f32) -> [u8; 3] {
    if stops.len() == 1 || t <= stops[0].0 {
        return stops[0].1;
    }
    if t >= stops[stops.len() - 1].0 {
        return stops[stops.len() - 1].1;
    }
    // Find the two stops surrounding t
    for i in 0..stops.len() - 1 {
        let (p0, c0) = stops[i];
        let (p1, c1) = stops[i + 1];
        if t >= p0 && t <= p1 {
            let frac = if (p1 - p0).abs() < 1e-9 {
                0.0
            } else {
                (t - p0) / (p1 - p0)
            };
            return [
                (c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * frac + 0.5) as u8,
                (c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * frac + 0.5) as u8,
                (c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * frac + 0.5) as u8,
            ];
        }
    }
    stops[stops.len() - 1].1
}

/// Invert a 3x3 matrix (row-major). Returns None if singular.
pub fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

/// Public wrapper for integration tests.
pub fn invert_3x3_public(m: &[f64; 9]) -> Option<[f64; 9]> {
    invert_3x3(m)
}

/// Detect if a 2D kernel is separable (rank-1: K = col * row^T).
///
/// Returns `Some((row_kernel, col_kernel))` if separable, `None` otherwise.
pub fn is_separable(kernel: &[f32], kw: usize, kh: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    // Find the first non-zero row to use as reference
    let mut ref_row = None;
    for r in 0..kh {
        let row_sum: f32 = (0..kw).map(|c| kernel[r * kw + c].abs()).sum();
        if row_sum > 1e-10 {
            ref_row = Some(r);
            break;
        }
    }
    let ref_row = ref_row?;

    // Extract row kernel from the reference row
    let row_k: Vec<f32> = (0..kw).map(|c| kernel[ref_row * kw + c]).collect();

    // Find the first non-zero element in the reference row for column scale
    let ref_col = (0..kw).find(|&c| row_k[c].abs() > 1e-10)?;
    let scale = row_k[ref_col];

    // Extract column kernel: col[r] = kernel[r][ref_col] / scale
    let col_k: Vec<f32> = (0..kh).map(|r| kernel[r * kw + ref_col] / scale).collect();

    // Verify: kernel[r][c] ≈ col[r] * row[c] for all r, c
    for r in 0..kh {
        for c in 0..kw {
            let expected = col_k[r] * row_k[c];
            if (kernel[r * kw + c] - expected).abs() > 1e-4 {
                return None;
            }
        }
    }

    Some((row_k, col_k))
}

/// Laplacian edge detection (internal — returns raw Gray8 bytes).
pub fn laplacian(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, laplacian);
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;
    let gray = to_grayscale(pixels, channels);
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let r0 = y * pw;
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            let p00 = padded[r0 + x] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p11 = padded[r1 + x + 1] as f32;
            let p20 = padded[r2 + x] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            // OpenCV Laplacian ksize=3: kernel [2,0,2; 0,-8,0; 2,0,2]
            let lap = 2.0 * p00 + 2.0 * p02 - 8.0 * p11 + 2.0 * p20 + 2.0 * p22;
            out[y * w + x] = lap.abs().min(255.0) as u8;
        }
    }
    Ok(out)
}

/// Build Laplacian pyramid for a 3-channel f32 image.
/// Returns levels+1 entries: levels Laplacian layers + 1 low-res residual.
/// Each entry is (pixels, width, height).
pub fn laplacian_pyramid_rgb(src: &[f32], w: u32, h: u32, levels: usize) -> Vec<(Vec<f32>, u32, u32)> {
    // Build Gaussian pyramid first
    let mut gpyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels + 1);
    gpyr.push((src.to_vec(), w, h));
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let (down, nw, nh) = pyr_down_rgb(gpyr.last().unwrap().0.as_slice(), cw, ch);
        cw = nw;
        ch = nh;
        gpyr.push((down, nw, nh));
    }

    // Laplacian = Gaussian[i] - pyrUp(Gaussian[i+1])
    let mut lpyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels + 1);
    for i in 0..levels {
        let (ref g_curr, gw, gh) = gpyr[i];
        let (ref g_next, nw, nh) = gpyr[i + 1];
        let upsampled = pyr_up_rgb(g_next, nw, nh, gw, gh);
        let npx = (gw * gh) as usize * 3;
        let mut diff = Vec::with_capacity(npx);
        for j in 0..npx {
            diff.push(g_curr[j] - upsampled[j]);
        }
        lpyr.push((diff, gw, gh));
    }
    // Last level is the low-res residual
    let (ref last, lw, lh) = gpyr[levels];
    lpyr.push((last.clone(), lw, lh));

    lpyr
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

pub fn line_intersection(l1: &LineSegment, l2: &LineSegment) -> Option<(f32, f32)> {
    let d1x = (l1.x2 - l1.x1) as f32;
    let d1y = (l1.y2 - l1.y1) as f32;
    let d2x = (l2.x2 - l2.x1) as f32;
    let d2y = (l2.y2 - l2.y1) as f32;

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-6 {
        return None;
    }

    let t = ((l2.x1 - l1.x1) as f32 * d2y - (l2.y1 - l1.y1) as f32 * d2x) / denom;
    Some((l1.x1 as f32 + t * d1x, l1.y1 as f32 + t * d1y))
}

/// BT.601 luminance in normalized [0,1] space.
#[inline]
pub fn lum(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Generate a flat disc kernel of the given radius.
///
/// All pixels whose center falls within the circle of `radius` get weight 1.0.
/// Returns `(kernel, side_length)` where `side_length = 2 * radius + 1`.
pub fn make_disc_kernel(radius: u32) -> (Vec<f32>, usize) {
    let side = (radius * 2 + 1) as usize;
    let center = radius as f32;
    let r2 = (radius as f32 + 0.5) * (radius as f32 + 0.5);
    let mut kernel = vec![0.0f32; side * side];

    for y in 0..side {
        for x in 0..side {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            if dx * dx + dy * dy <= r2 {
                kernel[y * side + x] = 1.0;
            }
        }
    }
    (kernel, side)
}

/// Generate a flat regular hexagonal kernel of the given radius.
///
/// Uses the pointy-top hexagon inscribed in a circle of `radius`. A pixel
/// is inside the hexagon if it satisfies all 6 half-plane constraints of the
/// regular hexagon with circumradius `radius + 0.5`.
pub fn make_hex_kernel(radius: u32) -> (Vec<f32>, usize) {
    let side = (radius * 2 + 1) as usize;
    let center = radius as f32;
    let cr = radius as f32 + 0.5; // circumradius
    let mut kernel = vec![0.0f32; side * side];

    // Regular hexagon (pointy-top): a point (dx, dy) is inside if
    // |dy| <= cr * sqrt(3)/2  AND  |dy| * 0.5 + |dx| * sqrt(3)/2 <= cr * sqrt(3)/2
    let h = cr * (3.0_f32.sqrt() / 2.0);

    for y in 0..side {
        for x in 0..side {
            let dx = (x as f32 - center).abs();
            let dy = (y as f32 - center).abs();
            if dy <= h && dy * 0.5 + dx * (3.0_f32.sqrt() / 2.0) <= h {
                kernel[y * side + x] = 1.0;
            }
        }
    }
    (kernel, side)
}

/// Generate a regular polygon kernel with N sides, rotated by angle degrees.
pub fn make_polygon_kernel(radius: u32, sides: u32, rotation_deg: f32) -> (Vec<f32>, usize) {
    let side = (radius * 2 + 1) as usize;
    let center = radius as f32;
    let cr = radius as f32 + 0.5;
    let rot = rotation_deg.to_radians();
    let n = sides as f32;
    let mut kernel = vec![0.0f32; side * side];

    for y in 0..side {
        for x in 0..side {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > cr {
                continue;
            }
            // Check if point is inside the regular polygon
            // A point is inside if for each edge, it's on the interior side
            let angle = dy.atan2(dx) - rot;
            // Angular distance to nearest vertex
            let sector = (angle * n / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
            let half_angle = std::f32::consts::PI / n;
            // The polygon edge at this sector has distance: cr * cos(half_angle) / cos(...)
            let sector_angle = (sector - 0.5).abs() * 2.0 * half_angle;
            let edge_dist = cr * half_angle.cos() / sector_angle.cos().max(0.001);
            if dist <= edge_dist {
                kernel[y * side + x] = 1.0;
            }
        }
    }
    (kernel, side)
}

/// Generate a structuring element as a boolean mask.
pub fn make_structuring_element(shape: MorphShape, kw: usize, kh: usize) -> Vec<bool> {
    let mut se = vec![false; kw * kh];
    let cx = kw / 2;
    let cy = kh / 2;
    match shape {
        MorphShape::Rect => {
            se.fill(true);
        }
        MorphShape::Cross => {
            for y in 0..kh {
                for x in 0..kw {
                    se[y * kw + x] = x == cx || y == cy;
                }
            }
        }
        MorphShape::Ellipse => {
            // Exact match with OpenCV getStructuringElement(MORPH_ELLIPSE).
            // From OpenCV source (morph.dispatch.cpp):
            //   r = ksize.height/2, c = ksize.width/2
            //   inv_r2 = 1.0/(r*r)
            //   for row i: j = c if dy==0, else round(sqrt(c²*(1 - dy²*inv_r2)))
            //   fill from c-j to c+j
            let r = (kh / 2) as f64;
            let c = (kw / 2) as f64;
            let inv_r2 = if r > 0.0 { 1.0 / (r * r) } else { 0.0 };
            for y in 0..kh {
                let dy = y as f64 - r;
                let j = if dy != 0.0 {
                    let t = c * c * (1.0 - dy * dy * inv_r2);
                    (t.max(0.0).sqrt()).round() as isize
                } else {
                    c as isize
                }
                .max(0) as usize;
                let x_start = cx.saturating_sub(j);
                let x_end = (cx + j + 1).min(kw);
                for x in x_start..x_end {
                    se[y * kw + x] = true;
                }
            }
        }
    }
    se
}

/// Histogram sliding-window median (Huang algorithm) for large radii.
///
/// Maintains a 256-bin histogram. When sliding horizontally, removes the
/// leftmost column and adds the rightmost column — O(2*diameter) per pixel
/// instead of O(diameter^2).
pub fn median_histogram(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    radius: u32,
) -> Result<Vec<u8>, ImageError> {
    let r = radius as i32;
    let diameter = (2 * r + 1) as usize;
    let median_pos = (diameter * diameter) / 2;
    let mut out = vec![0u8; pixels.len()];

    for c in 0..channels {
        for y in 0..h {
            let mut hist = [0u32; 256];
            let mut _count = 0u32;

            // Initialize histogram for first window in this row
            for ky in -r..=r {
                let sy = reflect(y as i32 + ky, h);
                for kx in -r..=r {
                    let sx = reflect(kx, w);
                    hist[pixels[(sy * w + sx) * channels + c] as usize] += 1;
                    _count += 1;
                }
            }

            // Find median for first pixel
            out[y * w * channels + c] = find_median_in_hist(&hist, median_pos);

            // Slide right across the row
            for x in 1..w {
                // Remove leftmost column (x - r - 1)
                let old_x = x as i32 - r - 1;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(old_x, w);
                    let val = pixels[(sy * w + sx) * channels + c] as usize;
                    hist[val] -= 1;
                    _count -= 1;
                }

                // Add rightmost column (x + r)
                let new_x = x as i32 + r;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(new_x, w);
                    let val = pixels[(sy * w + sx) * channels + c] as usize;
                    hist[val] += 1;
                    _count += 1;
                }

                out[(y * w + x) * channels + c] = find_median_in_hist(&hist, median_pos);
            }
        }
    }
    Ok(out)
}

/// Sorting-based median for small radii (radius <= 2).
pub fn median_sort(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    radius: u32,
) -> Result<Vec<u8>, ImageError> {
    let r = radius as i32;
    let window_size = ((2 * r + 1) * (2 * r + 1)) as usize;
    let median_pos = window_size / 2;
    let mut out = vec![0u8; pixels.len()];
    let mut window = Vec::with_capacity(window_size);

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                window.clear();
                for ky in -r..=r {
                    for kx in -r..=r {
                        let sy = reflect(y as i32 + ky, h);
                        let sx = reflect(x as i32 + kx, w);
                        window.push(pixels[(sy * w + sx) * channels + c]);
                    }
                }
                window.sort_unstable();
                out[(y * w + x) * channels + c] = window[median_pos];
            }
        }
    }
    Ok(out)
}

/// Mertens exposure fusion — blends multiple exposures without HDR intermediate.
///
/// Takes a list of same-size RGB8 images and produces a fused result.
/// Uses Laplacian pyramid blending with per-pixel weights based on
/// contrast, saturation, and well-exposedness.
///
/// Reference: OpenCV cv2.createMergeMertens (photo/src/merge.cpp).
/// Algorithm: Mertens et al. "Exposure Fusion" (Pacific Graphics 2007).
pub fn mertens_fusion(
    images: &[&[u8]],
    info: &ImageInfo,
    params: &MertensParams,
) -> Result<Vec<u8>, ImageError> {
    if images.len() < 2 {
        return Err(ImageError::InvalidInput("need at least 2 images".into()));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "mertens fusion requires Rgb8 input".into(),
        ));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;
    let expected_len = n * 3;
    for img in images {
        if img.len() != expected_len {
            return Err(ImageError::InvalidInput("image size mismatch".into()));
        }
    }

    let n_images = images.len();

    // Convert images to f32 [0,1] (3-channel interleaved)
    let images_f: Vec<Vec<f32>> = images
        .iter()
        .map(|img| img.iter().map(|&v| v as f32 / 255.0).collect())
        .collect();

    // Step 1: Compute per-pixel weights for each image
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n_images);
    for img_f in &images_f {
        let weight = compute_mertens_weight(img_f, w, h, params);
        weights.push(weight);
    }

    // Step 2: Normalize weights (sum to 1 per pixel)
    for px in 0..n {
        let sum: f32 = weights.iter().map(|w| w[px]).sum();
        if sum > 0.0 {
            for w in &mut weights {
                w[px] /= sum;
            }
        }
    }

    // Step 3: Laplacian pyramid blending
    // Pyramid depth: log2(min(w,h))
    // Match OpenCV: int(logf(float(min(w,h))) / logf(2.0f))
    let maxlevel = ((w.min(h) as f32).ln() / 2.0f32.ln()) as usize;

    // Build weight Gaussian pyramids and image Laplacian pyramids
    let weight_pyrs: Vec<Vec<Vec<f32>>> = weights
        .iter()
        .map(|w| gaussian_pyramid_gray(w, info.width, info.height, maxlevel))
        .collect();

    let image_lap_pyrs: Vec<Vec<(Vec<f32>, u32, u32)>> = images_f
        .iter()
        .map(|img| laplacian_pyramid_rgb(img, info.width, info.height, maxlevel))
        .collect();

    // Step 4: Merge at each level
    let mut merged_pyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(maxlevel + 1);
    for level in 0..=maxlevel {
        let (_, lw, lh) = image_lap_pyrs[0][level];
        let lpx = (lw * lh) as usize;
        let mut merged = vec![0.0f32; lpx * 3];

        for i in 0..n_images {
            let (ref lap, _, _) = image_lap_pyrs[i][level];
            let weight = &weight_pyrs[i][level];
            for px in 0..lpx {
                let wt = weight[px];
                merged[px * 3] += wt * lap[px * 3];
                merged[px * 3 + 1] += wt * lap[px * 3 + 1];
                merged[px * 3 + 2] += wt * lap[px * 3 + 2];
            }
        }
        merged_pyr.push((merged, lw, lh));
    }

    // Step 5: Collapse the merged Laplacian pyramid
    let (mut result, mut rw, mut rh) = merged_pyr.pop().unwrap();
    for level in (0..maxlevel).rev() {
        let (ref lap, lw, lh) = merged_pyr[level];
        let upsampled = pyr_up_rgb(&result, rw, rh, lw, lh);
        result = Vec::with_capacity((lw * lh) as usize * 3);
        let lpx = (lw * lh) as usize;
        for px in 0..(lpx * 3) {
            result.push(upsampled[px] + lap[px]);
        }
        rw = lw;
        rh = lh;
    }

    // Convert back to u8, clamp
    let mut output = vec![0u8; n * 3];
    for i in 0..(n * 3) {
        output[i] = (result[i] * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(output)
}

/// Mertens fusion returning f32 output in [0,1] range (for precision testing).
pub fn mertens_fusion_f32(
    images: &[&[u8]],
    info: &ImageInfo,
    params: &MertensParams,
) -> Result<Vec<f32>, ImageError> {
    if images.len() < 2 {
        return Err(ImageError::InvalidInput("need at least 2 images".into()));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "mertens fusion requires Rgb8 input".into(),
        ));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;
    let expected_len = n * 3;
    for img in images {
        if img.len() != expected_len {
            return Err(ImageError::InvalidInput("image size mismatch".into()));
        }
    }

    let n_images = images.len();
    let images_f: Vec<Vec<f32>> = images
        .iter()
        .map(|img| img.iter().map(|&v| v as f32 / 255.0).collect())
        .collect();

    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n_images);
    for img_f in &images_f {
        let weight = compute_mertens_weight(img_f, w, h, params);
        weights.push(weight);
    }

    for px in 0..n {
        let sum: f32 = weights.iter().map(|w| w[px]).sum();
        if sum > 0.0 {
            for w in &mut weights {
                w[px] /= sum;
            }
        }
    }

    // Match OpenCV: int(logf(float(min(w,h))) / logf(2.0f))
    let maxlevel = ((w.min(h) as f32).ln() / 2.0f32.ln()) as usize;

    let weight_pyrs: Vec<Vec<Vec<f32>>> = weights
        .iter()
        .map(|w| gaussian_pyramid_gray(w, info.width, info.height, maxlevel))
        .collect();

    let image_lap_pyrs: Vec<Vec<(Vec<f32>, u32, u32)>> = images_f
        .iter()
        .map(|img| laplacian_pyramid_rgb(img, info.width, info.height, maxlevel))
        .collect();

    let mut merged_pyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(maxlevel + 1);
    for level in 0..=maxlevel {
        let (_, lw, lh) = image_lap_pyrs[0][level];
        let lpx = (lw * lh) as usize;
        let mut merged = vec![0.0f32; lpx * 3];

        for i in 0..n_images {
            let (ref lap, _, _) = image_lap_pyrs[i][level];
            let weight = &weight_pyrs[i][level];
            for px in 0..lpx {
                let wt = weight[px];
                merged[px * 3] += wt * lap[px * 3];
                merged[px * 3 + 1] += wt * lap[px * 3 + 1];
                merged[px * 3 + 2] += wt * lap[px * 3 + 2];
            }
        }
        merged_pyr.push((merged, lw, lh));
    }

    let (mut result, mut rw, mut rh) = merged_pyr.pop().unwrap();
    for level in (0..maxlevel).rev() {
        let (ref lap, lw, lh) = merged_pyr[level];
        let upsampled = pyr_up_rgb(&result, rw, rh, lw, lh);
        result = Vec::with_capacity((lw * lh) as usize * 3);
        let lpx = (lw * lh) as usize;
        for px in 0..(lpx * 3) {
            result.push(upsampled[px] + lap[px]);
        }
        rw = lw;
        rh = lh;
    }

    Ok(result)
}

/// Black-hat: closing - input. Extracts small dark features.
pub fn morph_blackhat(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let closed = morph_close(pixels, info, ksize, shape)?;
    Ok(closed
        .iter()
        .zip(pixels.iter())
        .map(|(&c, &p)| c.saturating_sub(p))
        .collect())
}

/// Morphological closing: dilate then erode. Fills small dark holes.
pub fn morph_close(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let dilated = dilate(pixels, info, ksize, shape)?;
    erode(&dilated, info, ksize, shape)
}

/// Morphological gradient: dilate - erode. Highlights edges.
pub fn morph_gradient(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let dilated = dilate(pixels, info, ksize, shape)?;
    let eroded = erode(pixels, info, ksize, shape)?;
    Ok(dilated
        .iter()
        .zip(eroded.iter())
        .map(|(&d, &e)| d.saturating_sub(e))
        .collect())
}

/// Core morphological operation (erode=min, dilate=max).
pub fn morph_op(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
    is_erode: bool,
) -> Result<Vec<u8>, ImageError> {
    let ch = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "morphology on {other:?} not supported"
            )));
        }
    };
    let w = info.width as usize;
    let h = info.height as usize;
    let kw = ksize as usize;
    let kh = ksize as usize;
    let kx = kw / 2;
    let ky = kh / 2;
    let se = make_structuring_element(shape, kw, kh);

    let process_ch = if info.format == PixelFormat::Rgba8 {
        3
    } else {
        ch
    };
    let mut out = pixels.to_vec();

    for y in 0..h {
        for x in 0..w {
            for c in 0..process_ch {
                let mut val = if is_erode { 255u8 } else { 0u8 };
                for ky2 in 0..kh {
                    for kx2 in 0..kw {
                        if !se[ky2 * kw + kx2] {
                            continue;
                        }
                        // Reflect101 boundary
                        let sy = reflect101(y as isize + ky2 as isize - ky as isize, h as isize)
                            as usize;
                        let sx = reflect101(x as isize + kx2 as isize - kx as isize, w as isize)
                            as usize;
                        let p = pixels[(sy * w + sx) * ch + c];
                        if is_erode {
                            val = val.min(p);
                        } else {
                            val = val.max(p);
                        }
                    }
                }
                out[(y * w + x) * ch + c] = val;
            }
        }
    }
    Ok(out)
}

/// Morphological opening: erode then dilate. Removes small bright spots.
pub fn morph_open(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let eroded = erode(pixels, info, ksize, shape)?;
    dilate(&eroded, info, ksize, shape)
}

pub fn morph_shape_from_u32(v: u32) -> MorphShape {
    match v {
        1 => MorphShape::Ellipse,
        2 => MorphShape::Cross,
        _ => MorphShape::Rect,
    }
}

/// Top-hat: input - opening. Extracts small bright features.
pub fn morph_tophat(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    let opened = morph_open(pixels, info, ksize, shape)?;
    Ok(pixels
        .iter()
        .zip(opened.iter())
        .map(|(&p, &o)| p.saturating_sub(o))
        .collect())
}

/// Non-local means denoising for grayscale images.
///
/// With `NlmAlgorithm::OpenCv` (default): replicates OpenCV's
/// `fastNlMeansDenoising` exactly — integer SSD with bit-shift division
/// to approximate average, precomputed weight LUT indexed by integer
/// almost-average-distance, fixed-point integer accumulation.
///
/// With `NlmAlgorithm::Classic`: standard Buades et al. 2005 with float math.
pub fn nlm_denoise(
    pixels: &[u8],
    info: &ImageInfo,
    params: &NlmParams,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "NLM denoising currently supports Gray8 only".into(),
        ));
    }
    match params.algorithm {
        NlmAlgorithm::OpenCv => nlm_denoise_opencv(pixels, info, params),
        NlmAlgorithm::Classic => nlm_denoise_classic(pixels, info, params),
    }
}

/// Classic NLM (Buades 2005) with float math.
pub fn nlm_denoise_classic(
    pixels: &[u8],
    info: &ImageInfo,
    params: &NlmParams,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let ps = params.patch_size as usize;
    let ss = params.search_size as usize;
    let pr = ps / 2;
    let sr = ss / 2;
    let h2 = params.h * params.h;
    let patch_area = (ps * ps) as f32;

    let mut out = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut weight_sum: f32 = 0.0;
            let mut pixel_sum: f32 = 0.0;

            let sy_start = (y as i32 - sr as i32).max(0) as usize;
            let sy_end = (y + sr + 1).min(h);
            let sx_start = (x as i32 - sr as i32).max(0) as usize;
            let sx_end = (x + sr + 1).min(w);

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    let mut ssd: f32 = 0.0;
                    for py in 0..ps {
                        for ppx in 0..ps {
                            let y1 = reflect101(y as isize + py as isize - pr as isize, h as isize)
                                as usize;
                            let x1 = reflect101(x as isize + ppx as isize - pr as isize, w as isize)
                                as usize;
                            let y2 = reflect101(sy as isize + py as isize - pr as isize, h as isize)
                                as usize;
                            let x2 =
                                reflect101(sx as isize + ppx as isize - pr as isize, w as isize)
                                    as usize;
                            let d = pixels[y1 * w + x1] as f32 - pixels[y2 * w + x2] as f32;
                            ssd += d * d;
                        }
                    }
                    let weight = (-ssd / (patch_area * h2)).exp();
                    weight_sum += weight;
                    pixel_sum += weight * pixels[sy * w + sx] as f32;
                }
            }

            out[y * w + x] = if weight_sum > 0.0 {
                (pixel_sum / weight_sum).round().clamp(0.0, 255.0) as u8
            } else {
                pixels[y * w + x]
            };
        }
    }

    Ok(out)
}

/// OpenCV-exact NLM implementation.
///
/// Replicates `FastNlMeansDenoisingInvoker` from OpenCV 4.x:
/// - `copyMakeBorder(BORDER_DEFAULT)` → reflect101 padding
/// - Integer SSD between patches
/// - `almostAvgDist = ssd >> bin_shift` (bit-shift approximation of SSD/N)
/// - Precomputed `almost_dist2weight[almostAvgDist]` LUT
/// - Fixed-point integer accumulation with `fixed_point_mult`
/// - `divByWeightsSum` with rounding
pub fn nlm_denoise_opencv(
    pixels: &[u8],
    info: &ImageInfo,
    params: &NlmParams,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let tw = params.patch_size as usize; // template window size
    let sw = params.search_size as usize; // search window size
    let thr = tw / 2; // template half radius
    let shr = sw / 2; // search half radius
    let border = shr + thr;

    // Create border-extended image (BORDER_REFLECT_101)
    let ew = w + 2 * border;
    let eh = h + 2 * border;
    let mut ext = vec![0u8; ew * eh];
    for ey in 0..eh {
        for ex in 0..ew {
            let sy = reflect101(ey as isize - border as isize, h as isize) as usize;
            let sx = reflect101(ex as isize - border as isize, w as isize) as usize;
            ext[ey * ew + ex] = pixels[sy * w + sx];
        }
    }

    // Precompute weight LUT (matches OpenCV's constructor)
    let tw_sq = tw * tw;
    let bin_shift = {
        let mut p = 0u32;
        while (1u32 << p) < tw_sq as u32 {
            p += 1;
        }
        p
    };
    let almost_dist2actual: f64 = (1u64 << bin_shift) as f64 / tw_sq as f64;
    // DistSquared::maxDist<uchar>() = sampleMax * sampleMax * channels = 255*255*1
    let max_dist: i32 = 255 * 255;
    let almost_max_dist = (max_dist as f64 / almost_dist2actual + 1.0) as usize;

    // fixed_point_mult: max value that won't overflow i32 accumulation
    let max_estimate_sum = sw as i64 * sw as i64 * 255i64;
    let fixed_point_mult = (i32::MAX as i64 / max_estimate_sum).min(255) as i32;

    let weight_threshold = (0.001 * fixed_point_mult as f64) as i32;

    let mut lut = vec![0i32; almost_max_dist];
    for (ad, lut_entry) in lut.iter_mut().enumerate().take(almost_max_dist) {
        let dist = ad as f64 * almost_dist2actual;
        // OpenCV DistSquared::calcWeight: exp(-dist / (h*h * channels))
        // Note: -dist (NOT -dist*dist) because dist is already squared per-pixel distance.
        // For grayscale (channels=1): exp(-dist / (h*h))
        let wf = (-dist / (params.h as f64 * params.h as f64)).exp();
        let wi = (fixed_point_mult as f64 * wf + 0.5) as i32;
        *lut_entry = if wi < weight_threshold { 0 } else { wi };
    }

    let mut out = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut estimation: i64 = 0;
            let mut weights_sum: i64 = 0;

            // For each search window position
            for sy in 0..sw {
                for sx in 0..sw {
                    // Compute SSD between patches (integer)
                    let mut ssd: i32 = 0;
                    for ty in 0..tw {
                        for tx in 0..tw {
                            let a_y = border + y - thr + ty;
                            let a_x = border + x - thr + tx;
                            let b_y = border + y - shr + sy - thr + ty;
                            let b_x = border + x - shr + sx - thr + tx;
                            let a = ext[a_y * ew + a_x] as i32;
                            let b = ext[b_y * ew + b_x] as i32;
                            ssd += (a - b) * (a - b);
                        }
                    }

                    let almost_avg_dist = (ssd >> bin_shift) as usize;
                    let weight = lut[almost_avg_dist.min(lut.len() - 1)] as i64;

                    let p = ext[(border + y - shr + sy) * ew + (border + x - shr + sx)] as i64;
                    estimation += weight * p;
                    weights_sum += weight;
                }
            }

            // OpenCV divByWeightsSum: (unsigned(estimation) + weights_sum/2) / weights_sum
            out[y * w + x] = if weights_sum > 0 {
                ((estimation as u64 + weights_sum as u64 / 2) / weights_sum as u64).min(255) as u8
            } else {
                pixels[y * w + x]
            };
        }
    }

    Ok(out)
}

/// Compute Otsu's optimal threshold for a grayscale image.
///
/// Maximizes inter-class variance between foreground and background.
/// Returns the threshold value [0, 255].
///
/// Reference: OpenCV cv2.threshold(..., THRESH_OTSU).
pub fn otsu_threshold(pixels: &[u8], info: &ImageInfo) -> Result<u8, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "otsu requires Gray8 input".into(),
        ));
    }

    let n = pixels.len() as f64;
    if n == 0.0 {
        return Ok(0);
    }

    // Build histogram
    let mut hist = [0u32; 256];
    for &v in pixels {
        hist[v as usize] += 1;
    }

    // Compute total mean
    let mut total_sum = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        total_sum += i as f64 * h as f64;
    }

    let mut best_thresh = 0u8;
    let mut best_var = 0.0f64;
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;

    for (t, &ht) in hist.iter().enumerate() {
        w0 += ht as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = n - w0;
        if w1 == 0.0 {
            break;
        }

        sum0 += t as f64 * ht as f64;
        let mu0 = sum0 / w0;
        let mu1 = (total_sum - sum0) / w1;
        let between_var = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

        if between_var > best_var {
            best_var = between_var;
            best_thresh = t as u8;
        }
    }

    Ok(best_thresh)
}

/// Create a padded copy of the image with reflected borders.
///
/// Eliminates per-pixel boundary checks — interior pixels use direct indexing.
pub fn pad_reflect(pixels: &[u8], w: usize, h: usize, channels: usize, pad: usize) -> Vec<u8> {
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;
    let mut out = vec![0u8; pw * ph * channels];

    for py in 0..ph {
        let sy = reflect(py as i32 - pad as i32, h);
        for px in 0..pw {
            let sx = reflect(px as i32 - pad as i32, w);
            let src = (sy * w + sx) * channels;
            let dst = (py * pw + px) * channels;
            out[dst..dst + channels].copy_from_slice(&pixels[src..src + channels]);
        }
    }
    out
}

/// Parse a JSON string of control points: `[[x,y],[x,y],...]` into `Vec<(f32, f32)>`.
pub fn parse_curve_points(json: &str) -> Result<Vec<(f32, f32)>, ImageError> {
    // Minimal JSON array parser — avoids serde dependency for simple [[f,f],...] arrays
    let s = json.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(ImageError::InvalidParameters(
            "curves points must be a JSON array: [[x,y],...]".into(),
        ));
    }
    // Strip outer brackets
    let inner = &s[1..s.len() - 1];
    let mut points = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, ch) in inner.char_indices() {
        match ch {
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let pair = &inner[start + 1..i];
                    let parts: Vec<&str> = pair.split(',').collect();
                    if parts.len() != 2 {
                        return Err(ImageError::InvalidParameters(format!(
                            "each curve point must be [x,y], got: [{pair}]"
                        )));
                    }
                    let x: f32 = parts[0].trim().parse().map_err(|_| {
                        ImageError::InvalidParameters(format!(
                            "invalid x in curve point: {}",
                            parts[0].trim()
                        ))
                    })?;
                    let y: f32 = parts[1].trim().parse().map_err(|_| {
                        ImageError::InvalidParameters(format!(
                            "invalid y in curve point: {}",
                            parts[1].trim()
                        ))
                    })?;
                    points.push((x, y));
                }
            }
            _ => {}
        }
    }
    if points.len() < 2 {
        return Err(ImageError::InvalidParameters(
            "curves requires at least 2 control points".into(),
        ));
    }
    Ok(points)
}

/// Parse gradient stops from string format "pos:RRGGBB,pos:RRGGBB,...".
pub fn parse_gradient_stops(stops: &str) -> Result<Vec<(f32, [u8; 3])>, ImageError> {
    let mut result = Vec::new();
    for entry in stops.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(ImageError::InvalidParameters(format!(
                "gradient stop must be 'pos:RRGGBB', got '{entry}'"
            )));
        }
        let pos: f32 = parts[0].parse().map_err(|_| {
            ImageError::InvalidParameters(format!("invalid position: '{}'", parts[0]))
        })?;
        let hex = parts[1].trim_start_matches('#');
        if hex.len() != 6 {
            return Err(ImageError::InvalidParameters(format!(
                "color must be 6-digit hex, got '{hex}'"
            )));
        }
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex color: '{hex}'")))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex color: '{hex}'")))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex color: '{hex}'")))?;
        result.push((pos, [r, g, b]));
    }
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if result.is_empty() {
        return Err(ImageError::InvalidParameters(
            "gradient must have at least one stop".into(),
        ));
    }
    Ok(result)
}

/// Parse sparse color control points from "x,y:RRGGBB;x,y:RRGGBB;..." format.
pub fn parse_sparse_points(points: &str) -> Result<Vec<(f32, f32, [u8; 3])>, ImageError> {
    let mut result = Vec::new();
    for entry in points.split(';') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(ImageError::InvalidParameters(format!(
                "sparse point must be 'x,y:RRGGBB', got '{entry}'"
            )));
        }
        let coords: Vec<&str> = parts[0].split(',').collect();
        if coords.len() != 2 {
            return Err(ImageError::InvalidParameters(format!(
                "coordinates must be 'x,y', got '{}'",
                parts[0]
            )));
        }
        let x: f32 = coords[0]
            .trim()
            .parse()
            .map_err(|_| ImageError::InvalidParameters(format!("invalid x: '{}'", coords[0])))?;
        let y: f32 = coords[1]
            .trim()
            .parse()
            .map_err(|_| ImageError::InvalidParameters(format!("invalid y: '{}'", coords[1])))?;
        let hex = parts[1].trim().trim_start_matches('#');
        if hex.len() != 6 {
            return Err(ImageError::InvalidParameters(format!(
                "color must be 6-digit hex, got '{hex}'"
            )));
        }
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex: '{hex}'")))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex: '{hex}'")))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex: '{hex}'")))?;
        result.push((x, y, [r, g, b]));
    }
    if result.is_empty() {
        return Err(ImageError::InvalidParameters(
            "sparse_color requires at least one control point".into(),
        ));
    }
    Ok(result)
}

/// Single-octave improved Perlin noise at (x, y). Returns [-1, 1].
pub fn perlin_2d(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32 & 255;
    let yi = y.floor() as i32 & 255;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = fade(xf);
    let v = fade(yf);

    let aa = perm[(perm[xi as usize] as i32 + yi) as usize & 511];
    let ab = perm[(perm[xi as usize] as i32 + yi + 1) as usize & 511];
    let ba = perm[(perm[(xi + 1) as usize & 255] as i32 + yi) as usize & 511];
    let bb = perm[(perm[(xi + 1) as usize & 255] as i32 + yi + 1) as usize & 511];

    lerp(
        v,
        lerp(u, grad_perlin(aa, xf, yf), grad_perlin(ba, xf - 1.0, yf)),
        lerp(
            u,
            grad_perlin(ab, xf, yf - 1.0),
            grad_perlin(bb, xf - 1.0, yf - 1.0),
        ),
    )
}

#[cfg(target_arch = "wasm32")]
pub fn perlin_2d_f32(perm: &[u8; 512], x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32 & 255;
    let yi = y.floor() as i32 & 255;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = fade_f32(xf);
    let v = fade_f32(yf);
    let aa = perm[(perm[xi as usize] as i32 + yi) as usize & 511];
    let ab = perm[(perm[xi as usize] as i32 + yi + 1) as usize & 511];
    let ba = perm[(perm[(xi + 1) as usize & 255] as i32 + yi) as usize & 511];
    let bb = perm[(perm[(xi + 1) as usize & 255] as i32 + yi + 1) as usize & 511];
    lerp_f32(
        v,
        lerp_f32(
            u,
            grad_perlin_f32(aa, xf, yf),
            grad_perlin_f32(ba, xf - 1.0, yf),
        ),
        lerp_f32(
            u,
            grad_perlin_f32(ab, xf, yf - 1.0),
            grad_perlin_f32(bb, xf - 1.0, yf - 1.0),
        ),
    )
}

/// Compute BT.601 luminance from 0–255 RGB values.
#[inline]
pub fn pixel_luminance(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Knuth's algorithm for Poisson random variates (small lambda ≤ 30).
/// For larger lambda, use normal approximation.
#[inline]
pub fn poisson_random(lambda: f64, rng: &mut u64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda > 30.0 {
        // Normal approximation: Poisson(λ) ≈ N(λ, λ)
        return (lambda + box_muller(rng) * lambda.sqrt()).max(0.0);
    }
    // Knuth's algorithm
    let l = (-lambda).exp();
    let mut k = 0.0f64;
    let mut p = 1.0f64;
    loop {
        k += 1.0;
        p *= xorshift64_f64(rng);
        if p <= l {
            return k - 1.0;
        }
    }
}

/// Gaussian pyramid downsample: blur + subsample by 2.
///
/// Applies a 5x5 Gaussian kernel then takes every other pixel.
/// Output is (w+1)/2 x (h+1)/2. Matches `cv2.pyrDown`.
pub fn pyr_down(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "pyr_down requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let ow = w.div_ceil(2);
    let oh = h.div_ceil(2);

    // 5x5 Gaussian kernel (1/256 normalization): [1,4,6,4,1] x [1,4,6,4,1]
    let kernel_1d: [i32; 5] = [1, 4, 6, 4, 1];

    // Horizontal pass → temp buffer
    let mut temp = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sx = reflect101(x as isize + k as isize - 2, w as isize) as usize;
                sum += pixels[y * w + sx] as i32 * kernel_1d[k as usize];
            }
            temp[y * w + x] = sum;
        }
    }

    // Vertical pass + subsample
    let mut output = vec![0u8; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let x = ox * 2;
            let y = oy * 2;
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sy = reflect101(y as isize + k as isize - 2, h as isize) as usize;
                sum += temp[sy * w + x] * kernel_1d[k as usize];
            }
            // Normalize by 256 (16*16)
            output[oy * ow + ox] = ((sum + 128) >> 8).clamp(0, 255) as u8;
        }
    }

    let new_info = ImageInfo {
        width: ow as u32,
        height: oh as u32,
        format: info.format,
        color_space: info.color_space,
    };
    Ok((output, new_info))
}

/// pyrDown for single-channel f32 image.
/// Applies 5×5 Gaussian blur then subsamples by 2 in each dimension.
/// Border handling: BORDER_REFLECT_101 (default OpenCV border for pyrDown).
pub fn pyr_down_gray(src: &[f32], sw: u32, sh: u32) -> (Vec<f32>, u32, u32) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let sws = sw as isize;
    let shs = sh as isize;

    // Horizontal pass → temp (sh × dw)
    let mut tmp = vec![0.0f32; sh * dw];
    for y in 0..sh {
        for dx in 0..dw {
            let sx = (dx * 2) as isize;
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let col = reflect101_safe(sx + k - 2, sws);
                sum += PYR_KERNEL[k as usize] * src[y * sw + col];
            }
            tmp[y * dw + dx] = sum;
        }
    }

    // Vertical pass → dst (dh × dw)
    let mut dst = vec![0.0f32; dh * dw];
    for dy in 0..dh {
        let sy = (dy * 2) as isize;
        for x in 0..dw {
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let row = reflect101_safe(sy + k - 2, shs);
                sum += PYR_KERNEL[k as usize] * tmp[row * dw + x];
            }
            dst[dy * dw + x] = sum;
        }
    }

    (dst, dw as u32, dh as u32)
}

/// pyrDown for 3-channel f32 image (interleaved RGB).
pub fn pyr_down_rgb(src: &[f32], sw: u32, sh: u32) -> (Vec<f32>, u32, u32) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let sws = sw as isize;
    let shs = sh as isize;

    // Horizontal pass
    let mut tmp = vec![0.0f32; sh * dw * 3];
    for y in 0..sh {
        for dx in 0..dw {
            let sx = (dx * 2) as isize;
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let col = reflect101_safe(sx + k - 2, sws);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * src[(y * sw + col) * 3];
                sum[1] += wt * src[(y * sw + col) * 3 + 1];
                sum[2] += wt * src[(y * sw + col) * 3 + 2];
            }
            tmp[(y * dw + dx) * 3] = sum[0];
            tmp[(y * dw + dx) * 3 + 1] = sum[1];
            tmp[(y * dw + dx) * 3 + 2] = sum[2];
        }
    }

    // Vertical pass
    let mut dst = vec![0.0f32; dh * dw * 3];
    for dy in 0..dh {
        let sy = (dy * 2) as isize;
        for x in 0..dw {
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let row = reflect101_safe(sy + k - 2, shs);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * tmp[(row * dw + x) * 3];
                sum[1] += wt * tmp[(row * dw + x) * 3 + 1];
                sum[2] += wt * tmp[(row * dw + x) * 3 + 2];
            }
            dst[(dy * dw + x) * 3] = sum[0];
            dst[(dy * dw + x) * 3 + 1] = sum[1];
            dst[(dy * dw + x) * 3 + 2] = sum[2];
        }
    }

    (dst, dw as u32, dh as u32)
}

/// Gaussian pyramid upsample: upsample by 2 + blur.
///
/// Inserts zeros between pixels, then applies 5x5 Gaussian * 4.
/// Output is w*2 x h*2. Matches `cv2.pyrUp`.
pub fn pyr_up(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "pyr_up requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let ow = w * 2;
    let oh = h * 2;

    // Upsample: insert zeros
    let mut upsampled = vec![0i32; ow * oh];
    for y in 0..h {
        for x in 0..w {
            upsampled[y * 2 * ow + x * 2] = pixels[y * w + x] as i32 * 4; // *4 to compensate for zero-insertion
        }
    }

    // 5x5 Gaussian blur on upsampled
    let kernel_1d: [i32; 5] = [1, 4, 6, 4, 1];

    // Horizontal pass
    let mut temp = vec![0i32; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sx = reflect101(x as isize + k as isize - 2, ow as isize) as usize;
                sum += upsampled[y * ow + sx] * kernel_1d[k as usize];
            }
            temp[y * ow + x] = sum;
        }
    }

    // Vertical pass
    let mut output = vec![0u8; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sy = reflect101(y as isize + k as isize - 2, oh as isize) as usize;
                sum += temp[sy * ow + x] * kernel_1d[k as usize];
            }
            output[y * ow + x] = ((sum + 128) >> 8).clamp(0, 255) as u8;
        }
    }

    let new_info = ImageInfo {
        width: ow as u32,
        height: oh as u32,
        format: info.format,
        color_space: info.color_space,
    };
    Ok((output, new_info))
}

/// pyrUp for single-channel f32 — upsample by 2 then apply 5×5 Gaussian × 4.
#[allow(dead_code)] // reserved for pyramid reconstruction path
pub fn pyr_up_gray(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = dw as usize;
    let dh = dh as usize;
    let dws = dw as isize;
    let dhs = dh as isize;

    // Insert zeros: place src pixels at even positions, zeros at odd
    let mut upsampled = vec![0.0f32; dh * dw];
    for y in 0..sh {
        for x in 0..sw {
            if y * 2 < dh && x * 2 < dw {
                upsampled[y * 2 * dw + x * 2] = src[y * sw + x] * 4.0;
            }
        }
    }

    // Apply 5×5 Gaussian filter (separable)
    let mut tmp = vec![0.0f32; dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let col = reflect101_safe(x as isize + k - 2, dws);
                sum += PYR_KERNEL[k as usize] * upsampled[y * dw + col];
            }
            tmp[y * dw + x] = sum;
        }
    }

    let mut dst = vec![0.0f32; dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let row = reflect101_safe(y as isize + k - 2, dhs);
                sum += PYR_KERNEL[k as usize] * tmp[row * dw + x];
            }
            dst[y * dw + x] = sum;
        }
    }

    dst
}

/// pyrUp for 3-channel f32 (interleaved RGB).
pub fn pyr_up_rgb(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = dw as usize;
    let dh = dh as usize;
    let dws = dw as isize;
    let dhs = dh as isize;

    // Insert zeros with 4× scaling at even positions
    let mut upsampled = vec![0.0f32; dh * dw * 3];
    for y in 0..sh {
        for x in 0..sw {
            if y * 2 < dh && x * 2 < dw {
                let di = (y * 2 * dw + x * 2) * 3;
                let si = (y * sw + x) * 3;
                upsampled[di] = src[si] * 4.0;
                upsampled[di + 1] = src[si + 1] * 4.0;
                upsampled[di + 2] = src[si + 2] * 4.0;
            }
        }
    }

    // Horizontal pass
    let mut tmp = vec![0.0f32; dh * dw * 3];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let col = reflect101_safe(x as isize + k - 2, dws);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * upsampled[(y * dw + col) * 3];
                sum[1] += wt * upsampled[(y * dw + col) * 3 + 1];
                sum[2] += wt * upsampled[(y * dw + col) * 3 + 2];
            }
            tmp[(y * dw + x) * 3] = sum[0];
            tmp[(y * dw + x) * 3 + 1] = sum[1];
            tmp[(y * dw + x) * 3 + 2] = sum[2];
        }
    }

    // Vertical pass
    let mut dst = vec![0.0f32; dh * dw * 3];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let row = reflect101_safe(y as isize + k - 2, dhs);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * tmp[(row * dw + x) * 3];
                sum[1] += wt * tmp[(row * dw + x) * 3 + 1];
                sum[2] += wt * tmp[(row * dw + x) * 3 + 2];
            }
            dst[(y * dw + x) * 3] = sum[0];
            dst[(y * dw + x) * 3 + 1] = sum[1];
            dst[(y * dw + x) * 3 + 2] = sum[2];
        }
    }

    dst
}

/// Process a single channel through the Local Laplacian pyramid.
pub fn pyramid_detail_remap_channel(
    input: &[f32],
    w: usize,
    h: usize,
    levels: usize,
    sigma: f32,
) -> Vec<f32> {
    // Build Gaussian pyramid
    let mut gauss_pyramid = vec![input.to_vec()];
    let mut cw = w;
    let mut ch = h;
    for _ in 1..levels {
        let prev = gauss_pyramid.last().unwrap();
        let (nw, nh) = (cw.div_ceil(2), ch.div_ceil(2));
        let downsampled = downsample_2x(prev, cw, ch);
        gauss_pyramid.push(downsampled);
        cw = nw;
        ch = nh;
    }

    // Build output Laplacian pyramid with remapped detail
    let mut output_laplacian: Vec<Vec<f32>> = Vec::with_capacity(levels);
    cw = w;
    ch = h;

    for level in 0..levels - 1 {
        let (nw, nh) = (cw.div_ceil(2), ch.div_ceil(2));

        // Laplacian = current level - upsampled(next level)
        let upsampled = upsample_2x(&gauss_pyramid[level + 1], nw, nh, cw, ch);
        let mut laplacian = vec![0.0f32; cw * ch];
        #[allow(clippy::needless_range_loop)]
        for i in 0..cw * ch {
            laplacian[i] = gauss_pyramid[level][i] - upsampled[i];
        }

        // Remap detail: attenuate or amplify based on sigma
        // Enhancement: small sigma compresses large gradients, preserves small detail
        // Smoothing: large sigma suppresses small detail
        for laplacian_val in laplacian.iter_mut().take(cw * ch) {
            let d = *laplacian_val;
            // Sigmoidal remapping: f(d) = d * (sigma / (sigma + |d|))
            // sigma < 1: enhances small detail (compresses large)
            // sigma > 1: smooths (suppresses small detail)
            *laplacian_val = d * sigma / (sigma + d.abs());
        }

        output_laplacian.push(laplacian);
        cw = nw;
        ch = nh;
    }

    // Coarsest level is kept as-is (DC component)
    output_laplacian.push(gauss_pyramid[levels - 1].clone());

    // Reconstruct from Laplacian pyramid
    let mut reconstructed = output_laplacian[levels - 1].clone();
    let _ = gauss_pyramid[levels - 1].len(); // dims recalculated below

    // Recompute dimensions for each level
    let mut dims: Vec<(usize, usize)> = Vec::with_capacity(levels);
    let (mut tw, mut th) = (w, h);
    for _ in 0..levels {
        dims.push((tw, th));
        tw = tw.div_ceil(2);
        th = th.div_ceil(2);
    }

    for level in (0..levels - 1).rev() {
        let (target_w, target_h) = dims[level];
        let (src_w, src_h) = dims[level + 1];
        let upsampled = upsample_2x(&reconstructed, src_w, src_h, target_w, target_h);
        reconstructed = vec![0.0f32; target_w * target_h];
        for i in 0..target_w * target_h {
            reconstructed[i] = (upsampled[i] + output_laplacian[level][i]).clamp(0.0, 1.0);
        }
    }

    reconstructed
}

/// Reflect-edge coordinate clamping.
pub fn reflect(v: i32, size: usize) -> usize {
    if v < 0 {
        (-v).min(size as i32 - 1) as usize
    } else if v >= size as i32 {
        (2 * size as i32 - v - 2).max(0) as usize
    } else {
        v as usize
    }
}

/// BORDER_REFLECT_101: reflect at boundary without duplicating edge pixel.
/// Matches OpenCV's default border mode.
#[inline(always)]
pub fn reflect101(idx: isize, size: isize) -> isize {
    if idx < 0 {
        -idx
    } else if idx >= size {
        2 * size - 2 - idx
    } else {
        idx
    }
}

/// BORDER_REFLECT_101 with clamping for small sizes.
/// Handles the case where a single reflection is insufficient (e.g., idx=-2 with size=2).
#[inline]
pub fn reflect101_safe(idx: isize, size: isize) -> usize {
    if size <= 1 {
        return 0;
    }
    let mut i = idx;
    // Bring into range [-(size-1), 2*(size-1)] first
    let cycle = 2 * (size - 1);
    if i < 0 {
        i = -i;
    }
    if i >= cycle {
        i %= cycle;
    }
    if i >= size {
        i = cycle - i;
    }
    i as usize
}

/// Multi-Scale Retinex (MSR).
///
/// Averages SSR at multiple Gaussian scales for balanced enhancement across
/// fine and coarse detail. Default scales: sigma = [15, 80, 250].
///
/// `MSR(x,y) = (1/N) * sum_i [log(I(x,y)) - log(G(x,y,sigma_i) * I(x,y))]`
///
/// - `sigmas`: Gaussian scales (typical: &[15.0, 80.0, 250.0])
///
/// Reference: Jobson, Rahman, Woodell — "A Multiscale Retinex for Bridging
/// the Gap Between Color Images and the Human Observation of Scenes"
/// (IEEE Trans. Image Processing, 1997)
pub fn retinex_msr(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    sigmas: &[f32],
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "retinex requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let num_scales = sigmas.len() as f32;

    // Accumulate retinex across scales
    let mut retinex = vec![0.0f32; n * 3];

    for &sigma in sigmas {
        let blurred = {
            let r = Rect::new(0, 0, info.width, info.height);
            let mut u = |_: Rect| Ok(pixels.to_vec());
            gaussian_blur_cv(r, &mut u, info, &GaussianBlurCvParams { sigma })?
        };
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                let orig = (pixels[pi + c] as f32).max(1.0);
                let surround = (blurred[pi + c] as f32).max(1.0);
                retinex[i * 3 + c] += (orig.ln() - surround.ln()) / num_scales;
            }
        }
    }

    // Normalize to [0, 255]
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in &retinex {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let range = (max_val - min_val).max(1e-6);

    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let pi = i * channels;
        for c in 0..3 {
            let v = (retinex[i * 3 + c] - min_val) / range * 255.0;
            result[pi + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3];
        }
    }

    Ok(result)
}

/// Multi-Scale Retinex with Color Restoration (MSRCR).
///
/// Extends MSR with a color restoration factor that prevents desaturation:
/// `MSRCR(x,y) = C(x,y) * MSR(x,y)`
/// where `C(x,y) = beta * log(alpha * I_c / sum(I_channels))`
///
/// - `sigmas`: Gaussian scales (typical: &[15.0, 80.0, 250.0])
/// - `alpha`: color restoration nonlinearity (typical: 125.0)
/// - `beta`: color restoration gain (typical: 46.0)
///
/// Reference: Jobson, Rahman, Woodell — "A Multiscale Retinex for Bridging
/// the Gap Between Color Images and the Human Observation of Scenes"
/// (IEEE Trans. Image Processing, 1997)
pub fn retinex_msrcr(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    sigmas: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "retinex requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);
    let num_scales = sigmas.len() as f32;

    // Compute MSR (OpenCV-compatible blur for reference alignment)
    let mut msr = vec![0.0f32; n * 3];
    for &sigma in sigmas {
        let blurred = {
            let r = Rect::new(0, 0, info.width, info.height);
            let mut u = |_: Rect| Ok(pixels.to_vec());
            gaussian_blur_cv(r, &mut u, info, &GaussianBlurCvParams { sigma })?
        };
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                let orig = (pixels[pi + c] as f32).max(1.0);
                let surround = (blurred[pi + c] as f32).max(1.0);
                msr[i * 3 + c] += (orig.ln() - surround.ln()) / num_scales;
            }
        }
    }

    // Color restoration: C(x,y) = beta * log(alpha * I_c / sum(I))
    let mut msrcr = vec![0.0f32; n * 3];
    for i in 0..n {
        let pi = i * channels;
        let sum_channels = pixels[pi] as f32 + pixels[pi + 1] as f32 + pixels[pi + 2] as f32;
        let sum_channels = sum_channels.max(1.0);
        for c in 0..3 {
            let ic = (pixels[pi + c] as f32).max(1.0);
            let color_restore = beta * (alpha * ic / sum_channels).ln();
            msrcr[i * 3 + c] = color_restore * msr[i * 3 + c];
        }
    }

    // Normalize to [0, 255]
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in &msrcr {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let range = (max_val - min_val).max(1e-6);

    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let pi = i * channels;
        for c in 0..3 {
            let v = (msrcr[i * 3 + c] - min_val) / range * 255.0;
            result[pi + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3];
        }
    }

    Ok(result)
}

/// Saturation = max(C) - min(C).
#[inline]
pub fn sat(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

/// Scharr edge detection (internal — returns raw Gray8 bytes).
pub fn scharr(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, scharr);
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;
    let gray = to_grayscale(pixels, channels);
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let r0 = y * pw;
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            let p00 = padded[r0 + x] as f32;
            let p01 = padded[r0 + x + 1] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p10 = padded[r1 + x] as f32;
            let p12 = padded[r1 + x + 2] as f32;
            let p20 = padded[r2 + x] as f32;
            let p21 = padded[r2 + x + 1] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            // Scharr Gx = [[-3,0,3],[-10,0,10],[-3,0,3]]
            let gx = -3.0 * p00 + 3.0 * p02 - 10.0 * p10 + 10.0 * p12 - 3.0 * p20 + 3.0 * p22;
            // Scharr Gy = [[-3,-10,-3],[0,0,0],[3,10,3]]
            let gy = -3.0 * p00 - 10.0 * p01 - 3.0 * p02 + 3.0 * p20 + 10.0 * p21 + 3.0 * p22;

            out[y * w + x] = (gx * gx + gy * gy).sqrt().min(255.0) as u8;
        }
    }
    Ok(out)
}

/// Set luminance of a color to `target_l`, then clip.
#[inline]
pub fn set_lum(r: f32, g: f32, b: f32, target_l: f32) -> (f32, f32, f32) {
    let d = target_l - lum(r, g, b);
    clip_color(r + d, g + d, b + d)
}

/// Combined SetSat + SetLum: set saturation then luminance of a color.
#[inline]
pub fn set_lum_sat(r: f32, g: f32, b: f32, target_s: f32, target_l: f32) -> (f32, f32, f32) {
    let (r2, g2, b2) = set_sat(r, g, b, target_s);
    set_lum(r2, g2, b2, target_l)
}

/// Set saturation of color `c` to `target_s`.
///
/// Sorts channels, scales mid proportionally, zeros min, sets max to target_s.
#[inline]
pub fn set_sat(r: f32, g: f32, b: f32, target_s: f32) -> (f32, f32, f32) {
    // Identify min/mid/max channels by index (0=R, 1=G, 2=B)
    let c = [r, g, b];
    let (min_i, mid_i, max_i) = if c[0] <= c[1] {
        if c[1] <= c[2] {
            (0, 1, 2)
        } else if c[0] <= c[2] {
            (0, 2, 1)
        } else {
            (2, 0, 1)
        }
    } else if c[0] <= c[2] {
        (1, 0, 2)
    } else if c[1] <= c[2] {
        (1, 2, 0)
    } else {
        (2, 1, 0)
    };

    let mut out = [0.0f32; 3];
    if c[max_i] > c[min_i] {
        out[mid_i] = (c[mid_i] - c[min_i]) * target_s / (c[max_i] - c[min_i]);
        out[max_i] = target_s;
    }
    // out[min_i] stays 0.0
    (out[0], out[1], out[2])
}

pub fn sharpen_impl(
    pixels: &[u8],
    info: &ImageInfo,
    config: &SharpenParams,
) -> Result<Vec<u8>, ImageError> {
    let amount = config.amount;

    validate_format(info.format)?;

    // 16-bit: work in f32 for full precision
    if is_16bit(info.format) {
        let orig_f32 = u16_pixels_to_f32(pixels);
        let _info_8 = ImageInfo {
            format: match info.format {
                PixelFormat::Rgb16 => PixelFormat::Rgb8,
                PixelFormat::Rgba16 => PixelFormat::Rgba8,
                PixelFormat::Gray16 => PixelFormat::Gray8,
                other => other,
            },
            ..*info
        };
        let blurred = blur_impl(pixels, info, &BlurParams { radius: 1.0 })?;
        let blur_f32 = u16_pixels_to_f32(&blurred);
        let result_f32: Vec<f32> = orig_f32
            .iter()
            .zip(blur_f32.iter())
            .map(|(&o, &b)| (o + amount * (o - b)).clamp(0.0, 1.0))
            .collect();
        return Ok(f32_to_u16_pixels(&result_f32));
    }

    // Blur with a small radius for the unsharp mask
    let blurred = blur_impl(pixels, info, &BlurParams { radius: 1.0 })?;

    let mut result = vec![0u8; pixels.len()];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let amount_vec = f32x4_splat(amount);
        let zero = f32x4_splat(0.0);
        let max_val = f32x4_splat(255.0);
        let len = pixels.len();
        let chunks = len / 4;

        for i in 0..chunks {
            let base = i * 4;
            // Load 4 bytes, widen to f32x4
            let orig = f32x4(
                pixels[base] as f32,
                pixels[base + 1] as f32,
                pixels[base + 2] as f32,
                pixels[base + 3] as f32,
            );
            let blur_v = f32x4(
                blurred[base] as f32,
                blurred[base + 1] as f32,
                blurred[base + 2] as f32,
                blurred[base + 3] as f32,
            );
            let diff = f32x4_sub(orig, blur_v);
            let scaled = f32x4_mul(diff, amount_vec);
            let sharp = f32x4_add(orig, scaled);
            let clamped = f32x4_max(zero, f32x4_min(max_val, sharp));

            result[base] = f32x4_extract_lane::<0>(clamped) as u8;
            result[base + 1] = f32x4_extract_lane::<1>(clamped) as u8;
            result[base + 2] = f32x4_extract_lane::<2>(clamped) as u8;
            result[base + 3] = f32x4_extract_lane::<3>(clamped) as u8;
        }
        // Remainder
        for i in chunks * 4..len {
            let orig = pixels[i] as f32;
            let blur_val = blurred[i] as f32;
            let sharp = orig + amount * (orig - blur_val);
            result[i] = sharp.clamp(0.0, 255.0) as u8;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..pixels.len() {
            let orig = pixels[i] as f32;
            let blur_val = blurred[i] as f32;
            let sharp = orig + amount * (orig - blur_val);
            result[i] = sharp.clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}

/// Single-octave OpenSimplex noise at (x, y). Returns approximately [-1, 1].
pub fn simplex_2d(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    let stretch = (x + y) * SIMPLEX_STRETCH;
    let xs = x + stretch;
    let ys = y + stretch;

    let xsb = xs.floor() as i32;
    let ysb = ys.floor() as i32;

    let squish = (xsb + ysb) as f64 * SIMPLEX_SQUISH;
    let xb = xsb as f64 + squish;
    let yb = ysb as f64 + squish;

    let dx0 = x - xb;
    let dy0 = y - yb;

    let xins = xs - xsb as f64;
    let yins = ys - ysb as f64;

    let mut value = 0.0f64;

    // Contribution from (0, 0)
    let at0 = 2.0 - dx0 * dx0 - dy0 * dy0;
    if at0 > 0.0 {
        let at0 = at0 * at0;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += at0 * at0 * (SIMPLEX_GRADS[gi].0 * dx0 + SIMPLEX_GRADS[gi].1 * dy0);
    }

    // Contribution from (1, 0)
    let dx1 = dx0 - 1.0 - SIMPLEX_SQUISH;
    let dy1 = dy0 - SIMPLEX_SQUISH;
    let at1 = 2.0 - dx1 * dx1 - dy1 * dy1;
    if at1 > 0.0 {
        let at1 = at1 * at1;
        let gi = perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += at1 * at1 * (SIMPLEX_GRADS[gi].0 * dx1 + SIMPLEX_GRADS[gi].1 * dy1);
    }

    // Contribution from (0, 1)
    let dx2 = dx0 - SIMPLEX_SQUISH;
    let dy2 = dy0 - 1.0 - SIMPLEX_SQUISH;
    let at2 = 2.0 - dx2 * dx2 - dy2 * dy2;
    if at2 > 0.0 {
        let at2 = at2 * at2;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
        value += at2 * at2 * (SIMPLEX_GRADS[gi].0 * dx2 + SIMPLEX_GRADS[gi].1 * dy2);
    }

    // Contribution from (1, 1) — only if in the upper triangle
    if xins + yins > 1.0 {
        let dx3 = dx0 - 1.0 - 2.0 * SIMPLEX_SQUISH;
        let dy3 = dy0 - 1.0 - 2.0 * SIMPLEX_SQUISH;
        let at3 = 2.0 - dx3 * dx3 - dy3 * dy3;
        if at3 > 0.0 {
            let at3 = at3 * at3;
            let gi =
                perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
            value += at3 * at3 * (SIMPLEX_GRADS[gi].0 * dx3 + SIMPLEX_GRADS[gi].1 * dy3);
        }
    }

    // Scale to approximately [-1, 1]
    value * 47.0
}

#[cfg(target_arch = "wasm32")]
pub fn simplex_2d_f32(perm: &[u8; 512], x: f32, y: f32) -> f32 {
    const SS: f32 = -0.211324865;
    const SQ: f32 = 0.366025404;
    const SR2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const GR: [(f32, f32); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (SR2, SR2),
        (-SR2, SR2),
        (SR2, -SR2),
        (-SR2, -SR2),
    ];
    let stretch = (x + y) * SS;
    let xs = x + stretch;
    let ys = y + stretch;
    let xsb = xs.floor() as i32;
    let ysb = ys.floor() as i32;
    let squish = (xsb + ysb) as f32 * SQ;
    let dx0 = x - (xsb as f32 + squish);
    let dy0 = y - (ysb as f32 + squish);
    let xins = xs - xsb as f32;
    let yins = ys - ysb as f32;
    let mut value = 0.0f32;
    let a0 = 2.0 - dx0 * dx0 - dy0 * dy0;
    if a0 > 0.0 {
        let a = a0 * a0;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += a * a * (GR[gi].0 * dx0 + GR[gi].1 * dy0);
    }
    let d1x = dx0 - 1.0 - SQ;
    let d1y = dy0 - SQ;
    let a1 = 2.0 - d1x * d1x - d1y * d1y;
    if a1 > 0.0 {
        let a = a1 * a1;
        let gi = perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb) as usize & 511] as usize & 7;
        value += a * a * (GR[gi].0 * d1x + GR[gi].1 * d1y);
    }
    let d2x = dx0 - SQ;
    let d2y = dy0 - 1.0 - SQ;
    let a2 = 2.0 - d2x * d2x - d2y * d2y;
    if a2 > 0.0 {
        let a = a2 * a2;
        let gi = perm[(perm[xsb as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
        value += a * a * (GR[gi].0 * d2x + GR[gi].1 * d2y);
    }
    if xins + yins > 1.0 {
        let d3x = dx0 - 1.0 - 2.0 * SQ;
        let d3y = dy0 - 1.0 - 2.0 * SQ;
        let a3 = 2.0 - d3x * d3x - d3y * d3y;
        if a3 > 0.0 {
            let a = a3 * a3;
            let gi =
                perm[(perm[(xsb + 1) as usize & 255] as i32 + ysb + 1) as usize & 511] as usize & 7;
            value += a * a * (GR[gi].0 * d3x + GR[gi].1 * d3y);
        }
    }
    value * 47.0
}

/// Sobel edge detection (internal — returns raw Gray8 bytes).
pub fn sobel(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, sobel);
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

    let gray = to_grayscale(pixels, channels);

    // Pad with 1-pixel reflected border to eliminate boundary checks
    let padded = pad_reflect(&gray, w, h, 1, 1);
    let pw = w + 2;
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        let r0 = y * pw; // row above (in padded coords, offset by pad=1 → y+1-1 = y)
        let r1 = (y + 1) * pw;
        let r2 = (y + 2) * pw;
        for x in 0..w {
            // Direct Sobel — unrolled 3x3, no loop
            // Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
            let p00 = padded[r0 + x] as f32;
            let p02 = padded[r0 + x + 2] as f32;
            let p10 = padded[r1 + x] as f32;
            let p12 = padded[r1 + x + 2] as f32;
            let p20 = padded[r2 + x] as f32;
            let p22 = padded[r2 + x + 2] as f32;

            let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;

            // Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
            let p01 = padded[r0 + x + 1] as f32;
            let p21 = padded[r2 + x + 1] as f32;

            let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

            out[y * w + x] = (gx * gx + gy * gy).sqrt().min(255.0) as u8;
        }
    }
    Ok(out)
}

pub fn solve_homography_4pt(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Option<[f64; 9]> {
    // Build 8×8 system A*x = b, where x = [c00, c01, c02, c10, c11, c12, c20, c21]
    // and c22 = 1 (assumed).
    let mut a = [0.0f64; 8 * 8];
    let mut b = [0.0f64; 8];

    for i in 0..4 {
        let (sx, sy) = (src[i].0 as f64, src[i].1 as f64);
        let (dx, dy) = (dst[i].0 as f64, dst[i].1 as f64);

        // Row i: x-equation → c00*sx + c01*sy + c02 - c20*sx*dx - c21*sy*dx = dx
        a[i * 8] = sx;
        a[i * 8 + 1] = sy;
        a[i * 8 + 2] = 1.0;
        // a[i*8+3..5] = 0 (c10, c11, c12 terms)
        a[i * 8 + 6] = -sx * dx;
        a[i * 8 + 7] = -sy * dx;
        b[i] = dx;

        // Row i+4: y-equation → c10*sx + c11*sy + c12 - c20*sx*dy - c21*sy*dy = dy
        // a[(i+4)*8+0..2] = 0 (c00, c01, c02 terms)
        a[(i + 4) * 8 + 3] = sx;
        a[(i + 4) * 8 + 4] = sy;
        a[(i + 4) * 8 + 5] = 1.0;
        a[(i + 4) * 8 + 6] = -sx * dy;
        a[(i + 4) * 8 + 7] = -sy * dy;
        b[i + 4] = dy;
    }

    // Solve via Gaussian elimination with partial pivoting (DECOMP_LU)
    let n = 8usize;
    let mut aug = [0.0f64; 8 * 9]; // augmented [A|b]
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a[r * n + c];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    for col in 0..n {
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular — would need SVD fallback
        }
        if max_row != col {
            for c in 0..(n + 1) {
                aug.swap(col * (n + 1) + c, max_row * (n + 1) + c);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for c in col..(n + 1) {
            aug[col * (n + 1) + c] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * (n + 1) + col];
            for c in col..(n + 1) {
                aug[row * (n + 1) + c] -= factor * aug[col * (n + 1) + c];
            }
        }
    }

    // Extract solution: x[i] = aug[i][n]
    // Pack into 3×3 with M[8] = 1.0 (c22 = 1)
    #[allow(clippy::identity_op, clippy::erasing_op)]
    Some([
        aug[0 * (n + 1) + n], // c00
        aug[1 * (n + 1) + n], // c01
        aug[2 * (n + 1) + n], // c02
        aug[3 * (n + 1) + n], // c10
        aug[4 * (n + 1) + n], // c11
        aug[5 * (n + 1) + n], // c12
        aug[6 * (n + 1) + n], // c20
        aug[7 * (n + 1) + n], // c21
        1.0,                  // c22
    ])
}

/// Solve a 3x3 perspective transform from 4 point correspondences.
///
/// Matches OpenCV's `getPerspectiveTransform` exactly:
/// - Formulates A*x = b with c22=1 assumption
/// - Solves via LU decomposition (Gaussian elimination with partial pivoting)
/// - Row ordering: x-equations first (rows 0–3), then y-equations (rows 4–7)
///
/// Returns the 3×3 matrix (row-major) mapping src → dst.
/// Reference: OpenCV 4.x modules/imgproc/src/imgwarp.cpp getPerspectiveTransform
/// Public wrapper for integration tests.
pub fn solve_homography_4pt_public(
    src: &[(f32, f32); 4],
    dst: &[(f32, f32); 4],
) -> Option<[f64; 9]> {
    solve_homography_4pt(src, dst)
}

/// Solve overdetermined linear system via normal equations (A^T A x = A^T b).
/// Uses Cholesky-like Gaussian elimination on the normal equations.
pub fn solve_least_squares(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    // Form A^T A (n×n) and A^T b (n)
    let mut ata = vec![0.0f64; n * n];
    let mut atb = vec![0.0f64; n];

    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0f64;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
            ata[j * n + i] = sum;
        }
        let mut sum = 0.0f64;
        for k in 0..m {
            sum += a[k * n + i] * b[k];
        }
        atb[i] = sum;
    }

    // Gaussian elimination with partial pivoting
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = ata[i * n + j];
        }
        aug[i * (n + 1) + n] = atb[i];
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        x[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }

    x
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

/// 3-pass stackable box blur approximating a Gaussian with the given sigma.
///
/// Operates on an f32 buffer (single channel). O(1) per pixel regardless of sigma.
pub fn stackable_box_blur_f32(data: &mut [f32], w: usize, h: usize, sigma: f32) {
    let radii = box_blur_radii_for_gaussian(sigma);
    for &r in &radii {
        box_blur_pass_f32(data, w, h, r);
    }
}

/// Convert multi-channel pixels to single-channel grayscale.
/// Convert multi-channel pixels to single-channel grayscale.
///
/// Uses BT.601 fixed-point: (77*R + 150*G + 29*B + 128) >> 8.
/// Integer-only arithmetic — no floating point in the hot path.
pub fn to_grayscale(pixels: &[u8], channels: usize) -> Vec<u8> {
    if channels == 1 {
        return pixels.to_vec();
    }
    let pixel_count = pixels.len() / channels;

    #[cfg(target_arch = "wasm32")]
    {
        to_grayscale_simd128(pixels, channels, pixel_count)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        to_grayscale_scalar(pixels, channels, pixel_count)
    }
}

#[allow(dead_code)] // scalar fallback, SIMD path used on wasm32
pub fn to_grayscale_scalar(pixels: &[u8], channels: usize, pixel_count: usize) -> Vec<u8> {
    let mut gray = Vec::with_capacity(pixel_count);
    // BT.601 fixed-point: 77/256 ≈ 0.3008, 150/256 ≈ 0.5859, 29/256 ≈ 0.1133
    for i in 0..pixel_count {
        let r = pixels[i * channels] as u32;
        let g = pixels[i * channels + 1] as u32;
        let b = pixels[i * channels + 2] as u32;
        gray.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
    }
    gray
}

#[cfg(target_arch = "wasm32")]
pub fn to_grayscale_simd128(pixels: &[u8], channels: usize, pixel_count: usize) -> Vec<u8> {
    use std::arch::wasm32::*;

    let mut gray = vec![0u8; pixel_count];

    // Process 4 pixels at a time using i32x4 multiply-accumulate
    let chunks = pixel_count / 4;
    let coeff_r = i32x4_splat(77);
    let coeff_g = i32x4_splat(150);
    let coeff_b = i32x4_splat(29);
    let round = i32x4_splat(128);

    for chunk in 0..chunks {
        let out_base = chunk * 4;

        // Load 4 pixels, extract R/G/B channels
        let mut rv = [0i32; 4];
        let mut gv = [0i32; 4];
        let mut bv = [0i32; 4];
        for p in 0..4 {
            let base = (out_base + p) * channels;
            rv[p] = pixels[base] as i32;
            gv[p] = pixels[base + 1] as i32;
            bv[p] = pixels[base + 2] as i32;
        }

        // SAFETY: rv/gv/bv are [i32; 4] on stack, properly aligned for v128_load
        unsafe {
            let r = v128_load(rv.as_ptr() as *const v128);
            let g = v128_load(gv.as_ptr() as *const v128);
            let b = v128_load(bv.as_ptr() as *const v128);

            // Y = (77*R + 150*G + 29*B + 128) >> 8
            let sum = i32x4_add(
                i32x4_add(i32x4_mul(coeff_r, r), i32x4_mul(coeff_g, g)),
                i32x4_add(i32x4_mul(coeff_b, b), round),
            );
            let shifted = i32x4_shr(sum, 8);

            gray[out_base] = i32x4_extract_lane::<0>(shifted) as u8;
            gray[out_base + 1] = i32x4_extract_lane::<1>(shifted) as u8;
            gray[out_base + 2] = i32x4_extract_lane::<2>(shifted) as u8;
            gray[out_base + 3] = i32x4_extract_lane::<3>(shifted) as u8;
        }
    }

    // Handle remaining pixels
    for i in (chunks * 4)..pixel_count {
        let r = pixels[i * channels] as u32;
        let g = pixels[i * channels + 1] as u32;
        let b = pixels[i * channels + 2] as u32;
        gray[i] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    }

    gray
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

/// Trace a single border starting at (start_x, start_y) using Moore boundary tracing.
pub fn trace_border(
    img: &mut [i32],
    pw: usize,
    start_x: i32,
    start_y: i32,
    dx: &[i32; 8],
    dy: &[i32; 8],
    nbd: i32,
) -> Vec<(i32, i32)> {
    let mut contour = Vec::new();
    contour.push((start_x, start_y));

    // Find the first foreground neighbor going clockwise from the west direction
    let start_dir = 4; // start searching from west (the background side)
    let mut found_dir = None;
    for i in 0..8 {
        let d = (start_dir + i) % 8;
        let nx = start_x + dx[d];
        let ny = start_y + dy[d];
        let ni = ny as usize * pw + nx as usize;
        if img[ni] != 0 {
            found_dir = Some(d);
            break;
        }
    }

    let Some(mut dir) = found_dir else {
        // Isolated pixel
        img[start_y as usize * pw + start_x as usize] = -nbd;
        return contour;
    };

    let mut cx = start_x + dx[dir];
    let mut cy = start_y + dy[dir];

    if cx == start_x && cy == start_y {
        // Single pixel contour
        img[start_y as usize * pw + start_x as usize] = -nbd;
        return contour;
    }

    // Mark the start pixel
    img[start_y as usize * pw + start_x as usize] = nbd;

    let second_x = cx;
    let second_y = cy;

    loop {
        contour.push((cx, cy));
        img[cy as usize * pw + cx as usize] = nbd;

        // Search clockwise from (dir + 5) % 8 = opposite of arrival + 1
        let search_start = (dir + 5) % 8;
        let mut next_dir = None;
        for i in 0..8 {
            let d = (search_start + i) % 8;
            let nx = cx + dx[d];
            let ny = cy + dy[d];
            let ni = ny as usize * pw + nx as usize;
            if img[ni] != 0 {
                next_dir = Some(d);
                break;
            }
        }

        let Some(nd) = next_dir else {
            break; // shouldn't happen for a valid contour
        };

        let nx = cx + dx[nd];
        let ny = cy + dy[nd];
        dir = nd;

        // Termination: we've returned to start and the next step is the second point
        if nx == start_x && ny == start_y && cx == second_x && cy == second_y {
            break;
        }
        // Also terminate if we've returned to start
        if nx == start_x && ny == start_y {
            break;
        }

        cx = nx;
        cy = ny;

        // Safety: prevent infinite loops
        if contour.len() > (pw * img.len() / pw) {
            break;
        }
    }

    contour
}

/// Compute triangle threshold for a grayscale image.
///
/// Draws a line from the histogram peak to the farthest end,
/// then finds the bin with maximum perpendicular distance to that line.
///
/// Reference: OpenCV cv2.threshold(..., THRESH_TRIANGLE).
pub fn triangle_threshold(pixels: &[u8], info: &ImageInfo) -> Result<u8, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "triangle requires Gray8 input".into(),
        ));
    }

    let mut hist = [0u32; 256];
    for &v in pixels {
        hist[v as usize] += 1;
    }

    // Find histogram bounds and peak
    let mut left = 0usize;
    let mut right = 255usize;
    while left < 256 && hist[left] == 0 {
        left += 1;
    }
    while right > 0 && hist[right] == 0 {
        right -= 1;
    }
    if left >= right {
        return Ok(left as u8);
    }

    let mut peak = left;
    for i in left..=right {
        if hist[i] > hist[peak] {
            peak = i;
        }
    }

    // Determine which side is longer — the line goes from peak to the far end
    let flip = (peak - left) < (right - peak);
    let (a, b) = if flip { (peak, right) } else { (left, peak) };

    // Line from (a, hist[a]) to (b, hist[b])
    let ax = a as f64;
    let ay = hist[a] as f64;
    let bx = b as f64;
    let by = hist[b] as f64;

    // Find bin with max perpendicular distance to the line
    let line_len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    if line_len < 1e-10 {
        return Ok(peak as u8);
    }

    let mut best_dist = 0.0f64;
    let mut best_t = a;
    let range = if a < b { a..=b } else { b..=a };
    for t in range {
        let px = t as f64;
        let py = hist[t] as f64;
        // Perpendicular distance from point to line
        let dist = ((by - ay) * px - (bx - ax) * py + bx * ay - by * ax).abs() / line_len;
        if dist > best_dist {
            best_dist = dist;
            best_t = t;
        }
    }

    Ok(best_t as u8)
}

/// Upsample by 2x using bilinear interpolation.
pub fn upsample_2x(data: &[f32], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tw * th];
    for y in 0..th {
        for x in 0..tw {
            let sx = x as f32 / tw as f32 * sw as f32;
            let sy = y as f32 / th as f32 * sh as f32;
            let x0 = (sx as usize).min(sw - 1);
            let y0 = (sy as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            out[y * tw + x] = data[y0 * sw + x0] * (1.0 - fx) * (1.0 - fy)
                + data[y0 * sw + x1] * fx * (1.0 - fy)
                + data[y1 * sw + x0] * (1.0 - fx) * fy
                + data[y1 * sw + x1] * fx * fy;
        }
    }
    out
}

/// Convenience wrapper for non-tiled Gaussian vignette.
pub fn vignette_full(
    pixels: &[u8],
    info: &ImageInfo,
    sigma: f32,
    x_inset: u32,
    y_inset: u32,
) -> Result<Vec<u8>, ImageError> {
    let r = Rect::new(0, 0, info.width, info.height);
    let mut u = |_: Rect| Ok(pixels.to_vec());
    vignette(
        r,
        &mut u,
        info,
        &VignetteParams {
            sigma,
            x_inset,
            y_inset,
            full_width: info.width,
            full_height: info.height,
            tile_offset_x: 0,
            tile_offset_y: 0,
        },
    )
}

pub fn weighted_median_val(
    sorted: &[(f32, f32, f32)],
    total_weight: f32,
    val: impl Fn(&(f32, f32, f32)) -> f32,
) -> f32 {
    let half = total_weight / 2.0;
    let mut accum = 0.0f32;
    for item in sorted {
        accum += item.2;
        if accum >= half {
            return val(item);
        }
    }
    val(sorted.last().unwrap())
}

/// xorshift64 PRNG — fast, deterministic, good enough for noise generation.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Return a uniform f64 in [0, 1) from the PRNG.
#[inline]
pub fn xorshift64_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// 1D IIR gaussian blur (forward + backward with edge-replicated padding).
///
/// Uses f64 intermediates for precision (matching GEGL's gdouble).
/// Pads the buffer with replicated edge values to handle boundaries
/// correctly without the complex matrix correction.
pub fn yvv_blur_1d(buf: &mut [f32], b: &[f64; 4], _m: &[[f64; 3]; 3]) {
    let n = buf.len();
    if n < 4 {
        return;
    }

    // Pad with 3*sigma samples on each side (replicated edge) to let the
    // IIR settle. For the coefficients we use, 3 samples of warm-up on
    // each side is the theoretical minimum, but more padding gives better
    // boundary behavior. Use max(ceil(3*sigma), 32) padding, capped at n.
    // Since we don't have sigma here, use a generous fixed padding.
    let pad = n.min(64);

    let total = pad + n + pad;
    let mut tmp = vec![0.0f64; total];

    // Fill: left pad + data + right pad
    let left_val = buf[0] as f64;
    let right_val = buf[n - 1] as f64;
    for val in tmp.iter_mut().take(pad) {
        *val = left_val;
    }
    for (i, val) in tmp.iter_mut().skip(pad).take(n).enumerate() {
        *val = buf[i] as f64;
    }
    for val in tmp.iter_mut().skip(pad + n).take(pad) {
        *val = right_val;
    }

    // Forward (causal) pass
    let mut y1 = tmp[0];
    let mut y2 = tmp[0];
    let mut y3 = tmp[0];
    for val in tmp.iter_mut() {
        let y = b[0] * *val + b[1] * y1 + b[2] * y2 + b[3] * y3;
        *val = y;
        y3 = y2;
        y2 = y1;
        y1 = y;
    }

    // Backward (anti-causal) pass
    y1 = tmp[total - 1];
    y2 = tmp[total - 1];
    y3 = tmp[total - 1];
    for val in tmp.iter_mut().rev() {
        let y = b[0] * *val + b[1] * y1 + b[2] * y2 + b[3] * y3;
        *val = y;
        y3 = y2;
        y2 = y1;
        y1 = y;
    }

    // Extract the valid region
    for i in 0..n {
        buf[i] = tmp[pad + i] as f32;
    }
}

/// Compute Young/van Vliet IIR coefficients and boundary correction matrix.
/// Exact port of GEGL's `iir_young_find_constants`.
///
/// Returns (b[4], m[3][3]) where b[0] is scale, b[1-3] are recursive coefficients,
/// and m is the right-boundary correction matrix.
pub fn yvv_find_constants(sigma: f32) -> ([f64; 4], [[f64; 3]; 3]) {
    let sigma = sigma as f64;
    let k1 = 2.44413;
    let k2 = 1.4281;
    let k3 = 0.422205;

    let q = if sigma >= 2.5 {
        0.98711 * sigma - 0.96330
    } else {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).sqrt()
    };

    let b0 = 1.57825 + q * (k1 + q * (k2 + q * k3));
    let b1_raw = q * (k1 + q * (2.0 * k2 + q * 3.0 * k3));
    let b2_raw = -k2 * q * q - k3 * 3.0 * q * q * q;
    let b3_raw = q * q * q * k3;

    let a1 = b1_raw / b0;
    let a2 = b2_raw / b0;
    let a3 = b3_raw / b0;

    let b = [1.0 - (a1 + a2 + a3), a1, a2, a3];

    // Right-boundary correction matrix (GEGL's fix_right_boundary)
    let c = 1.0 / ((1.0 + a1 - a2 + a3) * (1.0 + a2 + (a1 - a3) * a3));

    let m = [
        [
            c * (-a3 * (a1 + a3) - a2 + 1.0),
            c * (a3 + a1) * (a2 + a3 * a1),
            c * a3 * (a1 + a3 * a2),
        ],
        [
            c * (a1 + a3 * a2),
            c * (1.0 - a2) * (a2 + a3 * a1),
            c * a3 * (1.0 - a3 * a1 - a3 * a3 - a2),
        ],
        [
            c * (a3 * a1 + a2 + a1 * a1 - a2 * a2),
            c * (a1 * a2 + a3 * a2 * a2 - a1 * a3 * a3 - a3 * a3 * a3 - a3 * a2 + a3),
            c * a3 * (a1 + a3 * a2),
        ],
    ];

    (b, m)
}

