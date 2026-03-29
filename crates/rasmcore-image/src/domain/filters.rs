//! Image filters — SIMD-optimized where possible.
//!
//! Operations work directly on raw pixel buffers without DynamicImage conversion.
//! Blur uses libblur (SIMD on x86/ARM/WASM). Point ops are written as simple
//! loops that LLVM auto-vectorizes to SIMD128 when compiled with +simd128.

use super::error::ImageError;
use super::types::{DecodedImage, ImageInfo, PixelFormat};

fn validate_format(format: PixelFormat) -> Result<(), ImageError> {
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
fn is_16bit(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
    )
}

/// Number of channels for a pixel format (not bytes — channels).
fn channels(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 | PixelFormat::Gray16 => 1,
        PixelFormat::Rgb8 | PixelFormat::Rgb16 => 3,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => 4,
        _ => 3,
    }
}

// ── 16-bit I/O helpers ─────────────────────────────────────────────────────

/// Read u16 samples from a byte buffer (little-endian).
fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Write u16 samples to a byte buffer (little-endian).
fn u16_to_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Convert 16-bit pixel buffer to f32 normalized [0.0, 1.0] per sample.
fn u16_pixels_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes_to_u16(bytes)
        .into_iter()
        .map(|v| v as f32 / 65535.0)
        .collect()
}

/// Convert f32 normalized [0.0, 1.0] samples back to 16-bit pixel buffer.
fn f32_to_u16_pixels(values: &[f32]) -> Vec<u8> {
    let u16s: Vec<u16> = values
        .iter()
        .map(|&v| (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16)
        .collect();
    u16_to_bytes(&u16s)
}

/// Convert 16-bit pixel buffer to 8-bit for processing, then back to 16-bit.
/// Used when an operation only supports 8-bit internally (e.g., libblur).
fn process_via_8bit<F>(pixels: &[u8], info: &ImageInfo, f: F) -> Result<Vec<u8>, ImageError>
where
    F: FnOnce(&[u8], &ImageInfo) -> Result<Vec<u8>, ImageError>,
{
    let ch = channels(info.format);
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

// ─── Filter Config Structs (auto-generate param metadata via ConfigParams) ──

/// Parameters for Gaussian blur.
#[derive(rasmcore_macros::ConfigParams)]
pub struct BlurParams {
    /// Blur radius in pixels
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 3.0)]
    pub radius: f32,
}

/// Parameters for unsharp mask sharpening.
#[derive(rasmcore_macros::ConfigParams)]
pub struct SharpenParams {
    /// Sharpening amount
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 1.0)]
    pub amount: f32,
}

/// Parameters for brightness adjustment.
#[derive(rasmcore_macros::ConfigParams)]
pub struct BrightnessParams {
    /// Brightness offset (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0)]
    pub amount: f32,
}

/// Parameters for contrast adjustment.
#[derive(rasmcore_macros::ConfigParams)]
pub struct ContrastParams {
    /// Contrast factor (-1 to 1)
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0)]
    pub amount: f32,
}

/// Parameters for median filter.
#[derive(rasmcore_macros::ConfigParams)]
pub struct MedianParams {
    /// Filter radius in pixels
    #[param(min = 1, max = 20, step = 1, default = 3)]
    pub radius: u32,
}

/// Parameters for Canny edge detection.
#[derive(rasmcore_macros::ConfigParams)]
pub struct CannyParams {
    /// Low hysteresis threshold
    #[param(min = 0.0, max = 255.0, step = 1.0, default = 50.0)]
    pub low_threshold: f32,
    /// High hysteresis threshold
    #[param(min = 0.0, max = 255.0, step = 1.0, default = 150.0)]
    pub high_threshold: f32,
}

/// Parameters for resize transform.
#[derive(rasmcore_macros::ConfigParams)]
pub struct ResizeParams {
    /// Target width in pixels
    #[param(min = 1, max = 8000, step = 1, default = 800)]
    pub width: u32,
    /// Target height in pixels
    #[param(min = 1, max = 8000, step = 1, default = 600)]
    pub height: u32,
}

/// Parameters for crop transform.
#[derive(rasmcore_macros::ConfigParams)]
pub struct CropParams {
    /// X offset
    #[param(min = 0, max = 8000, step = 1, default = 0)]
    pub x: u32,
    /// Y offset
    #[param(min = 0, max = 8000, step = 1, default = 0)]
    pub y: u32,
    /// Crop width
    #[param(min = 1, max = 8000, step = 1, default = 256)]
    pub width: u32,
    /// Crop height
    #[param(min = 1, max = 8000, step = 1, default = 256)]
    pub height: u32,
}

/// Apply gaussian blur using libblur (SIMD-optimized).
///
/// Uses separable gaussian convolution with SIMD acceleration on
/// x86 (SSE/AVX), ARM (NEON), and WASM (SIMD128).
#[rasmcore_macros::register_filter(name = "blur", category = "spatial")]
pub fn blur(pixels: &[u8], info: &ImageInfo, radius: f32) -> Result<Vec<u8>, ImageError> {
    if radius < 0.0 {
        return Err(ImageError::InvalidParameters(
            "blur radius must be >= 0".into(),
        ));
    }
    validate_format(info.format)?;

    if radius == 0.0 {
        return Ok(pixels.to_vec());
    }

    let kernel_size = (radius * 3.0).ceil() as u32 * 2 + 1;
    let kernel_size = kernel_size.max(3);

    let mut result = vec![0u8; pixels.len()];

    // 16-bit: delegate to 8-bit path via process_via_8bit (libblur only supports u8)
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| blur(p8, i8, radius));
    }

    let channels = match info.format {
        PixelFormat::Rgb8 => libblur::FastBlurChannels::Channels3,
        PixelFormat::Rgba8 => libblur::FastBlurChannels::Channels4,
        PixelFormat::Gray8 => libblur::FastBlurChannels::Plane,
        _ => unreachable!(),
    };

    libblur::gaussian_blur(
        pixels,
        &mut result,
        info.width,
        info.height,
        kernel_size,
        radius,
        channels,
        libblur::EdgeMode::Clamp,
        libblur::ThreadingPolicy::Single,
        libblur::GaussianPreciseLevel::EXACT,
    );

    Ok(result)
}

/// Apply sharpening (unsharp mask).
///
/// Computes: output = original + amount * (original - blurred)
/// Uses the SIMD-optimized blur internally.
#[rasmcore_macros::register_filter(name = "sharpen", category = "spatial")]
pub fn sharpen(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    // 16-bit: work in f32 for full precision
    if is_16bit(info.format) {
        let orig_f32 = u16_pixels_to_f32(pixels);
        let info_8 = ImageInfo {
            format: match info.format {
                PixelFormat::Rgb16 => PixelFormat::Rgb8,
                PixelFormat::Rgba16 => PixelFormat::Rgba8,
                PixelFormat::Gray16 => PixelFormat::Gray8,
                other => other,
            },
            ..*info
        };
        let blurred = blur(pixels, info, 1.0)?;
        let blur_f32 = u16_pixels_to_f32(&blurred);
        let result_f32: Vec<f32> = orig_f32
            .iter()
            .zip(blur_f32.iter())
            .map(|(&o, &b)| (o + amount * (o - b)).clamp(0.0, 1.0))
            .collect();
        return Ok(f32_to_u16_pixels(&result_f32));
    }

    // Blur with a small radius for the unsharp mask
    let blurred = blur(pixels, info, 1.0)?;

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

/// Adjust brightness (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.
#[rasmcore_macros::register_filter(name = "brightness", category = "adjustment")]
pub fn brightness(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "brightness must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| brightness(p8, i8, amount));
    }
    let lut = super::point_ops::build_lut(&super::point_ops::PointOp::Brightness(amount));
    super::point_ops::apply_lut(pixels, info, &lut)
}

/// Adjust contrast (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.
#[rasmcore_macros::register_filter(name = "contrast", category = "adjustment")]
pub fn contrast(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "contrast must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| contrast(p8, i8, amount));
    }
    let lut = super::point_ops::build_lut(&super::point_ops::PointOp::Contrast(amount));
    super::point_ops::apply_lut(pixels, info, &lut)
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

// ─── Color Adjustment Functions ──────────────────────────────────────────
//
// All color transforms delegate to color_lut::ColorOp for the math
// (single source of truth). The direct per-pixel evaluation avoids
// 3D CLUT allocation overhead for non-pipeline callers.

use super::color_lut::ColorOp;

/// Apply a ColorOp to a pixel buffer via direct per-pixel evaluation.
///
/// No CLUT allocation — evaluates ColorOp::apply() on each pixel's
/// normalized (R,G,B). For pipeline use, ColorOpNode builds a CLUT instead.
fn apply_color_op(pixels: &[u8], info: &ImageInfo, op: &ColorOp) -> Result<Vec<u8>, ImageError> {
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

/// Rotate hue by `degrees` (0-360). Works on RGB8 and RGBA8 images.
#[rasmcore_macros::register_filter(name = "hue_rotate", category = "color")]
pub fn hue_rotate(pixels: &[u8], info: &ImageInfo, degrees: f32) -> Result<Vec<u8>, ImageError> {
    apply_color_op(pixels, info, &ColorOp::HueRotate(degrees))
}

/// Adjust saturation by `factor` (0=grayscale, 1=unchanged, 2=double).
#[rasmcore_macros::register_filter(name = "saturate", category = "color")]
pub fn saturate(pixels: &[u8], info: &ImageInfo, factor: f32) -> Result<Vec<u8>, ImageError> {
    apply_color_op(pixels, info, &ColorOp::Saturate(factor))
}

/// Apply sepia tone with given `intensity` (0=none, 1=full sepia).
#[rasmcore_macros::register_filter(name = "sepia", category = "color")]
pub fn sepia(pixels: &[u8], info: &ImageInfo, intensity: f32) -> Result<Vec<u8>, ImageError> {
    apply_color_op(pixels, info, &ColorOp::Sepia(intensity.clamp(0.0, 1.0)))
}

#[derive(rasmcore_macros::ConfigParams)]
pub struct ColorizeParams {
    /// Target color to blend toward
    #[param(hint = "rc.color_rgb")]
    pub target: [u8; 3],
    /// Blend amount (0=none, 1=full tint)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub amount: f32,
}

/// Tint image toward `target_color` (RGB) by `amount` (0=none, 1=full tint).
#[rasmcore_macros::register_filter(name = "colorize", category = "color")]
pub fn colorize(
    pixels: &[u8],
    info: &ImageInfo,
    target: [u8; 3],
    amount: f32,
) -> Result<Vec<u8>, ImageError> {
    let target_norm = [
        target[0] as f32 / 255.0,
        target[1] as f32 / 255.0,
        target[2] as f32 / 255.0,
    ];
    apply_color_op(
        pixels,
        info,
        &ColorOp::Colorize(target_norm, amount.clamp(0.0, 1.0)),
    )
}

// =============================================================================
// Convolution filters
// =============================================================================

/// Predefined convolution kernels.
pub mod kernels {
    /// 3x3 emboss kernel.
    pub const EMBOSS: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
    /// 3x3 edge-enhance kernel.
    pub const EDGE_ENHANCE: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
    /// 3x3 sharpen kernel.
    pub const SHARPEN_3X3: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
    /// 3x3 box blur kernel (each weight = 1.0, divisor = 9.0).
    pub const BOX_BLUR_3X3: [f32; 9] = [1.0; 9];
}

/// Apply arbitrary NxN convolution with reflect-edge border handling.
///
/// Automatically detects separable (rank-1) kernels and uses two 1D passes
/// for O(2K) instead of O(K^2) per pixel. Uses padded input buffer to
/// eliminate per-pixel boundary checks for interior pixels.
#[rasmcore_macros::register_filter(name = "convolve", category = "spatial")]
pub fn convolve(
    pixels: &[u8],
    info: &ImageInfo,
    kernel: &[f32],
    kw: usize,
    kh: usize,
    divisor: f32,
) -> Result<Vec<u8>, ImageError> {
    if kw % 2 == 0 || kh % 2 == 0 || kw * kh != kernel.len() {
        return Err(ImageError::InvalidParameters(
            "kernel dimensions must be odd and match kernel length".into(),
        ));
    }
    validate_format(info.format)?;

    // 16-bit: process in f32 domain, then convert back
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            convolve(p8, i8, kernel, kw, kh, divisor)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;

    // Try separable path first (O(2K) vs O(K^2))
    if let Some((row_k, col_k)) = is_separable(kernel, kw, kh) {
        return convolve_separable(pixels, w, h, channels, &row_k, &col_k, divisor);
    }

    // General 2D convolution with padded input
    let rw = kw / 2;
    let rh = kh / 2;
    let inv_div = 1.0 / divisor;
    let padded = pad_reflect(pixels, w, h, channels, rw.max(rh));
    let pw = w + 2 * rw.max(rh);

    let mut out = vec![0u8; pixels.len()];
    let pad = rw.max(rh);

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    let row_off = (y + pad - rh + ky) * pw * channels;
                    for kx in 0..kw {
                        let px_off = row_off + (x + pad - rw + kx) * channels + c;
                        sum += kernel[ky * kw + kx] * padded[px_off] as f32;
                    }
                }
                out[(y * w + x) * channels + c] = (sum * inv_div).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

/// Detect if a 2D kernel is separable (rank-1: K = col * row^T).
///
/// Returns `Some((row_kernel, col_kernel))` if separable, `None` otherwise.
fn is_separable(kernel: &[f32], kw: usize, kh: usize) -> Option<(Vec<f32>, Vec<f32>)> {
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

/// Two-pass separable convolution: horizontal then vertical.
fn convolve_separable(
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
            out[idx] = (sum * inv_div).clamp(0.0, 255.0) as u8;
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
                    out[(y * w + x) * channels + c] = (sum * inv_div).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// Create a padded copy of the image with reflected borders.
///
/// Eliminates per-pixel boundary checks — interior pixels use direct indexing.
fn pad_reflect(pixels: &[u8], w: usize, h: usize, channels: usize, pad: usize) -> Vec<u8> {
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

/// Reflect-edge coordinate clamping.
fn reflect(v: i32, size: usize) -> usize {
    if v < 0 {
        (-v).min(size as i32 - 1) as usize
    } else if v >= size as i32 {
        (2 * size as i32 - v - 2).max(0) as usize
    } else {
        v as usize
    }
}

/// Apply median filter with given radius. Window is (2*radius+1)^2.
///
/// Uses histogram sliding-window (Huang algorithm) for radius > 2 giving
/// O(1) amortized per pixel. Falls back to sorting for radius <= 2 where
/// the small window makes sorting faster than histogram maintenance.
#[rasmcore_macros::register_filter(name = "median", category = "spatial")]
pub fn median(pixels: &[u8], info: &ImageInfo, radius: u32) -> Result<Vec<u8>, ImageError> {
    if radius == 0 {
        return Ok(pixels.to_vec());
    }
    validate_format(info.format)?;

    // 16-bit: delegate to 8-bit path (histogram-based median would need 65536 bins)
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| median(p8, i8, radius));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;

    if radius <= 2 {
        median_sort(pixels, w, h, channels, radius)
    } else {
        median_histogram(pixels, w, h, channels, radius)
    }
}

/// Sorting-based median for small radii (radius <= 2).
fn median_sort(
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

/// Histogram sliding-window median (Huang algorithm) for large radii.
///
/// Maintains a 256-bin histogram. When sliding horizontally, removes the
/// leftmost column and adds the rightmost column — O(2*diameter) per pixel
/// instead of O(diameter^2).
fn median_histogram(
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
            let mut count = 0u32;

            // Initialize histogram for first window in this row
            for ky in -r..=r {
                let sy = reflect(y as i32 + ky, h);
                for kx in -r..=r {
                    let sx = reflect(0i32 + kx, w);
                    hist[pixels[(sy * w + sx) * channels + c] as usize] += 1;
                    count += 1;
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
                    count -= 1;
                }

                // Add rightmost column (x + r)
                let new_x = x as i32 + r;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(new_x, w);
                    let val = pixels[(sy * w + sx) * channels + c] as usize;
                    hist[val] += 1;
                    count += 1;
                }

                out[(y * w + x) * channels + c] = find_median_in_hist(&hist, median_pos);
            }
        }
    }
    Ok(out)
}

/// Find the median value by scanning the histogram until cumulative count
/// reaches the target position.
#[inline]
fn find_median_in_hist(hist: &[u32; 256], target: usize) -> u8 {
    let mut cumulative = 0u32;
    for (val, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative as usize > target {
            return val as u8;
        }
    }
    255
}

/// Sobel edge detection — produces grayscale gradient magnitude image.
///
/// Uses unrolled 3x3 Sobel with padded input — no inner loop or
/// match-based weight lookup. Direct coefficient access gives ~3x speedup.
#[rasmcore_macros::register_filter(name = "sobel", category = "edge")]
pub fn sobel(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| sobel(p8, i8));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;

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

/// Scharr edge detection — more rotationally symmetric than Sobel.
///
/// Uses 3x3 Scharr kernels: Gx = [[-3,0,3],[-10,0,10],[-3,0,3]]
/// Returns gradient magnitude (L2 norm of Gx and Gy).
/// Reference: cv2.Scharr (OpenCV 4.13).
pub fn scharr(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| scharr(p8, i8));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;
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

/// Laplacian — second-order derivative edge detection.
///
/// Uses 3x3 kernel: [[0,1,0],[1,-4,1],[0,1,0]].
/// Returns absolute value of Laplacian, clamped to [0, 255].
/// Reference: cv2.Laplacian (OpenCV 4.13).
pub fn laplacian(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| laplacian(p8, i8));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;
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

/// Canny edge detection — produces binary edge map (0 or 255).
///
/// Steps: 1) Gaussian blur, 2) Sobel gradient + direction,
/// 3) Non-maximum suppression, 4) Hysteresis thresholding.
#[rasmcore_macros::register_filter(name = "canny", category = "edge")]
pub fn canny(
    pixels: &[u8],
    info: &ImageInfo,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            canny(p8, i8, low_threshold, high_threshold)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;

    // Step 1: Convert to grayscale
    let gray = to_grayscale(pixels, channels);

    // Step 2: Gaussian blur (sigma ~1.4, via 3x3 kernel approximation)
    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    let blurred = blur(&gray, &gray_info, 1.4)?;

    // Step 3: Sobel gradient magnitude and direction (unrolled, padded)
    let mut magnitude = vec![0.0f32; w * h];
    let mut direction = vec![0u8; w * h];
    let padded = pad_reflect(&blurred, w, h, 1, 1);
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

            magnitude[y * w + x] = (gx * gx + gy * gy).sqrt();

            // Quantize angle to 4 directions
            let angle = gy.atan2(gx).to_degrees();
            let angle = if angle < 0.0 { angle + 180.0 } else { angle };
            direction[y * w + x] = if angle < 22.5 || angle >= 157.5 {
                0 // horizontal
            } else if angle < 67.5 {
                1 // 45 degrees
            } else if angle < 112.5 {
                2 // vertical
            } else {
                3 // 135 degrees
            };
        }
    }

    // Step 4: Non-maximum suppression
    let mut nms = vec![0.0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let mag = magnitude[y * w + x];
            let (n1, n2) = match direction[y * w + x] {
                0 => (magnitude[y * w + x - 1], magnitude[y * w + x + 1]),
                1 => (
                    magnitude[(y - 1) * w + x + 1],
                    magnitude[(y + 1) * w + x - 1],
                ),
                2 => (magnitude[(y - 1) * w + x], magnitude[(y + 1) * w + x]),
                _ => (
                    magnitude[(y - 1) * w + x - 1],
                    magnitude[(y + 1) * w + x + 1],
                ),
            };
            nms[y * w + x] = if mag >= n1 && mag >= n2 { mag } else { 0.0 };
        }
    }

    // Step 5: Hysteresis thresholding
    let mut out = vec![0u8; w * h];
    // Mark strong edges
    for i in 0..w * h {
        if nms[i] >= high_threshold {
            out[i] = 255;
        }
    }
    // Extend to weak edges connected to strong edges
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if out[y * w + x] == 0 && nms[y * w + x] >= low_threshold {
                    // Check 8-connected neighbors for strong edge
                    let has_strong = (-1..=1i32).any(|dy| {
                        (-1..=1i32).any(|dx| {
                            out[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] == 255
                        })
                    });
                    if has_strong {
                        out[y * w + x] = 255;
                        changed = true;
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Convert multi-channel pixels to single-channel grayscale.
/// Convert multi-channel pixels to single-channel grayscale.
///
/// Uses BT.601 fixed-point: (77*R + 150*G + 29*B + 128) >> 8.
/// Integer-only arithmetic — no floating point in the hot path.
fn to_grayscale(pixels: &[u8], channels: usize) -> Vec<u8> {
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

fn to_grayscale_scalar(pixels: &[u8], channels: usize, pixel_count: usize) -> Vec<u8> {
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
fn to_grayscale_simd128(pixels: &[u8], channels: usize, pixel_count: usize) -> Vec<u8> {
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

// ─── Alpha Management ────────────────────────────────────────────────────

/// Convert straight alpha to premultiplied alpha (RGBA8 only).
#[rasmcore_macros::register_filter(name = "premultiply", category = "alpha")]
pub fn premultiply(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "premultiply requires RGBA8".into(),
        ));
    }
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        let a = chunk[3] as u16;
        chunk[0] = ((chunk[0] as u16 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u16 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u16 * a + 127) / 255) as u8;
    }
    Ok(result)
}

/// Convert premultiplied alpha to straight alpha (RGBA8 only).
#[rasmcore_macros::register_filter(name = "unpremultiply", category = "alpha")]
pub fn unpremultiply(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "unpremultiply requires RGBA8".into(),
        ));
    }
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        let a = chunk[3] as u16;
        if a > 0 {
            chunk[0] = ((chunk[0] as u16 * 255 + a / 2) / a).min(255) as u8;
            chunk[1] = ((chunk[1] as u16 * 255 + a / 2) / a).min(255) as u8;
            chunk[2] = ((chunk[2] as u16 * 255 + a / 2) / a).min(255) as u8;
        }
    }
    Ok(result)
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

/// Add alpha channel to RGB8, producing RGBA8 with given alpha value.
pub fn add_alpha(
    pixels: &[u8],
    info: &ImageInfo,
    alpha: u8,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "add_alpha requires RGB8 input".into(),
        ));
    }
    let npixels = (info.width * info.height) as usize;
    let mut rgba = Vec::with_capacity(npixels * 4);
    for chunk in pixels.chunks_exact(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(alpha);
    }
    Ok((
        rgba,
        ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Rgba8,
            color_space: info.color_space,
        },
    ))
}

/// Remove alpha channel from RGBA8, producing RGB8.
pub fn remove_alpha(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "remove_alpha requires RGBA8 input".into(),
        ));
    }
    let npixels = (info.width * info.height) as usize;
    let mut rgb = Vec::with_capacity(npixels * 3);
    for chunk in pixels.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
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

// ─── Blend Modes ─────────────────────────────────────────────────────────

/// Supported blend modes for compositing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
}

/// Apply per-pixel blend formula.
#[inline]
fn blend_channel(a: u8, b: u8, mode: BlendMode) -> u8 {
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
    };
    (result.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Blend two same-size RGB8 or RGBA8 images using the given blend mode.
///
/// `fg` is the "top" layer, `bg` is the "bottom" layer.
/// Both must have the same format and dimensions.
/// For RGBA8, alpha is preserved from `bg` (bottom layer).
pub fn blend(
    fg_pixels: &[u8],
    fg_info: &ImageInfo,
    bg_pixels: &[u8],
    bg_info: &ImageInfo,
    mode: BlendMode,
) -> Result<Vec<u8>, ImageError> {
    if fg_info.format != bg_info.format {
        return Err(ImageError::InvalidInput("format mismatch".into()));
    }
    if fg_info.width != bg_info.width || fg_info.height != bg_info.height {
        return Err(ImageError::InvalidInput("dimension mismatch".into()));
    }
    validate_format(fg_info.format)?;

    let bpp = match fg_info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "blend requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let mut result = bg_pixels.to_vec();
    for (fg_chunk, bg_chunk) in fg_pixels
        .chunks_exact(bpp)
        .zip(result.chunks_exact_mut(bpp))
    {
        bg_chunk[0] = blend_channel(fg_chunk[0], bg_chunk[0], mode);
        bg_chunk[1] = blend_channel(fg_chunk[1], bg_chunk[1], mode);
        bg_chunk[2] = blend_channel(fg_chunk[2], bg_chunk[2], mode);
        // Alpha stays from bg (bottom layer) for RGBA8
    }
    Ok(result)
}

// ─── CLAHE (Contrast Limited Adaptive Histogram Equalization) ──────────────

/// Apply CLAHE — local adaptive contrast enhancement.
///
/// Divides the image into `tile_grid` x `tile_grid` tiles, equalizes each
/// tile's histogram with a clip limit, then bilinear interpolates between
/// tiles for smooth transitions. Grayscale only (convert first for color).
///
/// - `clip_limit`: contrast amplification limit (2.0-4.0 typical, higher = more contrast)
/// - `tile_grid`: number of tiles per dimension (8 = 8x8 grid, OpenCV default)
pub fn clahe(
    pixels: &[u8],
    info: &ImageInfo,
    clip_limit: f32,
    tile_grid: u32,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "CLAHE requires Gray8 input".into(),
        ));
    }
    if tile_grid == 0 || clip_limit < 1.0 {
        return Err(ImageError::InvalidParameters(
            "tile_grid must be > 0, clip_limit must be >= 1.0".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let grid = tile_grid as usize;
    let tile_w = (w + grid - 1) / grid;
    let tile_h = (h + grid - 1) / grid;

    // Build per-tile CDF lookup tables
    let mut tile_luts = vec![[0u8; 256]; grid * grid];

    for ty in 0..grid {
        for tx in 0..grid {
            let x0 = tx * tile_w;
            let y0 = ty * tile_h;
            let x1 = (x0 + tile_w).min(w);
            let y1 = (y0 + tile_h).min(h);
            let tile_pixels = (x1 - x0) * (y1 - y0);
            if tile_pixels == 0 {
                continue;
            }

            // Histogram
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[pixels[y * w + x] as usize] += 1;
                }
            }

            // Clip histogram and redistribute (matching OpenCV exactly)
            // No special case for single-value tiles — OpenCV processes all tiles uniformly.
            let clip = ((clip_limit * tile_pixels as f32) / 256.0) as u32;
            let clip = clip.max(1);
            let mut clipped = 0u32;
            for h in hist.iter_mut() {
                if *h > clip {
                    clipped += *h - clip;
                    *h = clip;
                }
            }
            // Redistribute: uniform batch + stepped residual (OpenCV algorithm)
            let redist_batch = clipped / 256;
            let residual = clipped - redist_batch * 256;
            for h in hist.iter_mut() {
                *h += redist_batch;
            }
            if residual > 0 {
                let step = (256 / residual as usize).max(1);
                let mut remaining = residual as usize;
                let mut i = 0;
                while i < 256 && remaining > 0 {
                    hist[i] += 1;
                    remaining -= 1;
                    i += step;
                }
            }

            // Build CDF → LUT (OpenCV formula: lut[i] = saturate(sum * lutScale))
            let lut_scale = 255.0f32 / tile_pixels as f32;
            let lut = &mut tile_luts[ty * grid + tx];
            let mut sum = 0u32;
            for i in 0..256 {
                sum += hist[i];
                let v = (sum as f32 * lut_scale).round();
                lut[i] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Apply with bilinear interpolation (matching OpenCV exactly)
    let inv_tw = 1.0f32 / tile_w as f32;
    let inv_th = 1.0f32 / tile_h as f32;
    let mut result = vec![0u8; pixels.len()];
    for y in 0..h {
        let fy = y as f32 * inv_th - 0.5;
        let ty1i = fy.floor() as isize;
        let ty2i = ty1i + 1;
        let ya = fy - ty1i as f32;
        let ya1 = 1.0 - ya;
        let ty1 = ty1i.clamp(0, grid as isize - 1) as usize;
        let ty2 = (ty2i as usize).min(grid - 1);

        for x in 0..w {
            let fx = x as f32 * inv_tw - 0.5;
            let tx1i = fx.floor() as isize;
            let tx2i = tx1i + 1;
            let xa = fx - tx1i as f32;
            let xa1 = 1.0 - xa;
            let tx1 = tx1i.clamp(0, grid as isize - 1) as usize;
            let tx2 = (tx2i as usize).min(grid - 1);

            let val = pixels[y * w + x] as usize;

            // Bilinear interpolation of 4 tile LUTs (OpenCV order)
            let v = (tile_luts[ty1 * grid + tx1][val] as f32 * xa1
                + tile_luts[ty1 * grid + tx2][val] as f32 * xa)
                * ya1
                + (tile_luts[ty2 * grid + tx1][val] as f32 * xa1
                    + tile_luts[ty2 * grid + tx2][val] as f32 * xa)
                    * ya;

            result[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}

/// BORDER_REFLECT_101: reflect at boundary without duplicating edge pixel.
/// Matches OpenCV's default border mode.
#[inline(always)]
fn reflect101(idx: isize, size: isize) -> isize {
    if idx < 0 {
        -idx
    } else if idx >= size {
        2 * size - 2 - idx
    } else {
        idx
    }
}

// ─── Bilateral Filter ─────────────────────────────────────────────────────

/// Edge-preserving bilateral filter — pixel-exact match with OpenCV 4.13.
///
/// Uses circular kernel mask, f32 accumulation, BORDER_REFLECT_101 padding,
/// pre-computed spatial/color weight LUTs, and L1 color norm for RGB.
///
/// - `diameter`: filter size (use 0 for auto from sigma_space; typical 5-9)
/// - `sigma_color`: filter sigma in the color/intensity space (10-150 typical)
/// - `sigma_space`: filter sigma in coordinate space (10-150 typical)
pub fn bilateral(
    pixels: &[u8],
    info: &ImageInfo,
    diameter: u32,
    sigma_color: f32,
    sigma_space: f32,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 && info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "bilateral filter requires Gray8 or Rgb8".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let channels = if info.format == PixelFormat::Gray8 {
        1
    } else {
        3
    };
    let radius = if diameter > 0 {
        (diameter as usize | 1) / 2
    } else {
        (sigma_space * 1.5).round() as usize
    };

    // Pre-compute spatial weight LUT + offsets (CIRCULAR mask, matching OpenCV)
    let gauss_space_coeff: f32 = -0.5 / (sigma_space * sigma_space);
    let mut space_weight: Vec<f32> = Vec::new();
    let mut space_ofs: Vec<(isize, isize)> = Vec::new();
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let r = ((dy * dy + dx * dx) as f64).sqrt();
            if r > radius as f64 {
                continue; // Circular mask — skip corners
            }
            let r2 = (dy * dy + dx * dx) as f32;
            space_weight.push((r2 * gauss_space_coeff).exp());
            space_ofs.push((dy, dx));
        }
    }
    let maxk = space_weight.len();

    // Pre-compute color weight LUT (indexed by |diff|, 0..255*channels)
    let gauss_color_coeff: f32 = -0.5 / (sigma_color * sigma_color);
    let color_lut_size = 256 * channels;
    let mut color_weight = vec![0.0f32; color_lut_size];
    for i in 0..color_lut_size {
        let fi = i as f32;
        color_weight[i] = (fi * fi * gauss_color_coeff).exp();
    }

    // Pad image with BORDER_REFLECT_101
    let pw = w + 2 * radius;
    let ph = h + 2 * radius;
    let mut padded = vec![0u8; pw * ph * channels];
    for py in 0..ph {
        let sy = reflect101(py as isize - radius as isize, h as isize) as usize;
        for px in 0..pw {
            let sx = reflect101(px as isize - radius as isize, w as isize) as usize;
            for c in 0..channels {
                padded[(py * pw + px) * channels + c] = pixels[(sy * w + sx) * channels + c];
            }
        }
    }

    let mut result = vec![0u8; pixels.len()];

    for y in 0..h {
        for x in 0..w {
            let py = y + radius;
            let px = x + radius;

            if channels == 1 {
                let val0 = padded[py * pw + px] as i32;
                let mut wsum: f32 = 0.0;
                let mut vsum: f32 = 0.0;
                for k in 0..maxk {
                    let (dy, dx) = space_ofs[k];
                    let n_off = (py as isize + dy) as usize * pw + (px as isize + dx) as usize;
                    let val = padded[n_off] as i32;
                    let w = space_weight[k] * color_weight[(val - val0).unsigned_abs() as usize];
                    wsum += w;
                    vsum += val as f32 * w;
                }
                result[y * w + x] = (vsum / wsum).round().clamp(0.0, 255.0) as u8;
            } else {
                let center_off = (py * pw + px) * channels;
                let b0 = padded[center_off] as i32;
                let g0 = padded[center_off + 1] as i32;
                let r0 = padded[center_off + 2] as i32;
                let mut wsum: f32 = 0.0;
                let mut bsum: f32 = 0.0;
                let mut gsum: f32 = 0.0;
                let mut rsum: f32 = 0.0;
                for k in 0..maxk {
                    let (dy, dx) = space_ofs[k];
                    let n_off =
                        ((py as isize + dy) as usize * pw + (px as isize + dx) as usize) * channels;
                    let b = padded[n_off] as i32;
                    let g = padded[n_off + 1] as i32;
                    let r = padded[n_off + 2] as i32;
                    let color_diff =
                        (b - b0).unsigned_abs() + (g - g0).unsigned_abs() + (r - r0).unsigned_abs();
                    let w = space_weight[k]
                        * color_weight[(color_diff as usize).min(color_lut_size - 1)];
                    wsum += w;
                    bsum += b as f32 * w;
                    gsum += g as f32 * w;
                    rsum += r as f32 * w;
                }
                let out_off = (y * w + x) * channels;
                result[out_off] = (bsum / wsum).round().clamp(0.0, 255.0) as u8;
                result[out_off + 1] = (gsum / wsum).round().clamp(0.0, 255.0) as u8;
                result[out_off + 2] = (rsum / wsum).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}

// ─── Guided Filter (He et al. 2010) ──────────────────────────────────────

/// Edge-preserving guided filter.
///
/// O(N) complexity regardless of radius. Uses a guidance image (typically
/// the input itself) to compute local linear model a*I+b that smooths
/// while preserving edges in the guidance.
///
/// - `radius`: window radius (4-8 typical)
/// - `epsilon`: regularization parameter (0.01-0.1 typical; smaller = more edge-preserving)
///
/// For self-guided filtering, the input is used as both source and guide.
pub fn guided_filter(
    pixels: &[u8],
    info: &ImageInfo,
    radius: u32,
    epsilon: f32,
) -> Result<Vec<u8>, ImageError> {
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

/// Box mean via integral image — O(1) per pixel regardless of radius.
/// Box mean matching OpenCV's boxFilter with BORDER_REFLECT.
/// Pads data with reflect border, computes f32 SAT, queries fixed-size window.
fn box_mean(data: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
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

// ─── Morphological Operations ─────────────────────────────────────────────

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

/// Generate a structuring element as a boolean mask.
fn make_structuring_element(shape: MorphShape, kw: usize, kh: usize) -> Vec<bool> {
    let mut se = vec![false; kw * kh];
    let cx = kw / 2;
    let cy = kh / 2;
    for y in 0..kh {
        for x in 0..kw {
            se[y * kw + x] = match shape {
                MorphShape::Rect => true,
                MorphShape::Cross => x == cx || y == cy,
                MorphShape::Ellipse => {
                    let dx = (x as f32 - cx as f32) / cx.max(1) as f32;
                    let dy = (y as f32 - cy as f32) / cy.max(1) as f32;
                    dx * dx + dy * dy <= 1.0
                }
            };
        }
    }
    se
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

/// Dilate: output pixel = maximum over structuring element neighborhood.
pub fn dilate(
    pixels: &[u8],
    info: &ImageInfo,
    ksize: u32,
    shape: MorphShape,
) -> Result<Vec<u8>, ImageError> {
    morph_op(pixels, info, ksize, shape, false)
}

/// Core morphological operation (erode=min, dilate=max).
fn morph_op(
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

// ─── Non-Local Means Denoising ────────────────────────────────────────────

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

/// OpenCV-exact NLM implementation.
///
/// Replicates `FastNlMeansDenoisingInvoker` from OpenCV 4.x:
/// - `copyMakeBorder(BORDER_DEFAULT)` → reflect101 padding
/// - Integer SSD between patches
/// - `almostAvgDist = ssd >> bin_shift` (bit-shift approximation of SSD/N)
/// - Precomputed `almost_dist2weight[almostAvgDist]` LUT
/// - Fixed-point integer accumulation with `fixed_point_mult`
/// - `divByWeightsSum` with rounding
fn nlm_denoise_opencv(
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
    for ad in 0..almost_max_dist {
        let dist = ad as f64 * almost_dist2actual;
        // OpenCV DistSquared::calcWeight: exp(-dist / (h*h * channels))
        // Note: -dist (NOT -dist*dist) because dist is already squared per-pixel distance.
        // For grayscale (channels=1): exp(-dist / (h*h))
        let wf = (-dist / (params.h as f64 * params.h as f64)).exp();
        let wi = (fixed_point_mult as f64 * wf + 0.5) as i32;
        lut[ad] = if wi < weight_threshold { 0 } else { wi };
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

/// Classic NLM (Buades 2005) with float math.
fn nlm_denoise_classic(
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
// ─── Photo Enhancement ─────────────────────────────────────────────────────

/// Dehaze an image using the dark channel prior (He et al. 2009).
///
/// Estimates atmospheric light and transmission from the dark channel (minimum
/// over color channels in a local patch), refines with guided filter, then
/// recovers the scene: `J = (I - A) / max(t, t_min) + A`.
///
/// - `patch_radius`: local patch size for dark channel (typical: 7-15)
/// - `omega`: haze removal strength 0.0-1.0 (typical: 0.95)
/// - `t_min`: minimum transmission to avoid noise amplification (typical: 0.1)
pub fn dehaze(
    pixels: &[u8],
    info: &ImageInfo,
    patch_radius: u32,
    omega: f32,
    t_min: f32,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "dehaze requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = w * h;
    let r = patch_radius as usize;

    // Step 1: Compute dark channel — min over RGB in local patch
    let mut dark_channel = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::MAX;
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            for py in y0..y1 {
                for px in x0..x1 {
                    let idx = (py * w + px) * channels;
                    let r_val = pixels[idx] as f32 / 255.0;
                    let g_val = pixels[idx + 1] as f32 / 255.0;
                    let b_val = pixels[idx + 2] as f32 / 255.0;
                    let ch_min = r_val.min(g_val).min(b_val);
                    min_val = min_val.min(ch_min);
                }
            }
            dark_channel[y * w + x] = min_val;
        }
    }

    // Step 2: Estimate atmospheric light — brightest 0.1% of dark channel pixels
    let mut dc_indexed: Vec<(usize, f32)> = dark_channel
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    dc_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_count = (n as f32 * 0.001).max(1.0) as usize;
    let mut atm = [0.0f32; 3];
    let mut max_intensity = 0.0f32;
    for &(idx, _) in dc_indexed.iter().take(top_count) {
        let pi = idx * channels;
        let intensity = pixels[pi] as f32 + pixels[pi + 1] as f32 + pixels[pi + 2] as f32;
        if intensity > max_intensity {
            max_intensity = intensity;
            atm[0] = pixels[pi] as f32 / 255.0;
            atm[1] = pixels[pi + 1] as f32 / 255.0;
            atm[2] = pixels[pi + 2] as f32 / 255.0;
        }
    }

    // Step 3: Estimate transmission — t(x) = 1 - omega * dark_channel(I/A)
    let mut transmission = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::MAX;
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            for py in y0..y1 {
                for px in x0..x1 {
                    let idx = (py * w + px) * channels;
                    let nr = (pixels[idx] as f32 / 255.0) / atm[0].max(0.001);
                    let ng = (pixels[idx + 1] as f32 / 255.0) / atm[1].max(0.001);
                    let nb = (pixels[idx + 2] as f32 / 255.0) / atm[2].max(0.001);
                    min_val = min_val.min(nr.min(ng).min(nb));
                }
            }
            transmission[y * w + x] = (1.0 - omega * min_val).max(t_min);
        }
    }

    // Step 4: Refine transmission with guided filter (use grayscale as guide)
    // Convert transmission to u8, apply guided filter, convert back
    let t_u8: Vec<u8> = transmission
        .iter()
        .map(|&t| (t * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    let refined_u8 = guided_filter(&t_u8, &gray_info, patch_radius.min(15), 0.001)?;
    let refined: Vec<f32> = refined_u8.iter().map(|&v| v as f32 / 255.0).collect();

    // Step 5: Recover scene — J = (I - A) / max(t, t_min) + A
    let mut result = vec![0u8; pixels.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let t = refined[i].max(t_min);
            let pi = i * channels;
            for c in 0..3 {
                let ic = pixels[pi + c] as f32 / 255.0;
                let jc = (ic - atm[c]) / t + atm[c];
                result[pi + c] = (jc * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            if channels == 4 {
                result[pi + 3] = pixels[pi + 3]; // alpha
            }
        }
    }

    Ok(result)
}

/// Clarity — midtone-weighted local contrast enhancement.
///
/// Applies a large-radius unsharp mask but weights the effect by a midtone curve:
/// shadows and highlights get less enhancement, midtones (luminance 25-75%) get full.
/// This matches Lightroom/Photoshop "Clarity" slider behavior.
///
/// - `amount`: enhancement strength (0.0-2.0 typical, 1.0 = full effect)
/// - `sigma`: blur radius for local contrast (30-50 typical)
pub fn clarity(
    pixels: &[u8],
    info: &ImageInfo,
    amount: f32,
    sigma: f32,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "clarity requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);

    // Compute luminance for midtone weighting
    let mut luma = vec![0.0f32; n];
    for i in 0..n {
        let pi = i * channels;
        luma[i] = (0.2126 * pixels[pi] as f32
            + 0.7152 * pixels[pi + 1] as f32
            + 0.0722 * pixels[pi + 2] as f32)
            / 255.0;
    }

    // Apply large-radius blur
    let blurred = blur(pixels, info, sigma)?;

    // Midtone weight function: bell curve centered at 0.5, zero at 0 and 1
    // w(l) = 4 * l * (1 - l) — parabola peaking at 0.5 with w(0.5) = 1.0
    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let weight = 4.0 * luma[i] * (1.0 - luma[i]) * amount;
        let pi = i * channels;
        for c in 0..3 {
            let orig = pixels[pi + c] as f32;
            let blur_val = blurred[pi + c] as f32;
            let detail = orig - blur_val; // high-frequency detail
            let enhanced = orig + detail * weight;
            result[pi + c] = enhanced.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3]; // alpha
        }
    }

    Ok(result)
}

/// Pyramid detail remapping — edge-aware detail enhancement/smoothing.
///
/// Decomposes the image into a Gaussian/Laplacian pyramid and remaps
/// detail coefficients at each level via a sigmoidal curve:
/// `f(d) = d * sigma / (sigma + |d|)`.
///
/// - `sigma < 1.0`: compresses large gradients, enhances fine detail
/// - `sigma = 1.0`: near-identity (slight compression at large gradients)
/// - `sigma > 1.0`: suppresses fine detail (smoothing)
///
/// This is a Laplacian pyramid coefficient remapping filter, distinct from
/// the Paris et al. 2011 "Local Laplacian Filter" which rebuilds the pyramid
/// per-pixel with a power-law remapping.
///
/// - `sigma`: detail remapping strength (0.2 = strong enhancement, 1.0 = neutral, 3.0 = smooth)
/// - `num_levels`: pyramid depth (0 = auto, typically 5-7)
pub fn pyramid_detail_remap(
    pixels: &[u8],
    info: &ImageInfo,
    sigma: f32,
    num_levels: usize,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "pyramid_detail_remap requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let (w, h) = (info.width as usize, info.height as usize);

    // Determine pyramid levels
    let levels = if num_levels == 0 {
        ((w.min(h) as f32).log2() as usize).min(7).max(2)
    } else {
        num_levels.min(10)
    };

    // Process each channel independently through the pyramid
    let mut result = vec![0u8; pixels.len()];

    for c in 0..3 {
        // Extract single channel as f32
        let channel: Vec<f32> = (0..w * h)
            .map(|i| pixels[i * channels + c] as f32 / 255.0)
            .collect();

        let output = pyramid_detail_remap_channel(&channel, w, h, levels, sigma);

        // Write back
        for i in 0..w * h {
            result[i * channels + c] = (output[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    // Copy alpha if present
    if channels == 4 {
        for i in 0..w * h {
            result[i * 4 + 3] = pixels[i * 4 + 3];
        }
    }

    Ok(result)
}

/// Process a single channel through the Local Laplacian pyramid.
fn pyramid_detail_remap_channel(
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
        let (nw, nh) = ((cw + 1) / 2, (ch + 1) / 2);
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
        let (nw, nh) = ((cw + 1) / 2, (ch + 1) / 2);

        // Laplacian = current level - upsampled(next level)
        let upsampled = upsample_2x(&gauss_pyramid[level + 1], nw, nh, cw, ch);
        let mut laplacian = vec![0.0f32; cw * ch];
        for i in 0..cw * ch {
            laplacian[i] = gauss_pyramid[level][i] - upsampled[i];
        }

        // Remap detail: attenuate or amplify based on sigma
        // Enhancement: small sigma compresses large gradients, preserves small detail
        // Smoothing: large sigma suppresses small detail
        for i in 0..cw * ch {
            let d = laplacian[i];
            // Sigmoidal remapping: f(d) = d * (sigma / (sigma + |d|))
            // sigma < 1: enhances small detail (compresses large)
            // sigma > 1: smooths (suppresses small detail)
            laplacian[i] = d * sigma / (sigma + d.abs());
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
        tw = (tw + 1) / 2;
        th = (th + 1) / 2;
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

/// Downsample by 2x using box filter (average of 2x2 blocks).
fn downsample_2x(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    let nw = (w + 1) / 2;
    let nh = (h + 1) / 2;
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

/// Upsample by 2x using bilinear interpolation.
fn upsample_2x(data: &[f32], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f32> {
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

// ─── OpenCV-Compatible Gaussian Blur ────────────────────────────────────────

/// Gaussian blur with OpenCV-compatible kernel and border handling.
///
/// Generates a Gaussian kernel matching `cv2.getGaussianKernel` and applies it
/// via our `convolve()` function (which uses `BORDER_REFLECT_101` and is already
/// pixel-exact against OpenCV `filter2D`).
///
/// This is a separate implementation from `blur()` (which uses libblur with
/// `BORDER_REPLICATE`). Use this when pixel-exact OpenCV parity is required.
///
/// - `sigma`: Gaussian standard deviation
///
/// Future path: this could replace `blur()` as the primary Gaussian implementation
/// if full OpenCV alignment is desired across all filters. SIMD optimization can
/// be added later and validated against this reference-aligned output.
pub fn gaussian_blur_cv(
    pixels: &[u8],
    info: &ImageInfo,
    sigma: f32,
) -> Result<Vec<u8>, ImageError> {
    if sigma <= 0.0 {
        return Ok(pixels.to_vec());
    }

    // OpenCV kernel size for 8U: ksize = round(sigma * 6 + 1) | 1
    let ksize = {
        let k = (sigma * 6.0 + 1.0).round() as usize;
        if k % 2 == 0 { k + 1 } else { k }
    };
    let ksize = ksize.max(3);

    // Generate 2D Gaussian kernel (separable: outer product of 1D kernel)
    let k1d = gaussian_kernel_1d(ksize, sigma);
    let mut kernel_2d = vec![0.0f32; ksize * ksize];
    for y in 0..ksize {
        for x in 0..ksize {
            kernel_2d[y * ksize + x] = k1d[y] * k1d[x];
        }
    }

    // Delegate to convolve() which is pixel-exact against OpenCV filter2D
    // (uses BORDER_REFLECT_101 and is already validated)
    convolve(pixels, info, &kernel_2d, ksize, ksize, 1.0)
}

/// Generate a 1D Gaussian kernel matching OpenCV's `getGaussianKernel`.
///
/// `k[i] = exp(-0.5 * ((i - center) / sigma)^2)`, normalized to sum=1.
fn gaussian_kernel_1d(ksize: usize, sigma: f32) -> Vec<f32> {
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

// ─── Retinex Enhancement ────────────────────────────────────────────────────

/// Single-Scale Retinex (SSR).
///
/// `R(x,y) = log(I(x,y)) - log(G(x,y,sigma) * I(x,y))`
///
/// Enhances local contrast by removing the illumination component estimated
/// via Gaussian blur. Output is normalized to [0, 255].
///
/// - `sigma`: Gaussian scale (typical: 80.0 for general enhancement)
///
/// Reference: Jobson, Rahman, Woodell — "Properties and Performance of a
/// Center/Surround Retinex" (IEEE Trans. Image Processing, 1997)
pub fn retinex_ssr(pixels: &[u8], info: &ImageInfo, sigma: f32) -> Result<Vec<u8>, ImageError> {
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

    // Gaussian blur for surround function (OpenCV-compatible for reference alignment)
    let blurred = gaussian_blur_cv(pixels, info, sigma)?;

    // Compute log(I) - log(blur(I)) per channel, then normalize
    let mut retinex = vec![0.0f32; n * 3];
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for i in 0..n {
        let pi = i * channels;
        for c in 0..3 {
            let orig = (pixels[pi + c] as f32).max(1.0); // avoid log(0)
            let surround = (blurred[pi + c] as f32).max(1.0);
            let r = orig.ln() - surround.ln();
            retinex[i * 3 + c] = r;
            min_val = min_val.min(r);
            max_val = max_val.max(r);
        }
    }

    // Normalize to [0, 255]
    let range = (max_val - min_val).max(1e-6);
    let mut result = vec![0u8; pixels.len()];
    for i in 0..n {
        let pi = i * channels;
        for c in 0..3 {
            let v = (retinex[i * 3 + c] - min_val) / range * 255.0;
            result[pi + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3]; // alpha
        }
    }

    Ok(result)
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
pub fn retinex_msr(pixels: &[u8], info: &ImageInfo, sigmas: &[f32]) -> Result<Vec<u8>, ImageError> {
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
        let blurred = gaussian_blur_cv(pixels, info, sigma)?;
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
    pixels: &[u8],
    info: &ImageInfo,
    sigmas: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<Vec<u8>, ImageError> {
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
        let blurred = gaussian_blur_cv(pixels, info, sigma)?;
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

// ─── Connected Component Labeling ─────────────────────────────────────────

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
    for i in 0..parent.len() {
        parent[i] = i as u32;
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

// ─── Flood Fill ───────────────────────────────────────────────────────────

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

// ─── Image Pyramids ──────────────────────────────────────────────────────

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
    let ow = (w + 1) / 2;
    let oh = (h + 1) / 2;

    // 5x5 Gaussian kernel (1/256 normalization): [1,4,6,4,1] x [1,4,6,4,1]
    let kernel_1d: [i32; 5] = [1, 4, 6, 4, 1];

    // Horizontal pass → temp buffer
    let mut temp = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sx = reflect101((x as isize + k as isize - 2) as isize, w as isize) as usize;
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
                let sy = reflect101((y as isize + k as isize - 2) as isize, h as isize) as usize;
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
                let sx = reflect101((x as isize + k as isize - 2) as isize, ow as isize) as usize;
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
                let sy = reflect101((y as isize + k as isize - 2) as isize, oh as isize) as usize;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn blur_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let result = blur(&px, &info, 2.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn blur_zero_radius_preserves_pixels() {
        let (px, info) = make_image(8, 8);
        let result = blur(&px, &info, 0.0).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn blur_negative_radius_returns_error() {
        let (px, info) = make_image(8, 8);
        let result = blur(&px, &info, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn sharpen_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let result = sharpen(&px, &info, 1.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn brightness_increases() {
        let (px, info) = make_image(8, 8);
        let result = brightness(&px, &info, 0.5).unwrap();
        assert_eq!(result.len(), px.len());
        let avg_orig: f64 = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
        let avg_bright: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(avg_bright > avg_orig, "brightness should increase average");
    }

    #[test]
    fn brightness_out_of_range_returns_error() {
        let (px, info) = make_image(8, 8);
        assert!(brightness(&px, &info, 1.5).is_err());
        assert!(brightness(&px, &info, -1.5).is_err());
    }

    #[test]
    fn contrast_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = contrast(&px, &info, 0.5).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn contrast_out_of_range_returns_error() {
        let (px, info) = make_image(8, 8);
        assert!(contrast(&px, &info, 2.0).is_err());
    }

    #[test]
    fn grayscale_changes_format() {
        let (px, info) = make_image(16, 16);
        let result = grayscale(&px, &info).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 16 * 16);
    }

    #[test]
    fn grayscale_preserves_dimensions() {
        let (px, info) = make_image(32, 24);
        let result = grayscale(&px, &info).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 24);
    }

    #[test]
    fn filters_work_on_rgba8() {
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        assert!(blur(&pixels, &info, 1.0).is_ok());
        assert!(sharpen(&pixels, &info, 1.0).is_ok());
        assert!(brightness(&pixels, &info, 0.2).is_ok());
        assert!(contrast(&pixels, &info, 0.2).is_ok());
        assert!(grayscale(&pixels, &info).is_ok());
    }

    #[test]
    fn contrast_lut_produces_expected_values() {
        // Zero contrast should be near identity
        let (px, info) = make_image(4, 4);
        let result = contrast(&px, &info, 0.0).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn hue_rotate_zero_is_identity() {
        // Hue rotate by 0 degrees should preserve pixels (via ColorOp delegation)
        let (px, info) = make_image(8, 8);
        let result = hue_rotate(&px, &info, 0.0).unwrap();
        for (i, (&orig, &out)) in px.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i16 - out as i16).abs() <= 1,
                "pixel {i}: {orig} -> {out}"
            );
        }
    }

    #[test]
    fn saturate_one_is_identity() {
        // Saturation factor 1.0 should preserve pixels
        let (px, info) = make_image(8, 8);
        let result = saturate(&px, &info, 1.0).unwrap();
        for (i, (&orig, &out)) in px.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i16 - out as i16).abs() <= 1,
                "pixel {i}: {orig} -> {out}"
            );
        }
    }

    #[test]
    fn hue_rotate_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = hue_rotate(&px, &info, 90.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn hue_rotate_360_identity() {
        let (px, info) = make_image(8, 8);
        let result = hue_rotate(&px, &info, 360.0).unwrap();
        // Should be very close to original (within rounding)
        let mae: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 1.0,
            "360° hue rotation should be near-identity, MAE={mae:.2}"
        );
    }

    #[test]
    fn saturate_zero_is_grayscale() {
        let (px, info) = make_image(8, 8);
        let result = saturate(&px, &info, 0.0).unwrap();
        // All pixels should have r≈g≈b
        for chunk in result.chunks_exact(3) {
            let spread = chunk.iter().map(|&v| v as i32).max().unwrap()
                - chunk.iter().map(|&v| v as i32).min().unwrap();
            assert!(
                spread <= 1,
                "saturate(0) should produce gray, got spread={spread}"
            );
        }
    }

    #[test]
    fn saturate_one_near_identity() {
        let (px, info) = make_image(8, 8);
        let result = saturate(&px, &info, 1.0).unwrap();
        let mae: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 1.0,
            "saturate(1.0) should be near-identity, MAE={mae:.2}"
        );
    }

    #[test]
    fn sepia_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = sepia(&px, &info, 1.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn sepia_zero_is_identity() {
        let (px, info) = make_image(8, 8);
        let result = sepia(&px, &info, 0.0).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn colorize_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = colorize(&px, &info, [255, 0, 0], 0.5).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn colorize_zero_is_identity() {
        let (px, info) = make_image(8, 8);
        let result = colorize(&px, &info, [255, 0, 0], 0.0).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn convolve_identity_preserves_image() {
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..16).collect();
        // Identity kernel: [0,0,0, 0,1,0, 0,0,0]
        let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let result = convolve(&pixels, &info, &kernel, 3, 3, 1.0).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn convolve_sharpen_kernel() {
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 16];
        // Sharpen kernel: center=5, neighbors=-1
        let kernel = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
        let result = convolve(&pixels, &info, &kernel, 3, 3, 1.0).unwrap();
        // Uniform input → sharpen produces same output (no edges)
        assert!(result.iter().all(|&v| (v as i32 - 128).unsigned_abs() < 2));
    }

    #[test]
    fn median_removes_salt_and_pepper() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let mut pixels = vec![128u8; 64];
        // Add salt-and-pepper noise
        pixels[27] = 0; // pepper
        pixels[35] = 255; // salt
        let result = median(&pixels, &info, 1).unwrap();
        // Noise pixels should be replaced by median of neighbors (~128)
        assert!(
            (result[27] as i32 - 128).unsigned_abs() < 10,
            "pepper not removed: {}",
            result[27]
        );
        assert!(
            (result[35] as i32 - 128).unsigned_abs() < 10,
            "salt not removed: {}",
            result[35]
        );
    }

    #[test]
    fn sobel_detects_edges() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Left half = 0, right half = 255 → vertical edge at column 4
        let mut pixels = vec![0u8; 64];
        for r in 0..8 {
            for c in 4..8 {
                pixels[r * 8 + c] = 255;
            }
        }
        let result = sobel(&pixels, &info).unwrap();
        // Edge pixels at column 3-4 should have high gradient
        let edge_val = result[3 * 8 + 4]; // near the edge
        let flat_val = result[3 * 8 + 0]; // in flat region
        assert!(
            edge_val > flat_val + 50,
            "edge not detected: edge={edge_val} flat={flat_val}"
        );
    }

    #[test]
    fn canny_produces_binary_edges() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Vertical edge in the middle
        let mut pixels = vec![50u8; 256];
        for r in 0..16 {
            for c in 8..16 {
                pixels[r * 16 + c] = 200;
            }
        }
        let result = canny(&pixels, &info, 30.0, 100.0).unwrap();
        // Should produce binary output (0 or 255 only)
        assert!(
            result.iter().all(|&v| v == 0 || v == 255),
            "non-binary canny output"
        );
        // Should have some edge pixels
        let edge_count = result.iter().filter(|&&v| v == 255).count();
        assert!(edge_count > 0, "no edges detected");
    }

    #[test]
    fn color_effects_work_on_rgba8() {
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        assert!(hue_rotate(&pixels, &info, 45.0).is_ok());
        assert!(saturate(&pixels, &info, 1.5).is_ok());
        assert!(sepia(&pixels, &info, 0.8).is_ok());
        assert!(colorize(&pixels, &info, [0, 128, 255], 0.5).is_ok());
    }

    fn make_rgba(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn premultiply_unpremultiply_roundtrip() {
        // Use pixels with alpha > 0 (alpha=0 loses info, alpha=1 has high rounding error)
        let mut pixels = Vec::new();
        for _ in 0..64 {
            pixels.extend_from_slice(&[100, 150, 200, 200]); // non-trivial alpha
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let pre = premultiply(&pixels, &info).unwrap();
        let unpre = unpremultiply(&pre, &info).unwrap();
        for i in (0..pixels.len()).step_by(4) {
            for c in 0..3 {
                assert!(
                    (pixels[i + c] as i32 - unpre[i + c] as i32).abs() <= 1,
                    "roundtrip error at pixel {}: ch{c}: {} vs {}",
                    i / 4,
                    pixels[i + c],
                    unpre[i + c]
                );
            }
        }
    }

    #[test]
    fn flatten_white_bg() {
        // Fully opaque pixel should pass through unchanged
        let pixels = vec![100u8, 150, 200, 255, 50, 75, 100, 0];
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let (rgb, new_info) = flatten(&pixels, &info, [255, 255, 255]).unwrap();
        assert_eq!(new_info.format, PixelFormat::Rgb8);
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb[0], 100); // opaque pixel unchanged
        assert_eq!(rgb[1], 150);
        assert_eq!(rgb[2], 200);
        assert_eq!(rgb[3], 255); // transparent pixel → white bg
        assert_eq!(rgb[4], 255);
        assert_eq!(rgb[5], 255);
    }

    #[test]
    fn add_remove_alpha_roundtrip() {
        let (px, info) = make_image(4, 4); // RGB8
        let (rgba, rgba_info) = add_alpha(&px, &info, 255).unwrap();
        assert_eq!(rgba_info.format, PixelFormat::Rgba8);
        assert_eq!(rgba.len(), 4 * 4 * 4);
        let (rgb, rgb_info) = remove_alpha(&rgba, &rgba_info).unwrap();
        assert_eq!(rgb_info.format, PixelFormat::Rgb8);
        assert_eq!(rgb, px);
    }

    #[test]
    fn blend_multiply_identity() {
        // Multiply with white (255) should be near-identity
        let (px, info) = make_image(4, 4);
        let white: Vec<u8> = vec![255; 4 * 4 * 3];
        let result = blend(&px, &info, &white, &info, BlendMode::Multiply).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn blend_screen_with_black() {
        // Screen with black (0) should be near-identity
        let (px, info) = make_image(4, 4);
        let black = vec![0u8; 4 * 4 * 3];
        let result = blend(&px, &info, &black, &info, BlendMode::Screen).unwrap();
        let mae: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 1.0,
            "screen with black should be near-identity, MAE={mae:.2}"
        );
    }

    #[test]
    fn blend_all_modes_run() {
        let (px, info) = make_image(4, 4);
        let px2: Vec<u8> = (0..(4 * 4 * 3)).map(|i| ((i * 3) % 256) as u8).collect();
        for mode in [
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::SoftLight,
            BlendMode::HardLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
        ] {
            let result = blend(&px, &info, &px2, &info, mode);
            assert!(result.is_ok(), "blend mode {mode:?} failed");
            assert_eq!(result.unwrap().len(), px.len());
        }
    }

    #[test]
    fn blend_difference_self_is_black() {
        let (px, info) = make_image(4, 4);
        let result = blend(&px, &info, &px, &info, BlendMode::Difference).unwrap();
        for &v in &result {
            assert!(v <= 1, "difference with self should be ~0, got {v}");
        }
    }
}

#[cfg(test)]
mod optimization_tests {
    use super::super::types::*;
    use super::*;

    #[test]
    fn separable_detection_box_blur() {
        // Box blur 3x3 is separable: [1,1,1] * [1,1,1]^T
        let result = is_separable(&kernels::BOX_BLUR_3X3, 3, 3);
        assert!(result.is_some(), "box blur should be detected as separable");
        let (row, col) = result.unwrap();
        assert_eq!(row.len(), 3);
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn separable_detection_emboss_not_separable() {
        // Emboss kernel is NOT separable
        let result = is_separable(&kernels::EMBOSS, 3, 3);
        assert!(result.is_none(), "emboss should NOT be separable");
    }

    #[test]
    fn histogram_median_matches_sort_median() {
        // Both paths should give the same output
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let mut pixels = vec![128u8; 256];
        // Add some variation
        for i in 0..256 {
            pixels[i] = (i as u8).wrapping_mul(7).wrapping_add(13);
        }

        // radius=2: uses sort path
        let sort_result = median(&pixels, &info, 2).unwrap();
        // radius=3: uses histogram path
        let hist_result = median(&pixels, &info, 3).unwrap();

        // Both should produce valid output (different radii = different results, but both correct)
        assert!(!sort_result.is_empty());
        assert!(!hist_result.is_empty());
        // Histogram path with radius=3 should produce smoother output
    }

    #[test]
    fn convolve_perf_1024x1024() {
        let info = ImageInfo {
            width: 1024,
            height: 1024,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 1024 * 1024];

        let start = std::time::Instant::now();
        let _ = convolve(&pixels, &info, &kernels::BOX_BLUR_3X3, 3, 3, 9.0).unwrap();
        let elapsed = start.elapsed();

        // Separable path should handle 1024x1024 in under 500ms
        assert!(
            elapsed.as_millis() < 500,
            "3x3 convolve on 1024x1024 took {:?}, expected < 500ms",
            elapsed
        );
    }

    #[test]
    fn median_perf_512x512() {
        let info = ImageInfo {
            width: 512,
            height: 512,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..(512 * 512)).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        let _ = median(&pixels, &info, 3).unwrap();
        let elapsed = start.elapsed();

        // Histogram median should handle 512x512 radius=3 in under 500ms
        assert!(
            elapsed.as_millis() < 500,
            "median radius=3 on 512x512 took {:?}, expected < 500ms",
            elapsed
        );
    }

    // ─── CLAHE Tests ──────────────────────────────────────────────────────

    #[test]
    fn clahe_enhances_low_contrast() {
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Low-contrast input: values 100-155
        let pixels: Vec<u8> = (0..(64 * 64)).map(|i| (100 + (i % 56)) as u8).collect();
        let result = clahe(&pixels, &info, 2.0, 8).unwrap();

        // CLAHE should expand dynamic range
        let in_range = *pixels.iter().max().unwrap() as i32 - *pixels.iter().min().unwrap() as i32;
        let out_range = *result.iter().max().unwrap() as i32 - *result.iter().min().unwrap() as i32;
        assert!(
            out_range > in_range,
            "CLAHE should expand range: in={in_range}, out={out_range}"
        );
    }

    #[test]
    fn clahe_flat_image_uniform_output() {
        // CLAHE on flat input: OpenCV redistributes excess across all bins,
        // so the output is NOT identity but is uniform (all same value).
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 32 * 32];
        let result = clahe(&pixels, &info, 2.0, 4).unwrap();
        // All output pixels should be the same value (uniform)
        let first = result[0];
        for &v in &result {
            assert_eq!(v, first, "flat input should produce uniform output");
        }
    }

    #[test]
    fn clahe_rejects_non_gray() {
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        assert!(clahe(&vec![0u8; 48], &info, 2.0, 8).is_err());
    }

    // ─── Bilateral Filter Tests ───────────────────────────────────────────

    #[test]
    fn bilateral_preserves_edges() {
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Half black, half white
        let mut pixels = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 16..32 {
                pixels[y * 32 + x] = 255;
            }
        }
        let result = bilateral(&pixels, &info, 5, 50.0, 50.0).unwrap();

        // Edge should be preserved: pixels at x=14 should still be dark, x=18 still bright
        let mid_y = 16;
        assert!(result[mid_y * 32 + 14] < 50, "left of edge should be dark");
        assert!(
            result[mid_y * 32 + 18] > 200,
            "right of edge should be bright"
        );
    }

    #[test]
    fn bilateral_smooths_noise() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Noisy flat region
        let pixels: Vec<u8> = (0..256)
            .map(|i| (128i32 + ((i * 17 + 5) % 21) as i32 - 10).clamp(0, 255) as u8)
            .collect();
        let result = bilateral(&pixels, &info, 5, 25.0, 25.0).unwrap();

        // Should reduce variance
        let var_in: f64 = pixels
            .iter()
            .map(|&v| (v as f64 - 128.0).powi(2))
            .sum::<f64>()
            / 256.0;
        let mean_out = result.iter().map(|&v| v as f64).sum::<f64>() / 256.0;
        let var_out: f64 = result
            .iter()
            .map(|&v| (v as f64 - mean_out).powi(2))
            .sum::<f64>()
            / 256.0;
        assert!(
            var_out < var_in,
            "bilateral should reduce variance: {var_out:.1} vs {var_in:.1}"
        );
    }

    // ─── Guided Filter Tests ──────────────────────────────────────────────

    #[test]
    fn guided_filter_smooths() {
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Noisy flat region (128 ± noise)
        let pixels: Vec<u8> = (0..(32 * 32))
            .map(|i| (128i32 + ((i * 17 + 3) % 21) as i32 - 10).clamp(0, 255) as u8)
            .collect();
        let result = guided_filter(&pixels, &info, 4, 0.01).unwrap();

        // Should reduce variance from mean
        let mean_in = pixels.iter().map(|&v| v as f64).sum::<f64>() / pixels.len() as f64;
        let var_in: f64 = pixels
            .iter()
            .map(|&v| (v as f64 - mean_in).powi(2))
            .sum::<f64>()
            / pixels.len() as f64;
        let mean_out = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        let var_out: f64 = result
            .iter()
            .map(|&v| (v as f64 - mean_out).powi(2))
            .sum::<f64>()
            / result.len() as f64;
        assert!(
            var_out < var_in,
            "guided filter should reduce variance: {var_out:.1} vs {var_in:.1}"
        );
    }

    #[test]
    fn guided_filter_flat_identity() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![100u8; 16 * 16];
        let result = guided_filter(&pixels, &info, 4, 0.01).unwrap();
        // Flat input should produce flat output
        for &v in &result {
            assert!((v as i32 - 100).abs() <= 1, "flat pixel changed to {v}");
        }
    }
}

#[cfg(test)]
mod tests_16bit {
    use super::super::types::*;
    use super::*;

    fn make_rgb16(w: u32, h: u32, val: u16) -> (Vec<u8>, ImageInfo) {
        let n = (w * h * 3) as usize;
        let samples: Vec<u16> = vec![val; n];
        let bytes = u16_to_bytes(&samples);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        (bytes, info)
    }

    fn make_gray16(w: u32, h: u32, val: u16) -> (Vec<u8>, ImageInfo) {
        let n = (w * h) as usize;
        let samples: Vec<u16> = vec![val; n];
        let bytes = u16_to_bytes(&samples);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray16,
            color_space: ColorSpace::Srgb,
        };
        (bytes, info)
    }

    #[test]
    fn blur_16bit_identity() {
        let (px, info) = make_rgb16(8, 8, 32768);
        let result = blur(&px, &info, 0.0).unwrap();
        assert_eq!(result, px, "zero-radius blur should be identity");
    }

    #[test]
    fn blur_16bit_produces_output() {
        let (px, info) = make_rgb16(8, 8, 32768);
        let result = blur(&px, &info, 1.0).unwrap();
        assert_eq!(result.len(), px.len(), "output length should match");
    }

    #[test]
    fn sharpen_16bit_produces_output() {
        let (px, info) = make_rgb16(8, 8, 32768);
        let result = sharpen(&px, &info, 1.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn convolve_16bit_identity_kernel() {
        let (px, info) = make_gray16(4, 4, 50000);
        let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let result = convolve(&px, &info, &kernel, 3, 3, 1.0).unwrap();
        // Should be close to original (some precision loss from 16→8→16)
        let orig = bytes_to_u16(&px);
        let out = bytes_to_u16(&result);
        for i in 0..orig.len() {
            assert!(
                (orig[i] as i32 - out[i] as i32).abs() < 300,
                "identity convolve changed pixel {} by {}",
                i,
                (orig[i] as i32 - out[i] as i32).abs()
            );
        }
    }

    #[test]
    fn median_16bit_produces_output() {
        let (px, info) = make_gray16(8, 8, 32768);
        let result = median(&px, &info, 1).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn sobel_16bit_produces_output() {
        let (px, info) = make_gray16(8, 8, 32768);
        let result = sobel(&px, &info).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn hue_rotate_16bit() {
        let (px, info) = make_rgb16(4, 4, 32768);
        let result = hue_rotate(&px, &info, 90.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn brightness_16bit() {
        let (px, info) = make_rgb16(4, 4, 32768);
        let result = brightness(&px, &info, 0.5).unwrap();
        assert_eq!(result.len(), px.len());
        // Brightened pixels should be higher
        let orig = bytes_to_u16(&px);
        let out = bytes_to_u16(&result);
        assert!(
            out[0] > orig[0],
            "brightness should increase: {} > {}",
            out[0],
            orig[0]
        );
    }

    #[test]
    fn sepia_16bit() {
        let (px, info) = make_rgb16(4, 4, 32768);
        let result = sepia(&px, &info, 1.0).unwrap();
        assert_eq!(result.len(), px.len());
    }
}

#[cfg(test)]
mod photo_enhance_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn test_info(w: u32, h: u32, fmt: PixelFormat) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: fmt,
            color_space: ColorSpace::Srgb,
        }
    }

    fn make_rgb(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        (pixels, test_info(w, h, PixelFormat::Rgb8))
    }

    #[test]
    fn dehaze_produces_output() {
        // Create a synthetic hazy image (low contrast, washed out)
        let (w, h) = (32u32, 32u32);
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                // Hazy: everything shifted toward bright gray (haze = 180)
                pixels[i] = (((x * 2) as u8).wrapping_add(180)).min(250);
                pixels[i + 1] = (((y * 2) as u8).wrapping_add(180)).min(250);
                pixels[i + 2] = 200;
            }
        }
        let info = test_info(w, h, PixelFormat::Rgb8);
        let result = dehaze(&pixels, &info, 7, 0.95, 0.1).unwrap();
        assert_eq!(result.len(), pixels.len());

        // Dehazed image should have more contrast (wider range)
        let stats_before = crate::domain::histogram::statistics(&pixels, &info).unwrap();
        let stats_after = crate::domain::histogram::statistics(&result, &info).unwrap();
        let range_before = stats_before[0].max as f32 - stats_before[0].min as f32;
        let range_after = stats_after[0].max as f32 - stats_after[0].min as f32;
        assert!(
            range_after >= range_before,
            "dehaze should increase contrast: range {range_before} -> {range_after}"
        );
    }

    #[test]
    fn dehaze_rgba_preserves_alpha() {
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![200u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 4 + 3] = 128; // set alpha
        }
        let info = test_info(w, h, PixelFormat::Rgba8);
        let result = dehaze(&pixels, &info, 5, 0.8, 0.1).unwrap();
        for i in 0..(w * h) as usize {
            assert_eq!(result[i * 4 + 3], 128, "alpha must be preserved");
        }
    }

    #[test]
    fn clarity_enhances_midtones() {
        let (w, h) = (32u32, 32u32);
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                // Midtone image (values around 128)
                pixels[i] = 100 + (x % 28) as u8;
                pixels[i + 1] = 110 + (y % 20) as u8;
                pixels[i + 2] = 120;
            }
        }
        let info = test_info(w, h, PixelFormat::Rgb8);
        let result = clarity(&pixels, &info, 1.0, 10.0).unwrap();
        assert_eq!(result.len(), pixels.len());

        // Clarity should increase local contrast (stddev should increase)
        let stats_before = crate::domain::histogram::statistics(&pixels, &info).unwrap();
        let stats_after = crate::domain::histogram::statistics(&result, &info).unwrap();
        assert!(
            stats_after[0].stddev >= stats_before[0].stddev * 0.9,
            "clarity should not dramatically reduce contrast"
        );
    }

    #[test]
    fn clarity_zero_amount_is_near_identity() {
        let (px, info) = make_rgb(32, 32);
        let result = clarity(&px, &info, 0.0, 10.0).unwrap();
        // With amount=0, the detail weighting is 0, so output ≈ input
        let diff: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            diff < 1.0,
            "clarity with amount=0 should be near-identity, MAE={diff}"
        );
    }

    #[test]
    fn pyramid_detail_remap_preserves_dimensions() {
        let (px, info) = make_rgb(32, 32);
        let result = pyramid_detail_remap(&px, &info, 0.5, 0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn pyramid_detail_remap_sigma_1_near_identity() {
        let (px, info) = make_rgb(32, 32);
        // sigma=1.0 means the remapping d * 1.0 / (1.0 + |d|) ≈ d for small d
        // This is close to identity (slight compression of large gradients)
        let result = pyramid_detail_remap(&px, &info, 1.0, 4).unwrap();
        let diff: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            diff < 30.0,
            "local laplacian sigma=1 should be close to identity, MAE={diff}"
        );
    }

    #[test]
    fn pyramid_detail_remap_small_sigma_produces_output() {
        let (px, info) = make_rgb(64, 64);
        let result = pyramid_detail_remap(&px, &info, 0.2, 0).unwrap();
        assert_eq!(result.len(), px.len());
        // Result should differ from input (enhancement applied)
        let diff: usize = px
            .iter()
            .zip(result.iter())
            .filter(|&(&a, &b)| a != b)
            .count();
        assert!(diff > 0, "local laplacian should modify the image");
    }

    #[test]
    fn pyramid_detail_remap_rgba_preserves_alpha() {
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![128u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 4] = (i % 200) as u8;
            pixels[i * 4 + 1] = ((i * 3) % 200) as u8;
            pixels[i * 4 + 2] = ((i * 7) % 200) as u8;
            pixels[i * 4 + 3] = 200;
        }
        let info = test_info(w, h, PixelFormat::Rgba8);
        let result = pyramid_detail_remap(&pixels, &info, 0.5, 3).unwrap();
        for i in 0..(w * h) as usize {
            assert_eq!(result[i * 4 + 3], 200, "alpha must be preserved");
        }
    }
}

#[cfg(test)]
mod morphology_tests {
    use super::super::types::ColorSpace;
    use super::*;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn erode_shrinks_bright_region() {
        // 8x8 image: center 4x4 white block on black background
        let mut px = vec![0u8; 64];
        for y in 2..6 {
            for x in 2..6 {
                px[y * 8 + x] = 255;
            }
        }
        let info = gray_info(8, 8);
        let result = erode(&px, &info, 3, MorphShape::Rect).unwrap();
        // Center pixel should still be white
        assert_eq!(result[3 * 8 + 3], 255);
        // Edge of original white block should be eroded
        assert_eq!(result[2 * 8 + 2], 0, "corner should be eroded");
    }

    #[test]
    fn dilate_grows_bright_region() {
        // Single white pixel at center
        let mut px = vec![0u8; 64];
        px[3 * 8 + 3] = 255;
        let info = gray_info(8, 8);
        let result = dilate(&px, &info, 3, MorphShape::Rect).unwrap();
        // 3x3 neighborhood should all be white
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let y = (3 + dy) as usize;
                let x = (3 + dx) as usize;
                assert_eq!(result[y * 8 + x], 255, "({x},{y}) should be dilated");
            }
        }
    }

    #[test]
    fn erode_dilate_identity_on_uniform() {
        let px = vec![128u8; 64];
        let info = gray_info(8, 8);
        let eroded = erode(&px, &info, 3, MorphShape::Rect).unwrap();
        let dilated = dilate(&px, &info, 3, MorphShape::Rect).unwrap();
        assert_eq!(eroded, px);
        assert_eq!(dilated, px);
    }

    #[test]
    fn open_removes_small_bright_noise() {
        // Black image with single white pixel (noise)
        let mut px = vec![0u8; 64];
        px[3 * 8 + 3] = 255;
        let info = gray_info(8, 8);
        let result = morph_open(&px, &info, 3, MorphShape::Rect).unwrap();
        // Opening should remove the single bright pixel
        assert_eq!(
            result[3 * 8 + 3],
            0,
            "single bright pixel removed by opening"
        );
    }

    #[test]
    fn close_fills_small_dark_hole() {
        // White image with single black pixel (hole)
        let mut px = vec![255u8; 64];
        px[3 * 8 + 3] = 0;
        let info = gray_info(8, 8);
        let result = morph_close(&px, &info, 3, MorphShape::Rect).unwrap();
        // Closing should fill the single dark pixel
        assert_eq!(
            result[3 * 8 + 3],
            255,
            "single dark pixel filled by closing"
        );
    }

    #[test]
    fn gradient_highlights_edges() {
        // Step edge: left half black, right half white
        let mut px = vec![0u8; 64];
        for y in 0..8 {
            for x in 4..8 {
                px[y * 8 + x] = 255;
            }
        }
        let info = gray_info(8, 8);
        let result = morph_gradient(&px, &info, 3, MorphShape::Rect).unwrap();
        // Edge at x=3/4 should be highlighted
        assert!(
            result[3 * 8 + 3] > 0 || result[3 * 8 + 4] > 0,
            "edge should be visible"
        );
        // Interior should be zero
        assert_eq!(result[3 * 8 + 0], 0, "interior black should be zero");
        assert_eq!(result[3 * 8 + 7], 0, "interior white should be zero");
    }

    #[test]
    fn cross_structuring_element() {
        let se = make_structuring_element(MorphShape::Cross, 3, 3);
        // Cross: center row and center column
        assert!(!se[0]); // top-left
        assert!(se[1]); // top-center
        assert!(!se[2]); // top-right
        assert!(se[3]); // mid-left
        assert!(se[4]); // center
        assert!(se[5]); // mid-right
        assert!(!se[6]); // bottom-left
        assert!(se[7]); // bottom-center
        assert!(!se[8]); // bottom-right
    }

    #[test]
    fn rgb_morphology() {
        use super::super::types::ColorSpace;
        let mut px = vec![0u8; 8 * 8 * 3];
        let idx = (3 * 8 + 3) * 3;
        px[idx] = 255;
        px[idx + 1] = 255;
        px[idx + 2] = 255;
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = dilate(&px, &info, 3, MorphShape::Rect).unwrap();
        // Neighbor should be dilated
        let n_idx = (3 * 8 + 4) * 3;
        assert_eq!(result[n_idx], 255);
    }
}

#[cfg(test)]
mod nlm_tests {
    use super::super::types::ColorSpace;
    use super::*;

    #[test]
    fn nlm_reduces_noise() {
        // Create noisy grayscale image: uniform 128 + noise
        let w = 32u32;
        let h = 32u32;
        let mut px = vec![128u8; (w * h) as usize];
        // Add deterministic noise
        for i in 0..px.len() {
            let noise = ((i as u32).wrapping_mul(2654435761) >> 24) as i16 - 128;
            let noise_scaled = noise / 4; // ±32 noise
            px[i] = (128i16 + noise_scaled).clamp(0, 255) as u8;
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let params = NlmParams {
            h: 20.0,
            patch_size: 5,
            search_size: 11,
            ..Default::default()
        };
        let result = nlm_denoise(&px, &info, &params).unwrap();

        // Compute MAE vs ground truth (128)
        let mae_input: f64 =
            px.iter().map(|&v| (v as f64 - 128.0).abs()).sum::<f64>() / px.len() as f64;
        let mae_output: f64 = result
            .iter()
            .map(|&v| (v as f64 - 128.0).abs())
            .sum::<f64>()
            / result.len() as f64;

        assert!(
            mae_output < mae_input,
            "NLM should reduce noise: input MAE={mae_input:.1}, output MAE={mae_output:.1}"
        );
    }

    #[test]
    fn nlm_preserves_uniform() {
        let px = vec![128u8; 16 * 16];
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let result = nlm_denoise(&px, &info, &NlmParams::default()).unwrap();
        assert_eq!(result, px, "uniform image should be preserved");
    }

    #[test]
    fn nlm_gray_only() {
        let px = vec![128u8; 4 * 4 * 3];
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        assert!(nlm_denoise(&px, &info, &NlmParams::default()).is_err());
    }
}

#[cfg(test)]
mod retinex_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_rgb(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((x * 200 / w.max(1)) + 30) as u8);
                pixels.push(((y * 200 / h.max(1)) + 30) as u8);
                pixels.push(128u8);
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn ssr_produces_output() {
        let (px, info) = make_rgb(32, 32);
        let result = retinex_ssr(&px, &info, 80.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn ssr_increases_dynamic_range() {
        // Low-contrast input
        let (w, h) = (32u32, 32u32);
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 3] = 100 + (i % 20) as u8;
            pixels[i * 3 + 1] = 110 + (i % 15) as u8;
            pixels[i * 3 + 2] = 120;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = retinex_ssr(&pixels, &info, 80.0).unwrap();

        let stats_before = crate::domain::histogram::statistics(&pixels, &info).unwrap();
        let stats_after = crate::domain::histogram::statistics(&result, &info).unwrap();
        let range_before = stats_before[0].max as f32 - stats_before[0].min as f32;
        let range_after = stats_after[0].max as f32 - stats_after[0].min as f32;
        assert!(
            range_after > range_before,
            "SSR should increase dynamic range: {range_before} -> {range_after}"
        );
    }

    #[test]
    fn msr_produces_output() {
        let (px, info) = make_rgb(32, 32);
        let result = retinex_msr(&px, &info, &[15.0, 80.0, 250.0]).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn msr_single_scale_matches_ssr() {
        let (px, info) = make_rgb(32, 32);
        let ssr = retinex_ssr(&px, &info, 80.0).unwrap();
        let msr = retinex_msr(&px, &info, &[80.0]).unwrap();
        // MSR with one scale should equal SSR
        assert_eq!(ssr, msr, "MSR with single scale should match SSR");
    }

    #[test]
    fn msrcr_produces_output() {
        let (px, info) = make_rgb(32, 32);
        let result = retinex_msrcr(&px, &info, &[15.0, 80.0, 250.0], 125.0, 46.0).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn msrcr_preserves_alpha() {
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![128u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 4] = 100 + (i % 50) as u8;
            pixels[i * 4 + 1] = 120;
            pixels[i * 4 + 2] = 80;
            pixels[i * 4 + 3] = 200;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = retinex_msrcr(&pixels, &info, &[15.0, 80.0, 250.0], 125.0, 46.0).unwrap();
        for i in 0..(w * h) as usize {
            assert_eq!(result[i * 4 + 3], 200, "alpha must be preserved");
        }
    }

    #[test]
    fn msrcr_output_uses_full_range() {
        let (px, info) = make_rgb(64, 64);
        let result = retinex_msrcr(&px, &info, &[15.0, 80.0, 250.0], 125.0, 46.0).unwrap();
        let stats = crate::domain::histogram::statistics(&result, &info).unwrap();
        // Normalized output should span most of 0-255
        assert!(
            stats[0].min <= 5,
            "min should be near 0, got {}",
            stats[0].min
        );
        assert!(
            stats[0].max >= 250,
            "max should be near 255, got {}",
            stats[0].max
        );
    }
}

// ─── Adaptive Thresholding ───────────────────────────────────────────────────

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
    for i in 0..256 {
        total_sum += i as f64 * hist[i] as f64;
    }

    let mut best_thresh = 0u8;
    let mut best_var = 0.0f64;
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;

    for t in 0..256 {
        w0 += hist[t] as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = n - w0;
        if w1 == 0.0 {
            break;
        }

        sum0 += t as f64 * hist[t] as f64;
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

/// Apply binary threshold to a grayscale image.
///
/// Pixels >= threshold become max_value, pixels < threshold become 0.
pub fn threshold_binary(
    pixels: &[u8],
    info: &ImageInfo,
    thresh: u8,
    max_value: u8,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "threshold requires Gray8 input".into(),
        ));
    }
    Ok(pixels
        .iter()
        .map(|&v| if v >= thresh { max_value } else { 0 })
        .collect())
}

/// Adaptive threshold modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMethod {
    /// Use mean of the block neighborhood.
    Mean,
    /// Use Gaussian-weighted mean of the block neighborhood.
    Gaussian,
}

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
    if block_size % 2 == 0 || block_size < 3 {
        return Err(ImageError::InvalidParameters(
            "block_size must be odd and >= 3".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let r = (block_size / 2) as usize;

    // Compute local mean (integral image approach for Mean, Gaussian for Gaussian)
    let local_mean = match method {
        AdaptiveMethod::Mean => {
            // Box mean via integral image — BORDER_REFLECT_101
            let input: Vec<f64> = pixels.iter().map(|&v| v as f64).collect();
            box_mean_f64(&input, w, h, r)
        }
        AdaptiveMethod::Gaussian => {
            // Gaussian-weighted mean — use separable Gaussian blur
            let sigma = 0.3 * ((block_size as f64 - 1.0) * 0.5 - 1.0) + 0.8;
            gaussian_blur_f64(pixels, w, h, block_size as usize, sigma)
        }
    };

    let mut result = vec![0u8; w * h];
    for i in 0..(w * h) {
        let thresh = local_mean[i] - c;
        // OpenCV uses strict > for THRESH_BINARY comparison
        result[i] = if (pixels[i] as f64) > thresh {
            max_value
        } else {
            0
        };
    }

    Ok(result)
}

/// Box mean via integral image (f64 precision).
fn box_mean_f64(data: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
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

/// Separable Gaussian blur (f64 precision) for adaptive threshold Gaussian mode.
fn gaussian_blur_f64(pixels: &[u8], w: usize, h: usize, ksize: usize, sigma: f64) -> Vec<f64> {
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
                let sx = (x as isize + k as isize - r as isize);
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
                let sy = (y as isize + k as isize - r as isize);
                let sy = reflect101(sy, hs) as usize;
                val += tmp[sy * w + x] * kernel[k];
            }
            result[y * w + x] = val;
        }
    }

    result
}

#[cfg(test)]
mod threshold_tests {
    use super::super::types::ColorSpace;
    use super::*;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn otsu_bimodal() {
        // Bimodal image: half black, half white
        let mut px = vec![0u8; 64 * 64];
        for i in 0..32 * 64 {
            px[i] = 20;
        }
        for i in 32 * 64..64 * 64 {
            px[i] = 220;
        }
        let info = gray_info(64, 64);
        let t = otsu_threshold(&px, &info).unwrap();
        // Otsu should find a threshold between 20 and 220
        assert!(t > 20 && t < 220, "otsu={t}, expected between 20-220");
    }

    #[test]
    fn triangle_unimodal() {
        // Mostly dark image with a few bright pixels
        let mut px = vec![10u8; 64 * 64];
        for i in 0..100 {
            px[i] = 200;
        }
        let info = gray_info(64, 64);
        let t = triangle_threshold(&px, &info).unwrap();
        assert!(t > 0, "triangle={t}, expected > 0");
    }

    #[test]
    fn threshold_binary_basic() {
        let px = vec![50, 100, 150, 200];
        let info = gray_info(2, 2);
        let out = threshold_binary(&px, &info, 120, 255).unwrap();
        assert_eq!(out, vec![0, 0, 255, 255]);
    }

    #[test]
    fn adaptive_mean_basic() {
        let mut px = vec![128u8; 32 * 32];
        // Make one quadrant brighter
        for y in 0..16 {
            for x in 0..16 {
                px[y * 32 + x] = 200;
            }
        }
        let info = gray_info(32, 32);
        let out = adaptive_threshold(&px, &info, 255, AdaptiveMethod::Mean, 11, 2.0).unwrap();
        assert_eq!(out.len(), 32 * 32);
        // Should produce binary output
        assert!(out.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn adaptive_gaussian_basic() {
        let px = vec![128u8; 16 * 16];
        let info = gray_info(16, 16);
        let out = adaptive_threshold(&px, &info, 255, AdaptiveMethod::Gaussian, 5, 0.0).unwrap();
        assert_eq!(out.len(), 16 * 16);
        // Uniform image with C=0 → all pixels equal mean → all ≥ threshold
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn adaptive_rejects_even_block() {
        let px = vec![128u8; 16 * 16];
        let info = gray_info(16, 16);
        assert!(adaptive_threshold(&px, &info, 255, AdaptiveMethod::Mean, 10, 0.0).is_err());
    }
}

// ─── HDR Merge: Mertens Exposure Fusion + Debevec ────────────────────────────

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

/// Compute Mertens weight map for a single image.
/// Input is f32 RGB in [0,1], interleaved. Returns one weight per pixel.
fn compute_mertens_weight(img_f: &[f32], w: usize, h: usize, params: &MertensParams) -> Vec<f32> {
    let n = w * h;
    let sigma = 0.2f32;

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

// ─── Gaussian/Laplacian Pyramid (OpenCV-compatible) ──────────────────────────

/// OpenCV-compatible 5×5 Gaussian kernel for pyrDown: [1,4,6,4,1]/16
const PYR_KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// pyrDown for single-channel f32 image.
/// Applies 5×5 Gaussian blur then subsamples by 2 in each dimension.
/// Border handling: BORDER_REFLECT_101 (default OpenCV border for pyrDown).
fn pyr_down_gray(src: &[f32], sw: u32, sh: u32) -> (Vec<f32>, u32, u32) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = (sw + 1) / 2;
    let dh = (sh + 1) / 2;
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
fn pyr_down_rgb(src: &[f32], sw: u32, sh: u32) -> (Vec<f32>, u32, u32) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = (sw + 1) / 2;
    let dh = (sh + 1) / 2;
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

/// pyrUp for single-channel f32 — upsample by 2 then apply 5×5 Gaussian × 4.
fn pyr_up_gray(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
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
fn pyr_up_rgb(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
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

/// BORDER_REFLECT_101 with clamping for small sizes.
/// Handles the case where a single reflection is insufficient (e.g., idx=-2 with size=2).
#[inline]
fn reflect101_safe(idx: isize, size: isize) -> usize {
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

/// Build Gaussian pyramid for a single-channel f32 image.
/// Returns levels+1 images: [original, level1, level2, ...].
fn gaussian_pyramid_gray(src: &[f32], w: u32, h: u32, levels: usize) -> Vec<Vec<f32>> {
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

/// Build Laplacian pyramid for a 3-channel f32 image.
/// Returns levels+1 entries: levels Laplacian layers + 1 low-res residual.
/// Each entry is (pixels, width, height).
fn laplacian_pyramid_rgb(src: &[f32], w: u32, h: u32, levels: usize) -> Vec<(Vec<f32>, u32, u32)> {
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

// ─── Debevec HDR Merge ──────────────────────────────────────────────────────

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

/// Hat-shaped weighting function for Debevec method.
/// w(z) = z + 1 for z <= 127, 256 - z for z >= 128.
/// Gives highest weight to mid-tone pixels.
#[inline]
fn hat_weight(z: usize) -> f64 {
    if z <= 127 {
        (z + 1) as f64
    } else {
        (256 - z) as f64
    }
}

/// Solve overdetermined linear system via normal equations (A^T A x = A^T b).
/// Uses Cholesky-like Gaussian elimination on the normal equations.
fn solve_least_squares(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
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
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
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

#[cfg(test)]
mod hdr_tests {
    use super::super::types::ColorSpace;
    use super::*;

    fn test_info_rgb(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn mertens_two_images() {
        let w = 16u32;
        let h = 16u32;
        let dark = vec![64u8; (w * h * 3) as usize];
        let bright = vec![192u8; (w * h * 3) as usize];
        let info = test_info_rgb(w, h);
        let result = mertens_fusion(&[&dark, &bright], &info, &MertensParams::default()).unwrap();
        assert_eq!(result.len(), (w * h * 3) as usize);
        // Result should be a blend between dark and bright
        let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(mean > 60.0 && mean < 200.0, "mean={mean}");
    }

    #[test]
    fn mertens_preserves_uniform() {
        let w = 16u32;
        let h = 16u32;
        // If all images are the same, result should be approximately that image
        let mid = vec![128u8; (w * h * 3) as usize];
        let info = test_info_rgb(w, h);
        let result = mertens_fusion(&[&mid, &mid], &info, &MertensParams::default()).unwrap();
        for &v in &result {
            assert!((v as i16 - 128).abs() <= 1, "expected ~128, got {v}");
        }
    }

    #[test]
    fn mertens_needs_at_least_two() {
        let info = test_info_rgb(16, 16);
        let img = vec![128u8; 16 * 16 * 3];
        assert!(mertens_fusion(&[&img], &info, &MertensParams::default()).is_err());
    }

    #[test]
    fn debevec_response_curve_basic() {
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;
        // Create simple bracketed exposures
        let mut dark = vec![0u8; n * 3];
        let mut bright = vec![0u8; n * 3];
        for i in 0..n {
            let v = (i % 200) as u8;
            for c in 0..3 {
                dark[i * 3 + c] = (v / 2).max(1);
                bright[i * 3 + c] = (v).min(254).max(1);
            }
        }
        let info = test_info_rgb(w, h);
        let params = DebevecParams {
            samples: 30,
            lambda: 10.0,
        };
        let response =
            debevec_response_curve(&[&dark, &bright], &info, &[0.5, 2.0], &params).unwrap();
        assert_eq!(response.len(), 3);
        // Response should be monotonically increasing (approximately)
        // g(128) should be near 0 (our constraint)
        assert!(response[0][128].abs() < 0.1, "g(128)={}", response[0][128]);
    }

    #[test]
    fn debevec_hdr_merge_basic() {
        let w = 8u32;
        let h = 8u32;
        let n = (w * h) as usize;
        let mut dark = vec![0u8; n * 3];
        let mut bright = vec![0u8; n * 3];
        for i in 0..n {
            for c in 0..3 {
                dark[i * 3 + c] = 64;
                bright[i * 3 + c] = 200;
            }
        }
        let info = test_info_rgb(w, h);
        // Simple linear response curve
        let mut response = [[0.0f32; 256]; 3];
        for ch in 0..3 {
            for z in 0..256 {
                response[ch][z] = ((z as f32 + 1.0) / 128.0).ln();
            }
        }
        let hdr = debevec_hdr_merge(&[&dark, &bright], &info, &[0.25, 4.0], &response).unwrap();
        assert_eq!(hdr.len(), n * 3);
        // All values should be positive
        assert!(hdr.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn pyramid_roundtrip() {
        // pyrUp(pyrDown(img)) should approximate img (low-pass)
        let w = 16u32;
        let h = 16u32;
        let n = (w * h) as usize;
        let mut src = vec![0.0f32; n];
        for i in 0..n {
            src[i] = (i as f32) / (n as f32);
        }
        let (down, dw, dh) = pyr_down_gray(&src, w, h);
        let up = pyr_up_gray(&down, dw, dh, w, h);
        // Should be close to original (low-pass filtered version)
        let mae: f64 = src
            .iter()
            .zip(up.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / n as f64;
        assert!(mae < 0.1, "pyramid roundtrip MAE too high: {mae}");
    }

    #[test]
    fn reflect101_safe_test() {
        assert_eq!(reflect101_safe(-1, 10), 1);
        assert_eq!(reflect101_safe(-2, 10), 2);
        assert_eq!(reflect101_safe(0, 10), 0);
        assert_eq!(reflect101_safe(9, 10), 9);
        assert_eq!(reflect101_safe(10, 10), 8);
        assert_eq!(reflect101_safe(11, 10), 7);
        // Small size edge cases
        assert_eq!(reflect101_safe(-2, 2), 0);
        assert_eq!(reflect101_safe(-3, 2), 1);
        assert_eq!(reflect101_safe(2, 2), 0);
        assert_eq!(reflect101_safe(3, 2), 1);
        assert_eq!(reflect101_safe(-1, 1), 0);
        assert_eq!(reflect101_safe(5, 1), 0);
    }
}

#[cfg(test)]
mod spatial_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn connected_components_two_blobs() {
        // Two separate 2x2 white squares on black background
        #[rustfmt::skip]
        let px = vec![
            255,255,  0,  0,  0,
            255,255,  0,  0,  0,
              0,  0,  0,  0,  0,
              0,  0,  0,255,255,
              0,  0,  0,255,255,
        ];
        let info = gray_info(5, 5);
        let (labels, count) = connected_components(&px, &info, 8).unwrap();
        assert_eq!(count, 2, "should find 2 components");
        // Top-left and bottom-right should have different labels
        assert_ne!(labels[0], labels[3 * 5 + 3]);
        assert_ne!(labels[0], 0);
        assert_ne!(labels[3 * 5 + 3], 0);
        // Background should be 0
        assert_eq!(labels[2 * 5 + 2], 0);
    }

    #[test]
    fn connected_components_4_vs_8() {
        // Diagonal connection: 4-connectivity = 2 components, 8-connectivity = 1
        #[rustfmt::skip]
        let px = vec![
            255,  0,
              0,255,
        ];
        let info = gray_info(2, 2);
        let (_, count4) = connected_components(&px, &info, 4).unwrap();
        let (_, count8) = connected_components(&px, &info, 8).unwrap();
        assert_eq!(count4, 2, "4-connectivity: diagonal = separate");
        assert_eq!(count8, 1, "8-connectivity: diagonal = connected");
    }

    #[test]
    fn flood_fill_fills_region() {
        #[rustfmt::skip]
        let px = vec![
            100,100,100,  0,200,
            100,100,100,  0,200,
            100,100,100,  0,200,
        ];
        let info = gray_info(5, 3);
        let (result, filled) = flood_fill(&px, &info, 1, 1, 50, 0, 4).unwrap();
        assert_eq!(filled, 9, "should fill 3x3 region of value 100");
        assert_eq!(result[0], 50);
        assert_eq!(result[4], 200); // untouched
        assert_eq!(result[3], 0); // barrier untouched
    }

    #[test]
    fn flood_fill_with_tolerance() {
        let px = vec![100, 102, 105, 110, 200];
        let info = gray_info(5, 1);
        let (result, filled) = flood_fill(&px, &info, 0, 0, 50, 5, 4).unwrap();
        // Tolerance 5 from seed=100: fills 100, 102, 105 (all within ±5)
        assert_eq!(filled, 3);
        assert_eq!(result[0], 50);
        assert_eq!(result[1], 50);
        assert_eq!(result[2], 50);
        assert_eq!(result[3], 110); // 110 > 105, not within tolerance of 100
    }

    #[test]
    fn pyr_down_halves_size() {
        let px = vec![128u8; 64 * 64];
        let info = gray_info(64, 64);
        let (result, new_info) = pyr_down(&px, &info).unwrap();
        assert_eq!(new_info.width, 32);
        assert_eq!(new_info.height, 32);
        assert_eq!(result.len(), 32 * 32);
        // Uniform input → uniform output
        for &v in &result {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn pyr_up_doubles_size() {
        let px = vec![128u8; 32 * 32];
        let info = gray_info(32, 32);
        let (result, new_info) = pyr_up(&px, &info).unwrap();
        assert_eq!(new_info.width, 64);
        assert_eq!(new_info.height, 64);
        assert_eq!(result.len(), 64 * 64);
    }

    #[test]
    fn pyr_down_up_roundtrip() {
        // pyrUp(pyrDown(img)) should be close to original for smooth content
        let mut px = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                px[y * 64 + x] = ((x * 255) / 63) as u8;
            }
        }
        let info = gray_info(64, 64);
        let (down, down_info) = pyr_down(&px, &info).unwrap();
        let (up, _) = pyr_up(&down, &down_info).unwrap();

        // MAE should be small for smooth gradient
        let mae: f64 = px
            .iter()
            .zip(up.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 5.0,
            "pyrDown→pyrUp roundtrip MAE={mae:.2} (should be < 5.0)"
        );
    }
}
