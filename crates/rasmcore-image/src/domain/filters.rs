//! Image filters — SIMD-optimized where possible.
//!
//! Operations work directly on raw pixel buffers without DynamicImage conversion.
//! Blur uses libblur (SIMD on x86/ARM/WASM). Point ops are written as simple
//! loops that LLVM auto-vectorizes to SIMD128 when compiled with +simd128.

use super::error::ImageError;
use super::types::{DecodedImage, ImageInfo, PixelFormat};

fn validate_format(format: PixelFormat) -> Result<(), ImageError> {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => Ok(()),
        other => Err(ImageError::UnsupportedFormat(format!(
            "filter on {other:?} not supported"
        ))),
    }
}

/// Apply gaussian blur using libblur (SIMD-optimized).
///
/// Uses separable gaussian convolution with SIMD acceleration on
/// x86 (SSE/AVX), ARM (NEON), and WASM (SIMD128).
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
pub fn sharpen(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    // Blur with a small radius for the unsharp mask
    let blurred = blur(pixels, info, 1.0)?;

    let mut result = Vec::with_capacity(pixels.len());
    for i in 0..pixels.len() {
        let orig = pixels[i] as f32;
        let blur_val = blurred[i] as f32;
        let sharp = orig + amount * (orig - blur_val);
        result.push(sharp.clamp(0.0, 255.0) as u8);
    }
    Ok(result)
}

/// Adjust brightness (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.
pub fn brightness(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "brightness must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
    let lut = super::point_ops::build_lut(&super::point_ops::PointOp::Brightness(amount));
    super::point_ops::apply_lut(pixels, info, &lut)
}

/// Adjust contrast (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.
pub fn contrast(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "contrast must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
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

// ─── HSV/HSL Color Conversion ────────────────────────────────────────────

/// Convert RGB (0-255 each) to HSV (H: 0-360, S: 0-1, V: 0-1).
#[inline]
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let h = if delta == 0.0 {
        0.0
    } else if max == rf {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
}

/// Convert HSV (H: 0-360, S: 0-1, V: 0-1) to RGB (0-255 each).
#[inline]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
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
    let m = v - c;
    (
        ((r1 + m) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
    )
}

/// Convert RGB to HSL (H: 0-360, S: 0-1, L: 0-1).
#[inline]
fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
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

    let h = if max == rf {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, l)
}

/// Convert HSL to RGB.
#[inline]
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    if s == 0.0 {
        let v = (l * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        return (v, v, v);
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
    (
        ((r1 + m) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
    )
}

// ─── Color Adjustment Functions ──────────────────────────────────────────

/// Rotate hue by `degrees` (0-360). Works on RGB8 and RGBA8 images.
pub fn hue_rotate(pixels: &[u8], info: &ImageInfo, degrees: f32) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if info.format == PixelFormat::Gray8 {
        return Ok(pixels.to_vec()); // no hue in grayscale
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let (h, s, v) = rgb_to_hsv(chunk[0], chunk[1], chunk[2]);
        let new_h = (h + degrees) % 360.0;
        let new_h = if new_h < 0.0 { new_h + 360.0 } else { new_h };
        let (r, g, b) = hsv_to_rgb(new_h, s, v);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
    Ok(result)
}

/// Adjust saturation by `factor` (0=grayscale, 1=unchanged, 2=double).
/// Uses HSL conversion.
pub fn saturate(pixels: &[u8], info: &ImageInfo, factor: f32) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if info.format == PixelFormat::Gray8 {
        return Ok(pixels.to_vec());
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let (h, s, l) = rgb_to_hsl(chunk[0], chunk[1], chunk[2]);
        let new_s = (s * factor).clamp(0.0, 1.0);
        let (r, g, b) = hsl_to_rgb(h, new_s, l);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
    Ok(result)
}

/// Apply sepia tone with given `intensity` (0=none, 1=full sepia).
pub fn sepia(pixels: &[u8], info: &ImageInfo, intensity: f32) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    let intensity = intensity.clamp(0.0, 1.0);
    if info.format == PixelFormat::Gray8 {
        return Ok(pixels.to_vec());
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        // Standard sepia matrix
        let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(255.0);
        let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(255.0);
        let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(255.0);
        chunk[0] = (r + (sr - r) * intensity).clamp(0.0, 255.0) as u8;
        chunk[1] = (g + (sg - g) * intensity).clamp(0.0, 255.0) as u8;
        chunk[2] = (b + (sb - b) * intensity).clamp(0.0, 255.0) as u8;
    }
    Ok(result)
}

/// Tint image toward `target_color` (RGB) by `amount` (0=none, 1=full tint).
pub fn colorize(
    pixels: &[u8],
    info: &ImageInfo,
    target: [u8; 3],
    amount: f32,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    let amount = amount.clamp(0.0, 1.0);
    if info.format == PixelFormat::Gray8 {
        return Ok(pixels.to_vec());
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let luma = 0.2126 * chunk[0] as f32 + 0.7152 * chunk[1] as f32 + 0.0722 * chunk[2] as f32;
        let tinted_r = luma * (target[0] as f32 / 255.0);
        let tinted_g = luma * (target[1] as f32 / 255.0);
        let tinted_b = luma * (target[2] as f32 / 255.0);
        chunk[0] =
            (chunk[0] as f32 + (tinted_r - chunk[0] as f32) * amount).clamp(0.0, 255.0) as u8;
        chunk[1] =
            (chunk[1] as f32 + (tinted_g - chunk[1] as f32) * amount).clamp(0.0, 255.0) as u8;
        chunk[2] =
            (chunk[2] as f32 + (tinted_b - chunk[2] as f32) * amount).clamp(0.0, 255.0) as u8;
    }
    Ok(result)
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
/// `kernel` is row-major, `kw`/`kh` must be odd. Each output pixel is
/// `sum(kernel[i,j] * input[x+j-r, y+i-r]) / divisor`, clamped to 0-255.
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

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;
    let rw = kw / 2;
    let rh = kh / 2;
    let inv_div = 1.0 / divisor;

    let mut out = vec![0u8; pixels.len()];

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sy = reflect(y as i32 + ky as i32 - rh as i32, h);
                        let sx = reflect(x as i32 + kx as i32 - rw as i32, w);
                        sum += kernel[ky * kw + kx] * pixels[(sy * w + sx) * channels + c] as f32;
                    }
                }
                out[(y * w + x) * channels + c] = (sum * inv_div).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
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
/// Uses sorting for small windows (radius <= 2), histogram-based for larger.
pub fn median(pixels: &[u8], info: &ImageInfo, radius: u32) -> Result<Vec<u8>, ImageError> {
    if radius == 0 {
        return Ok(pixels.to_vec());
    }
    validate_format(info.format)?;

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;
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

/// Sobel edge detection — produces grayscale gradient magnitude image.
///
/// Applies 3x3 Sobel operators for horizontal and vertical gradients,
/// then computes magnitude = sqrt(Gx^2 + Gy^2), clamped to 0-255.
pub fn sobel(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    let w = info.width as usize;
    let h = info.height as usize;
    let channels = crate::domain::pipeline::graph::bytes_per_pixel(info.format) as usize;

    // Convert to grayscale if needed
    let gray = to_grayscale(pixels, channels);
    let mut out = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;

            for ky in 0..3i32 {
                for kx in 0..3i32 {
                    let sy = reflect(y as i32 + ky - 1, h);
                    let sx = reflect(x as i32 + kx - 1, w);
                    let val = gray[sy * w + sx] as f32;
                    // Sobel X: [[-1,0,1],[-2,0,2],[-1,0,1]]
                    let sx_w = match kx {
                        0 => -1.0,
                        2 => 1.0,
                        _ => 0.0,
                    } * match ky {
                        1 => 2.0,
                        _ => 1.0,
                    };
                    // Sobel Y: [[-1,-2,-1],[0,0,0],[1,2,1]]
                    let sy_w = match ky {
                        0 => -1.0,
                        2 => 1.0,
                        _ => 0.0,
                    } * match kx {
                        1 => 2.0,
                        _ => 1.0,
                    };
                    gx += sx_w * val;
                    gy += sy_w * val;
                }
            }
            out[y * w + x] = (gx * gx + gy * gy).sqrt().min(255.0) as u8;
        }
    }
    Ok(out)
}

/// Canny edge detection — produces binary edge map (0 or 255).
///
/// Steps: 1) Gaussian blur, 2) Sobel gradient + direction,
/// 3) Non-maximum suppression, 4) Hysteresis thresholding.
pub fn canny(
    pixels: &[u8],
    info: &ImageInfo,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

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

    // Step 3: Sobel gradient magnitude and direction
    let mut magnitude = vec![0.0f32; w * h];
    let mut direction = vec![0u8; w * h]; // 0=horizontal, 1=diagonal45, 2=vertical, 3=diagonal135

    for y in 0..h {
        for x in 0..w {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for ky in 0..3i32 {
                for kx in 0..3i32 {
                    let sy = reflect(y as i32 + ky - 1, h);
                    let sx = reflect(x as i32 + kx - 1, w);
                    let val = blurred[sy * w + sx] as f32;
                    let sx_w = match kx {
                        0 => -1.0,
                        2 => 1.0,
                        _ => 0.0,
                    } * match ky {
                        1 => 2.0,
                        _ => 1.0,
                    };
                    let sy_w = match ky {
                        0 => -1.0,
                        2 => 1.0,
                        _ => 0.0,
                    } * match kx {
                        1 => 2.0,
                        _ => 1.0,
                    };
                    gx += sx_w * val;
                    gy += sy_w * val;
                }
            }
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
fn to_grayscale(pixels: &[u8], channels: usize) -> Vec<u8> {
    if channels == 1 {
        return pixels.to_vec();
    }
    let pixel_count = pixels.len() / channels;
    let mut gray = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        let r = pixels[i * channels] as f32;
        let g = pixels[i * channels + 1] as f32;
        let b = pixels[i * channels + 2] as f32;
        gray.push((0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8);
    }
    gray
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
    fn hsv_roundtrip_full_range() {
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let (h, s, v) = rgb_to_hsv(r as u8, g as u8, b as u8);
                    let (r2, g2, b2) = hsv_to_rgb(h, s, v);
                    assert!(
                        (r as i32 - r2 as i32).abs() <= 1
                            && (g as i32 - g2 as i32).abs() <= 1
                            && (b as i32 - b2 as i32).abs() <= 1,
                        "HSV roundtrip: ({r},{g},{b}) -> ({h:.1},{s:.3},{v:.3}) -> ({r2},{g2},{b2})"
                    );
                }
            }
        }
    }

    #[test]
    fn hsl_roundtrip_full_range() {
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let (h, s, l) = rgb_to_hsl(r as u8, g as u8, b as u8);
                    let (r2, g2, b2) = hsl_to_rgb(h, s, l);
                    assert!(
                        (r as i32 - r2 as i32).abs() <= 1
                            && (g as i32 - g2 as i32).abs() <= 1
                            && (b as i32 - b2 as i32).abs() <= 1,
                        "HSL roundtrip: ({r},{g},{b}) -> ({h:.1},{s:.3},{l:.3}) -> ({r2},{g2},{b2})"
                    );
                }
            }
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
}
