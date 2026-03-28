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
/// Written as a simple per-byte loop that LLVM auto-vectorizes to SIMD128.
pub fn brightness(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "brightness must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;

    let offset = (amount * 128.0) as i16;
    let mut result = Vec::with_capacity(pixels.len());
    for &p in pixels {
        let v = (p as i16 + offset).clamp(0, 255) as u8;
        result.push(v);
    }
    Ok(result)
}

/// Adjust contrast (-1.0 to 1.0).
///
/// Uses the formula: output = factor * (pixel - 128) + 128
/// where factor = (1.0 + amount) for positive, (1.0 / (1.0 - amount)) adjusted.
pub fn contrast(pixels: &[u8], info: &ImageInfo, amount: f32) -> Result<Vec<u8>, ImageError> {
    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "contrast must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;

    // Build a 256-entry LUT for maximum throughput
    let factor = if amount >= 0.0 {
        1.0 + amount * 2.0
    } else {
        1.0 / (1.0 - amount * 2.0)
    };

    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = factor * (i as f32 - 128.0) + 128.0;
        *entry = v.clamp(0.0, 255.0) as u8;
    }

    let result: Vec<u8> = pixels.iter().map(|&p| lut[p as usize]).collect();
    Ok(result)
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
