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
}
