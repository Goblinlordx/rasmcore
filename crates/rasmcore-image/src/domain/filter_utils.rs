//! Shared utilities for filter implementations.
//!
//! These functions are used by both built-in and third-party filter crates.
//! They handle format validation, 16-bit conversion, and edge padding.

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

/// Validate that a pixel format is supported for filtering (RGB8, RGBA8, Gray8).
pub fn validate_format(format: PixelFormat) -> Result<(), ImageError> {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => Ok(()),
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

/// Get number of channels for a pixel format.
pub fn channels(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 | PixelFormat::Gray16 => 1,
        PixelFormat::Rgb8 | PixelFormat::Rgb16 => 3,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => 4,
        PixelFormat::Cmyk8 => 4,
        PixelFormat::Cmyka8 => 5,
        _ => 4, // conservative default
    }
}

/// Convert LE byte pairs to u16 values.
pub fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Convert u16 values to LE byte pairs.
pub fn u16_to_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Process a CMYK image by converting to RGB, applying an RGB filter, then
/// converting back to CMYK. Used as a boundary conversion for the filter pipeline.
pub fn process_cmyk_via_rgb<F>(pixels: &[u8], info: &ImageInfo, f: F) -> Result<Vec<u8>, ImageError>
where
    F: FnOnce(&[u8], &ImageInfo) -> Result<Vec<u8>, ImageError>,
{
    use super::color;

    let (rgb_pixels, rgb_info) = color::cmyk_to_rgb(pixels, info)?;
    let result_rgb = f(&rgb_pixels, &rgb_info)?;

    // Convert result back to CMYK
    let result_rgb_info = ImageInfo {
        format: rgb_info.format,
        ..*info
    };
    let (cmyk_pixels, _) = color::rgb_to_cmyk(&result_rgb, &result_rgb_info)?;
    Ok(cmyk_pixels)
}

/// Check if a pixel format is CMYK.
pub fn is_cmyk(format: PixelFormat) -> bool {
    matches!(format, PixelFormat::Cmyk8 | PixelFormat::Cmyka8)
}

/// Convert 16-bit pixel buffer to f32 normalized [0.0, 1.0] per sample.
pub fn u16_pixels_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes_to_u16(bytes)
        .into_iter()
        .map(|v| v as f32 / 65535.0)
        .collect()
}

/// Convert f32 [0.0, 1.0] samples back to 16-bit LE byte pairs.
pub fn f32_to_u16_pixels(values: &[f32]) -> Vec<u8> {
    let u16s: Vec<u16> = values
        .iter()
        .map(|&v| (v * 65535.0).round().clamp(0.0, 65535.0) as u16)
        .collect();
    u16_to_bytes(&u16s)
}

/// Process a 16-bit image by downsampling to 8-bit, applying an 8-bit function,
/// then upsampling back to 16-bit. Used as a fallback when a filter doesn't
/// have a native 16-bit path.
pub fn process_via_8bit<F>(pixels: &[u8], info: &ImageInfo, f: F) -> Result<Vec<u8>, ImageError>
where
    F: FnOnce(&[u8], &ImageInfo) -> Result<Vec<u8>, ImageError>,
{
    let _channels = channels(info.format);
    let bpc = 2; // bytes per channel for 16-bit

    // Downscale to 8-bit: (u16 + 128) / 257
    let pixels_8: Vec<u8> = pixels
        .chunks_exact(bpc)
        .map(|c| {
            let v = u16::from_le_bytes([c[0], c[1]]);
            ((v as u32 + 128) / 257) as u8
        })
        .collect();

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

    // Upscale back to 16-bit: value * 257
    let result_16: Vec<u16> = result_8.iter().map(|&v| v as u16 * 257).collect();
    Ok(u16_to_bytes(&result_16))
}

/// Pad an image with reflected borders (BORDER_REFLECT_101).
pub fn pad_reflect(pixels: &[u8], w: usize, h: usize, channels: usize, pad: usize) -> Vec<u8> {
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;
    let mut padded = vec![0u8; pw * ph * channels];

    for py in 0..ph {
        let sy = reflect(py as i32 - pad as i32, h);
        for px in 0..pw {
            let sx = reflect(px as i32 - pad as i32, w);
            let src = (sy * w + sx) * channels;
            let dst = (py * pw + px) * channels;
            padded[dst..dst + channels].copy_from_slice(&pixels[src..src + channels]);
        }
    }
    padded
}

/// Reflect index into valid range (BORDER_REFLECT_101 mode).
pub fn reflect(v: i32, size: usize) -> usize {
    let s = size as i32;
    if v < 0 {
        (-v).min(s - 1) as usize
    } else if v >= s {
        (2 * s - v - 2).max(0) as usize
    } else {
        v as usize
    }
}
