//! Pixel Format helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Upstream pixel request function.
pub type UpstreamFn<'a> = dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + 'a;

pub fn validate_format(format: PixelFormat) -> Result<(), ImageError> {
    match format {
        PixelFormat::Rgb8
        | PixelFormat::Rgba8
        | PixelFormat::Gray8
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Gray16
        | PixelFormat::Rgb16f
        | PixelFormat::Rgba16f
        | PixelFormat::Gray16f
        | PixelFormat::Rgb32f
        | PixelFormat::Rgba32f
        | PixelFormat::Gray32f => Ok(()),
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

/// Check if a pixel format is 16-bit float (half precision).
pub fn is_f16(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb16f | PixelFormat::Rgba16f | PixelFormat::Gray16f
    )
}

/// Check if a pixel format is 32-bit float.
pub fn is_f32(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb32f | PixelFormat::Rgba32f | PixelFormat::Gray32f
    )
}

/// Check if a pixel format is any high-precision float (f16 or f32).
pub fn is_float(format: PixelFormat) -> bool {
    is_f16(format) || is_f32(format)
}

/// Number of channels for a pixel format (not bytes — channels).
pub fn channels(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::Gray16f | PixelFormat::Gray32f => 1,
        PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::Rgb16f | PixelFormat::Rgb32f => 3,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::Rgba16f | PixelFormat::Rgba32f => 4,
        _ => 3,
    }
}

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

/// Read f16 samples from a byte buffer (little-endian).
pub fn bytes_to_f16(bytes: &[u8]) -> Vec<half::f16> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Write f16 samples to a byte buffer (little-endian).
pub fn f16_to_bytes(values: &[half::f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Read f32 samples from a byte buffer (little-endian).
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Write f32 samples to a byte buffer (little-endian).
pub fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Convert float (f16 or f32) pixel buffer to 8-bit for processing, then back.
/// Used when an operation only supports 8-bit internally and float input arrives.
pub fn process_via_standard<F>(pixels: &[u8], info: &ImageInfo, f: F) -> Result<Vec<u8>, ImageError>
where
    F: FnOnce(&[u8], &ImageInfo) -> Result<Vec<u8>, ImageError>,
{
    let (info_8, pixels_8) = match info.format {
        PixelFormat::Rgb32f | PixelFormat::Rgba32f | PixelFormat::Gray32f => {
            let samples = bytes_to_f32(pixels);
            let p8: Vec<u8> = samples
                .iter()
                .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
                .collect();
            let fmt = match info.format {
                PixelFormat::Rgb32f => PixelFormat::Rgb8,
                PixelFormat::Rgba32f => PixelFormat::Rgba8,
                _ => PixelFormat::Gray8,
            };
            (ImageInfo { format: fmt, ..*info }, p8)
        }
        PixelFormat::Rgb16f | PixelFormat::Rgba16f | PixelFormat::Gray16f => {
            let samples = bytes_to_f16(pixels);
            let p8: Vec<u8> = samples
                .iter()
                .map(|v| (v.to_f32() * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
                .collect();
            let fmt = match info.format {
                PixelFormat::Rgb16f => PixelFormat::Rgb8,
                PixelFormat::Rgba16f => PixelFormat::Rgba8,
                _ => PixelFormat::Gray8,
            };
            (ImageInfo { format: fmt, ..*info }, p8)
        }
        _ => return f(pixels, info),
    };

    let result_8 = f(&pixels_8, &info_8)?;

    // Convert back to original float format
    match info.format {
        PixelFormat::Rgb32f | PixelFormat::Rgba32f | PixelFormat::Gray32f => {
            let result_f32: Vec<f32> = result_8.iter().map(|&v| v as f32 / 255.0).collect();
            Ok(f32_to_bytes(&result_f32))
        }
        PixelFormat::Rgb16f | PixelFormat::Rgba16f | PixelFormat::Gray16f => {
            let result_f16: Vec<half::f16> = result_8
                .iter()
                .map(|&v| half::f16::from_f32(v as f32 / 255.0))
                .collect();
            Ok(f16_to_bytes(&result_f16))
        }
        _ => unreachable!(),
    }
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

