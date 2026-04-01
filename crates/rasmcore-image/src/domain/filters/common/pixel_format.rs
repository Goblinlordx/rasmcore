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

