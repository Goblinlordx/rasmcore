//! f32 pipeline quantization utilities.
//!
//! Shared functions for converting from Rgba32f (the canonical pipeline format)
//! down to any output format. Encoders use these — no inline bit-depth
//! conversion in encoder code.
//!
//! All functions take `&[u8]` (Rgba32f as raw bytes, 16 bytes/pixel) and produce
//! `Vec<u8>` in the target format.

use crate::domain::types::PixelFormat;

/// Quantize Rgba32f pixels to any target format for encoding.
///
/// Returns (quantized_pixels, target_format). If source is already the target
/// format, returns a clone. If source is not Rgba32f, returns as-is (no-op for
/// legacy paths during migration).
pub fn quantize_for_encoder(
    pixels: &[u8],
    src_format: PixelFormat,
    target_format: PixelFormat,
) -> (Vec<u8>, PixelFormat) {
    if src_format == target_format {
        return (pixels.to_vec(), target_format);
    }
    if src_format != PixelFormat::Rgba32f {
        // Not in f32 pipeline — pass through (legacy path)
        return (pixels.to_vec(), src_format);
    }
    match target_format {
        PixelFormat::Rgba8 => (rgba32f_to_rgba8(pixels), target_format),
        PixelFormat::Rgb8 => (rgba32f_to_rgb8(pixels), target_format),
        PixelFormat::Gray8 => (rgba32f_to_gray8(pixels), target_format),
        PixelFormat::Rgba16 => (rgba32f_to_rgba16(pixels), target_format),
        PixelFormat::Rgb16 => (rgba32f_to_rgb16(pixels), target_format),
        PixelFormat::Gray16 => (rgba32f_to_gray16(pixels), target_format),
        PixelFormat::Rgba16f => (rgba32f_to_rgba16f(pixels), target_format),
        PixelFormat::Rgb16f => (rgba32f_to_rgb16f(pixels), target_format),
        PixelFormat::Gray16f => (rgba32f_to_gray16f(pixels), target_format),
        PixelFormat::Rgb32f => (rgba32f_to_rgb32f(pixels), target_format),
        PixelFormat::Gray32f => (rgba32f_to_gray32f(pixels), target_format),
        PixelFormat::Rgba32f => (pixels.to_vec(), target_format),
        _ => (pixels.to_vec(), src_format), // unsupported — pass through
    }
}

/// Determine the best output format for an encoder given its format name.
/// Most encoders want Rgba8 or Rgb8. HDR/EXR want f32.
pub fn encoder_target_format(format: &str) -> PixelFormat {
    match format {
        "hdr" | "exr" => PixelFormat::Rgb32f,
        "fits" | "fit" => PixelFormat::Gray32f,
        "png" | "tiff" | "tif" => PixelFormat::Rgba8, // PNG/TIFF can do 16-bit but default to 8
        _ => PixelFormat::Rgba8, // JPEG, WebP, AVIF, BMP, GIF, QOI, ICO, DDS, etc.
    }
}

// ─── Individual quantization functions ─────────────────────────────────────

#[inline]
fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

/// BT.709 luma from linear RGB.
#[inline]
fn luma_f32(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

pub fn rgba32f_to_rgba8(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 4);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..4 {
            let v = read_f32(chunk, i * 4);
            out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

pub fn rgba32f_to_rgb8(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 3);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..3 {
            let v = read_f32(chunk, i * 4);
            out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

pub fn rgba32f_to_gray8(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count);
    for chunk in pixels.chunks_exact(16) {
        let r = read_f32(chunk, 0);
        let g = read_f32(chunk, 4);
        let b = read_f32(chunk, 8);
        out.push((luma_f32(r, g, b) * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
    }
    out
}

pub fn rgba32f_to_rgba16(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 8);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..4 {
            let v = read_f32(chunk, i * 4);
            let u = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    out
}

pub fn rgba32f_to_rgb16(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 6);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..3 {
            let v = read_f32(chunk, i * 4);
            let u = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    out
}

pub fn rgba32f_to_gray16(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 2);
    for chunk in pixels.chunks_exact(16) {
        let r = read_f32(chunk, 0);
        let g = read_f32(chunk, 4);
        let b = read_f32(chunk, 8);
        let u = (luma_f32(r, g, b) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

pub fn rgba32f_to_rgba16f(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 8);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..4 {
            let v = read_f32(chunk, i * 4);
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    out
}

pub fn rgba32f_to_rgb16f(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 6);
    for chunk in pixels.chunks_exact(16) {
        for i in 0..3 {
            let v = read_f32(chunk, i * 4);
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    out
}

pub fn rgba32f_to_gray16f(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 2);
    for chunk in pixels.chunks_exact(16) {
        let r = read_f32(chunk, 0);
        let g = read_f32(chunk, 4);
        let b = read_f32(chunk, 8);
        out.extend_from_slice(&half::f16::from_f32(luma_f32(r, g, b)).to_le_bytes());
    }
    out
}

pub fn rgba32f_to_rgb32f(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 12);
    for chunk in pixels.chunks_exact(16) {
        out.extend_from_slice(&chunk[..12]); // R, G, B — drop A
    }
    out
}

pub fn rgba32f_to_gray32f(pixels: &[u8]) -> Vec<u8> {
    let count = pixels.len() / 16;
    let mut out = Vec::with_capacity(count * 4);
    for chunk in pixels.chunks_exact(16) {
        let r = read_f32(chunk, 0);
        let g = read_f32(chunk, 4);
        let b = read_f32(chunk, 8);
        out.extend_from_slice(&luma_f32(r, g, b).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_u8() {
        // 0.0 → 0, 1.0 → 255, 0.5 → 128
        let rgba32f: Vec<u8> = [0.0f32, 0.5, 1.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let rgba8 = rgba32f_to_rgba8(&rgba32f);
        assert_eq!(rgba8, vec![0, 128, 255, 255]);
    }

    #[test]
    fn round_trip_u16() {
        let rgba32f: Vec<u8> = [0.0f32, 0.5, 1.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let rgba16 = rgba32f_to_rgba16(&rgba32f);
        let values: Vec<u16> = rgba16
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(values[0], 0);
        assert_eq!(values[1], 32768); // 0.5 * 65535 + 0.5 = 32768.0
        assert_eq!(values[2], 65535);
        assert_eq!(values[3], 65535);
    }

    #[test]
    fn gray_luma_bt709() {
        // Pure red: luma = 0.2126
        let rgba32f: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let gray8 = rgba32f_to_gray8(&rgba32f);
        assert_eq!(gray8[0], 54); // 0.2126 * 255 + 0.5 = 54.7 → 54
    }
}
