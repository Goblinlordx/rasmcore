//! Pixel format conversion — V1 DecodedImage (u8/u16/f32 multi-format) to V2 (f32 RGBA).
//!
//! V2 pipeline uses a single pixel format: `&[f32]` with 4 channels (RGBA) per pixel.
//! This module converts from V1's various pixel formats to that uniform representation.

use rasmcore_image::domain::types::{DecodedImage as V1DecodedImage, ImageInfo, PixelFormat};
use rasmcore_pipeline_v2::PipelineError;
use rasmcore_pipeline_v2::color_space::ColorSpace as V2ColorSpace;
use rasmcore_pipeline_v2::node::NodeInfo;
use rasmcore_pipeline_v2::ops::DecodedImage as V2DecodedImage;

/// Convert a V1 DecodedImage to V2 DecodedImage (f32 linear RGBA).
///
/// sRGB sources are degamma'd to linear during conversion so the V2 pipeline
/// operates entirely in linear light. The encoder re-applies the sRGB transfer
/// function on output.
pub fn v1_to_v2(v1: V1DecodedImage) -> Result<V2DecodedImage, PipelineError> {
    use rasmcore_pipeline_v2::image_metadata::{ImageMetadata, MetadataValue, derive_color_space};

    let pixels = pixels_to_f32_rgba(&v1.pixels, v1.info.format, v1.info.width, v1.info.height)?;

    // Build metadata from V1 fields
    let mut metadata = ImageMetadata::new();
    if let Some(icc) = v1.icc_profile {
        metadata.set("icc", MetadataValue::Bytes(icc));
    }

    // Derive color space from metadata (ICC profile, EXIF), fallback to V1's declared space
    let color_space =
        derive_color_space(&metadata).unwrap_or_else(|| map_color_space(v1.info.color_space));

    // Linearize sRGB sources — V2 pipeline is f32 linear throughout.
    let pixels = if color_space == V2ColorSpace::Srgb {
        linearize_srgb(pixels)
    } else {
        pixels
    };

    Ok(V2DecodedImage {
        pixels,
        info: NodeInfo {
            width: v1.info.width,
            height: v1.info.height,
            color_space: if color_space == V2ColorSpace::Srgb {
                V2ColorSpace::Linear
            } else {
                color_space
            },
        },
        metadata,
    })
}

/// Convert f32 RGBA pixels to V1-compatible u8 Rgba8 with optional sRGB gamma.
///
/// If `apply_gamma` is true, applies linear-to-sRGB transfer function before
/// quantization (for sRGB output formats like JPEG, PNG, WebP).
/// If false, clamps and quantizes directly (for linear output or formats
/// that expect linear input).
pub fn f32_to_v1_rgba8(
    pixels: &[f32],
    width: u32,
    height: u32,
    apply_gamma: bool,
) -> (Vec<u8>, ImageInfo) {
    let npixels = (width as usize) * (height as usize);
    let mut out = Vec::with_capacity(npixels * 4);

    if apply_gamma {
        for chunk in pixels.chunks_exact(4) {
            out.push(linear_to_srgb_u8(chunk[0]));
            out.push(linear_to_srgb_u8(chunk[1]));
            out.push(linear_to_srgb_u8(chunk[2]));
            out.push((chunk[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    } else {
        for chunk in pixels.chunks_exact(4) {
            out.push((chunk[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((chunk[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((chunk[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((chunk[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }

    let info = ImageInfo {
        width,
        height,
        format: PixelFormat::Rgba8,
        color_space: rasmcore_image::domain::types::ColorSpace::Srgb,
    };
    (out, info)
}

/// Convert f32 RGBA to V1-compatible u8 Rgb8 (alpha dropped) with optional sRGB gamma.
///
/// For formats that don't support alpha channels (BMP, JPEG, PNM, HDR).
pub fn f32_to_v1_rgb8(
    pixels: &[f32],
    width: u32,
    height: u32,
    apply_gamma: bool,
) -> (Vec<u8>, ImageInfo) {
    let npixels = (width as usize) * (height as usize);
    let mut out = Vec::with_capacity(npixels * 3);

    if apply_gamma {
        for chunk in pixels.chunks_exact(4) {
            out.push(linear_to_srgb_u8(chunk[0]));
            out.push(linear_to_srgb_u8(chunk[1]));
            out.push(linear_to_srgb_u8(chunk[2]));
        }
    } else {
        for chunk in pixels.chunks_exact(4) {
            out.push((chunk[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((chunk[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            out.push((chunk[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }

    let info = ImageInfo {
        width,
        height,
        format: PixelFormat::Rgb8,
        color_space: rasmcore_image::domain::types::ColorSpace::Srgb,
    };
    (out, info)
}

/// Convert f32 RGBA to V1-compatible f32 LE bytes (Rgba32f).
///
/// For linear HDR formats (EXR, HDR, FITS) that encode f32 directly.
pub fn f32_to_v1_rgba32f_bytes(pixels: &[f32], width: u32, height: u32) -> (Vec<u8>, ImageInfo) {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for &v in pixels {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let info = ImageInfo {
        width,
        height,
        format: PixelFormat::Rgba32f,
        color_space: rasmcore_image::domain::types::ColorSpace::LinearSrgb,
    };
    (out, info)
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Convert V1 pixel buffer to f32 RGBA based on the pixel format.
fn pixels_to_f32_rgba(
    data: &[u8],
    format: PixelFormat,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, PipelineError> {
    let npixels = (width as usize) * (height as usize);

    match format {
        PixelFormat::Rgba8 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(4) {
                out.push(chunk[0] as f32 / 255.0);
                out.push(chunk[1] as f32 / 255.0);
                out.push(chunk[2] as f32 / 255.0);
                out.push(chunk[3] as f32 / 255.0);
            }
            Ok(out)
        }
        PixelFormat::Rgb8 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(3) {
                out.push(chunk[0] as f32 / 255.0);
                out.push(chunk[1] as f32 / 255.0);
                out.push(chunk[2] as f32 / 255.0);
                out.push(1.0);
            }
            Ok(out)
        }
        PixelFormat::Gray8 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for &v in data {
                let f = v as f32 / 255.0;
                out.push(f);
                out.push(f);
                out.push(f);
                out.push(1.0);
            }
            Ok(out)
        }
        PixelFormat::Bgr8 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(3) {
                out.push(chunk[2] as f32 / 255.0); // R
                out.push(chunk[1] as f32 / 255.0); // G
                out.push(chunk[0] as f32 / 255.0); // B
                out.push(1.0);
            }
            Ok(out)
        }
        PixelFormat::Bgra8 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(4) {
                out.push(chunk[2] as f32 / 255.0); // R
                out.push(chunk[1] as f32 / 255.0); // G
                out.push(chunk[0] as f32 / 255.0); // B
                out.push(chunk[3] as f32 / 255.0); // A
            }
            Ok(out)
        }
        PixelFormat::Rgba16 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(8) {
                let r = u16::from_le_bytes([chunk[0], chunk[1]]);
                let g = u16::from_le_bytes([chunk[2], chunk[3]]);
                let b = u16::from_le_bytes([chunk[4], chunk[5]]);
                let a = u16::from_le_bytes([chunk[6], chunk[7]]);
                out.push(r as f32 / 65535.0);
                out.push(g as f32 / 65535.0);
                out.push(b as f32 / 65535.0);
                out.push(a as f32 / 65535.0);
            }
            Ok(out)
        }
        PixelFormat::Rgb16 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(6) {
                let r = u16::from_le_bytes([chunk[0], chunk[1]]);
                let g = u16::from_le_bytes([chunk[2], chunk[3]]);
                let b = u16::from_le_bytes([chunk[4], chunk[5]]);
                out.push(r as f32 / 65535.0);
                out.push(g as f32 / 65535.0);
                out.push(b as f32 / 65535.0);
                out.push(1.0);
            }
            Ok(out)
        }
        PixelFormat::Gray16 => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f = v as f32 / 65535.0;
                out.push(f);
                out.push(f);
                out.push(f);
                out.push(1.0);
            }
            Ok(out)
        }
        PixelFormat::Rgba32f => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(16) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                out.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
                out.push(f32::from_le_bytes([
                    chunk[8], chunk[9], chunk[10], chunk[11],
                ]));
                out.push(f32::from_le_bytes([
                    chunk[12], chunk[13], chunk[14], chunk[15],
                ]));
            }
            Ok(out)
        }
        PixelFormat::Rgb32f => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(12) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                out.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
                out.push(f32::from_le_bytes([
                    chunk[8], chunk[9], chunk[10], chunk[11],
                ]));
                out.push(1.0);
            }
            Ok(out)
        }
        PixelFormat::Gray32f => {
            let mut out = Vec::with_capacity(npixels * 4);
            for chunk in data.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(v);
                out.push(v);
                out.push(v);
                out.push(1.0);
            }
            Ok(out)
        }
        other => Err(PipelineError::ComputeError(format!(
            "unsupported V1 pixel format for V2 conversion: {other:?}"
        ))),
    }
}

/// Map V1 ColorSpace to V2 ColorSpace.
fn map_color_space(v1: rasmcore_image::domain::types::ColorSpace) -> V2ColorSpace {
    match v1 {
        rasmcore_image::domain::types::ColorSpace::Srgb => V2ColorSpace::Srgb,
        rasmcore_image::domain::types::ColorSpace::LinearSrgb => V2ColorSpace::Linear,
        rasmcore_image::domain::types::ColorSpace::DisplayP3 => V2ColorSpace::DisplayP3,
        rasmcore_image::domain::types::ColorSpace::Bt709 => V2ColorSpace::Rec709,
        rasmcore_image::domain::types::ColorSpace::Bt2020 => V2ColorSpace::Rec2020,
        // Map less common V1 color spaces to Unknown for now
        _ => V2ColorSpace::Unknown,
    }
}

/// Apply sRGB EOTF (degamma) to f32 pixels that are sRGB-encoded.
/// Converts from sRGB gamma space to linear light. Alpha is unchanged.
fn linearize_srgb(mut pixels: Vec<f32>) -> Vec<f32> {
    for chunk in pixels.chunks_exact_mut(4) {
        chunk[0] = srgb_to_linear(chunk[0]);
        chunk[1] = srgb_to_linear(chunk[1]);
        chunk[2] = srgb_to_linear(chunk[2]);
        // alpha unchanged
    }
    pixels
}

/// sRGB gamma-encoded → Linear. Per-channel. IEC 61966-2-1.
#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    let c = v.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear to sRGB, then quantize to u8. IEC 61966-2-1.
#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 1.0);
    let srgb = if clamped <= 0.0031308 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba8_to_f32() {
        // 2x1 image: red pixel, green pixel
        // pixels_to_f32_rgba does raw normalization (no degamma) — that's correct,
        // linearization happens in v1_to_v2() based on color space.
        let v1_pixels: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 128];
        let f32_pixels = pixels_to_f32_rgba(&v1_pixels, PixelFormat::Rgba8, 2, 1).unwrap();
        assert_eq!(f32_pixels.len(), 8);
        assert!((f32_pixels[0] - 1.0).abs() < 1e-6); // R=1.0
        assert!((f32_pixels[1] - 0.0).abs() < 1e-6); // G=0.0
        assert!((f32_pixels[4] - 0.0).abs() < 1e-6); // R=0.0
        assert!((f32_pixels[5] - 1.0).abs() < 1e-6); // G=1.0
        assert!((f32_pixels[7] - 128.0 / 255.0).abs() < 1e-5); // A
    }

    #[test]
    fn srgb_linearize_roundtrip() {
        // sRGB mid-gray (128) should linearize then re-encode to ~128
        let srgb_val = 128.0 / 255.0; // ~0.502
        let linear = srgb_to_linear(srgb_val);
        let back = linear_to_srgb_u8(linear);
        assert!(
            (back as i32 - 128).unsigned_abs() <= 1,
            "roundtrip: 128 → {back}"
        );
    }

    #[test]
    fn srgb_identity_roundtrip_all_values() {
        // Every u8 value should roundtrip through linearize→encode within ±1
        for i in 0u8..=255 {
            let srgb = i as f32 / 255.0;
            let linear = srgb_to_linear(srgb);
            let back = linear_to_srgb_u8(linear);
            assert!(
                (back as i32 - i as i32).unsigned_abs() <= 1,
                "u8 {i} → linear {linear:.6} → u8 {back} (expected ~{i})"
            );
        }
    }

    #[test]
    fn gray8_to_rgba() {
        let v1_pixels: Vec<u8> = vec![128];
        let f32_pixels = pixels_to_f32_rgba(&v1_pixels, PixelFormat::Gray8, 1, 1).unwrap();
        assert_eq!(f32_pixels.len(), 4);
        let expected = 128.0 / 255.0;
        assert!((f32_pixels[0] - expected).abs() < 1e-5);
        assert!((f32_pixels[1] - expected).abs() < 1e-5);
        assert!((f32_pixels[2] - expected).abs() < 1e-5);
        assert!((f32_pixels[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_to_srgb_u8_boundaries() {
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
        // Mid-gray in linear (0.2140) should map to ~128 in sRGB
        let mid = linear_to_srgb_u8(0.2140);
        assert!((mid as i32 - 128).unsigned_abs() <= 1);
    }

    #[test]
    fn rgb16_to_rgba() {
        // One white pixel in u16 LE
        let v1_pixels: Vec<u8> = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let f32_pixels = pixels_to_f32_rgba(&v1_pixels, PixelFormat::Rgb16, 1, 1).unwrap();
        assert!((f32_pixels[0] - 1.0).abs() < 1e-5);
        assert!((f32_pixels[3] - 1.0).abs() < 1e-6); // alpha
    }
}
