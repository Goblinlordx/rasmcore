use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo, PixelFormat};

/// Convert pixel format using raw buffer operations.
///
/// Supports all conversions between Gray8, Rgb8, Rgba8, Gray16, Rgb16, Rgba16.
/// Grayscale conversion uses BT.601 luma: (77*R + 150*G + 29*B + 128) >> 8.
/// 8↔16 bit conversion uses proper rounding: u8→u16 via v*257, u16→u8 via (v+128)/257.
pub fn convert_format(
    pixels: &[u8],
    info: &ImageInfo,
    target: PixelFormat,
) -> Result<DecodedImage, ImageError> {
    if info.format == target {
        return Ok(DecodedImage {
            pixels: pixels.to_vec(),
            info: info.clone(),
            icc_profile: None,
        });
    }

    let n = (info.width as usize) * (info.height as usize);
    let new_pixels = convert_pixels(pixels, info.format, target, n)?;

    Ok(DecodedImage {
        pixels: new_pixels,
        info: ImageInfo {
            width: info.width,
            height: info.height,
            format: target,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Convert pixel data between formats.
fn convert_pixels(
    pixels: &[u8],
    src: PixelFormat,
    dst: PixelFormat,
    pixel_count: usize,
) -> Result<Vec<u8>, ImageError> {
    // Strategy: normalize to RGBA8 or RGBA16 as intermediate if needed,
    // then convert to target. For efficiency, handle direct conversions first.
    match (src, dst) {
        // ── Identity ──
        (a, b) if a == b => Ok(pixels.to_vec()),

        // ── 8-bit direct conversions ──
        (PixelFormat::Rgb8, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(3) {
                out.extend_from_slice(chunk);
                out.push(255);
            }
            Ok(out)
        }
        (PixelFormat::Rgba8, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(4) {
                out.extend_from_slice(&chunk[..3]);
            }
            Ok(out)
        }
        (PixelFormat::Rgb8, PixelFormat::Gray8) | (PixelFormat::Rgba8, PixelFormat::Gray8) => {
            let bpp = if src == PixelFormat::Rgb8 { 3 } else { 4 };
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(bpp) {
                let r = chunk[0] as u16;
                let g = chunk[1] as u16;
                let b = chunk[2] as u16;
                out.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for &v in pixels {
                out.push(v);
                out.push(v);
                out.push(v);
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &v in pixels {
                out.push(v);
                out.push(v);
                out.push(v);
                out.push(255);
            }
            Ok(out)
        }

        // ── 16-bit direct conversions ──
        (PixelFormat::Rgb16, PixelFormat::Rgba16) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for chunk in pixels.chunks_exact(6) {
                out.extend_from_slice(chunk);
                out.extend_from_slice(&65535u16.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba16, PixelFormat::Rgb16) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for chunk in pixels.chunks_exact(8) {
                out.extend_from_slice(&chunk[..6]);
            }
            Ok(out)
        }

        // ── 8→16 bit promotion ──
        (PixelFormat::Gray8, PixelFormat::Gray16) => {
            let mut out = Vec::with_capacity(pixel_count * 2);
            for &v in pixels {
                out.extend_from_slice(&(v as u16 * 257).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgb8, PixelFormat::Rgb16) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for &v in pixels {
                out.extend_from_slice(&(v as u16 * 257).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba8, PixelFormat::Rgba16) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for &v in pixels {
                out.extend_from_slice(&(v as u16 * 257).to_le_bytes());
            }
            Ok(out)
        }

        // ── 16→8 bit demotion ──
        (PixelFormat::Gray16, PixelFormat::Gray8) => {
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(((v as u32 + 128) / 257) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Rgb16, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(((v as u32 + 128) / 257) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Rgba16, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(((v as u32 + 128) / 257) as u8);
            }
            Ok(out)
        }

        // ── f16 → 8-bit demotion ──
        (PixelFormat::Rgba16f, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(8) {
                for i in 0..4 {
                    let v = half::f16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]);
                    out.push((v.to_f32() * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                }
            }
            Ok(out)
        }
        (PixelFormat::Rgb16f, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(6) {
                for i in 0..3 {
                    let v = half::f16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]);
                    out.push((v.to_f32() * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                }
            }
            Ok(out)
        }
        (PixelFormat::Gray16f, PixelFormat::Gray8) => {
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(2) {
                let v = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                out.push((v.to_f32() * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
            }
            Ok(out)
        }

        // ── 8-bit → f16 promotion ──
        (PixelFormat::Rgba8, PixelFormat::Rgba16f) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for &v in pixels {
                out.extend_from_slice(&half::f16::from_f32(v as f32 / 255.0).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgb8, PixelFormat::Rgb16f) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for &v in pixels {
                out.extend_from_slice(&half::f16::from_f32(v as f32 / 255.0).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Gray16f) => {
            let mut out = Vec::with_capacity(pixel_count * 2);
            for &v in pixels {
                out.extend_from_slice(&half::f16::from_f32(v as f32 / 255.0).to_le_bytes());
            }
            Ok(out)
        }

        // ── f16 ↔ f32 ──
        (PixelFormat::Rgba16f, PixelFormat::Rgba32f)
        | (PixelFormat::Rgb16f, PixelFormat::Rgb32f)
        | (PixelFormat::Gray16f, PixelFormat::Gray32f) => {
            let mut out = Vec::with_capacity(pixels.len() * 2);
            for chunk in pixels.chunks_exact(2) {
                let v = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                out.extend_from_slice(&v.to_f32().to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Rgba16f)
        | (PixelFormat::Rgb32f, PixelFormat::Rgb16f)
        | (PixelFormat::Gray32f, PixelFormat::Gray16f) => {
            let mut out = Vec::with_capacity(pixels.len() / 2);
            for chunk in pixels.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            Ok(out)
        }

        // ── f16 channel conversions ──
        (PixelFormat::Rgb16f, PixelFormat::Rgba16f) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for chunk in pixels.chunks_exact(6) {
                out.extend_from_slice(chunk);
                out.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba16f, PixelFormat::Rgb16f) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for chunk in pixels.chunks_exact(8) {
                out.extend_from_slice(&chunk[..6]);
            }
            Ok(out)
        }

        // ── f32 → 8-bit demotion ──
        (PixelFormat::Rgba32f, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(16) {
                for i in 0..4 {
                    let v = f32::from_le_bytes([
                        chunk[i * 4],
                        chunk[i * 4 + 1],
                        chunk[i * 4 + 2],
                        chunk[i * 4 + 3],
                    ]);
                    out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                }
            }
            Ok(out)
        }
        (PixelFormat::Rgb32f, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(12) {
                for i in 0..3 {
                    let v = f32::from_le_bytes([
                        chunk[i * 4],
                        chunk[i * 4 + 1],
                        chunk[i * 4 + 2],
                        chunk[i * 4 + 3],
                    ]);
                    out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                }
            }
            Ok(out)
        }
        (PixelFormat::Gray32f, PixelFormat::Gray8) => {
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
            }
            Ok(out)
        }

        // ── 8-bit → f32 promotion ──
        (PixelFormat::Rgba8, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for &v in pixels {
                out.extend_from_slice(&(v as f32 / 255.0).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgb8, PixelFormat::Rgb32f) => {
            let mut out = Vec::with_capacity(pixel_count * 12);
            for &v in pixels {
                out.extend_from_slice(&(v as f32 / 255.0).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Gray32f) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &v in pixels {
                out.extend_from_slice(&(v as f32 / 255.0).to_le_bytes());
            }
            Ok(out)
        }

        // ── f32 channel conversions ──
        (PixelFormat::Rgb32f, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(12) {
                out.extend_from_slice(chunk);
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Rgb32f) => {
            let mut out = Vec::with_capacity(pixel_count * 12);
            for chunk in pixels.chunks_exact(16) {
                out.extend_from_slice(&chunk[..12]);
            }
            Ok(out)
        }

        // ── Direct lossless promotion to Rgba32f (f32 pipeline) ──
        // These paths preserve full precision without going through u8.
        (PixelFormat::Rgb8, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(3) {
                out.extend_from_slice(&(chunk[0] as f32 / 255.0).to_le_bytes());
                out.extend_from_slice(&(chunk[1] as f32 / 255.0).to_le_bytes());
                out.extend_from_slice(&(chunk[2] as f32 / 255.0).to_le_bytes());
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for &v in pixels {
                let f = (v as f32 / 255.0).to_le_bytes();
                out.extend_from_slice(&f);
                out.extend_from_slice(&f);
                out.extend_from_slice(&f);
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Bgr8, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(3) {
                out.extend_from_slice(&(chunk[2] as f32 / 255.0).to_le_bytes()); // R
                out.extend_from_slice(&(chunk[1] as f32 / 255.0).to_le_bytes()); // G
                out.extend_from_slice(&(chunk[0] as f32 / 255.0).to_le_bytes()); // B
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Bgra8, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(4) {
                out.extend_from_slice(&(chunk[2] as f32 / 255.0).to_le_bytes()); // R
                out.extend_from_slice(&(chunk[1] as f32 / 255.0).to_le_bytes()); // G
                out.extend_from_slice(&(chunk[0] as f32 / 255.0).to_le_bytes()); // B
                out.extend_from_slice(&(chunk[3] as f32 / 255.0).to_le_bytes()); // A
            }
            Ok(out)
        }
        (PixelFormat::Gray16, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f = (v as f32 / 65535.0).to_le_bytes();
                out.extend_from_slice(&f);
                out.extend_from_slice(&f);
                out.extend_from_slice(&f);
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgb16, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(6) {
                let r = u16::from_le_bytes([chunk[0], chunk[1]]);
                let g = u16::from_le_bytes([chunk[2], chunk[3]]);
                let b = u16::from_le_bytes([chunk[4], chunk[5]]);
                out.extend_from_slice(&(r as f32 / 65535.0).to_le_bytes());
                out.extend_from_slice(&(g as f32 / 65535.0).to_le_bytes());
                out.extend_from_slice(&(b as f32 / 65535.0).to_le_bytes());
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba16, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(8) {
                let r = u16::from_le_bytes([chunk[0], chunk[1]]);
                let g = u16::from_le_bytes([chunk[2], chunk[3]]);
                let b = u16::from_le_bytes([chunk[4], chunk[5]]);
                let a = u16::from_le_bytes([chunk[6], chunk[7]]);
                out.extend_from_slice(&(r as f32 / 65535.0).to_le_bytes());
                out.extend_from_slice(&(g as f32 / 65535.0).to_le_bytes());
                out.extend_from_slice(&(b as f32 / 65535.0).to_le_bytes());
                out.extend_from_slice(&(a as f32 / 65535.0).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Gray16f, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(2) {
                let v = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                let f = v.to_f32().to_le_bytes();
                out.extend_from_slice(&f);
                out.extend_from_slice(&f);
                out.extend_from_slice(&f);
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgb16f, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(6) {
                let r = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                let g = half::f16::from_le_bytes([chunk[2], chunk[3]]);
                let b = half::f16::from_le_bytes([chunk[4], chunk[5]]);
                out.extend_from_slice(&r.to_f32().to_le_bytes());
                out.extend_from_slice(&g.to_f32().to_le_bytes());
                out.extend_from_slice(&b.to_f32().to_le_bytes());
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Gray32f, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(4) {
                out.extend_from_slice(chunk); // R = luma
                out.extend_from_slice(chunk); // G = luma
                out.extend_from_slice(chunk); // B = luma
                out.extend_from_slice(&1.0f32.to_le_bytes()); // A = 1.0
            }
            Ok(out)
        }
        (PixelFormat::Cmyk8, PixelFormat::Rgba32f) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for chunk in pixels.chunks_exact(4) {
                let c = chunk[0] as f32 / 255.0;
                let m = chunk[1] as f32 / 255.0;
                let y = chunk[2] as f32 / 255.0;
                let k = chunk[3] as f32 / 255.0;
                let r = (1.0 - c) * (1.0 - k);
                let g = (1.0 - m) * (1.0 - k);
                let b = (1.0 - y) * (1.0 - k);
                out.extend_from_slice(&r.to_le_bytes());
                out.extend_from_slice(&g.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
                out.extend_from_slice(&1.0f32.to_le_bytes());
            }
            Ok(out)
        }

        // ── Direct quantization from Rgba32f (f32 pipeline) ──
        (PixelFormat::Rgba32f, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(16) {
                for i in 0..3 {
                    let v = f32::from_le_bytes([
                        chunk[i * 4],
                        chunk[i * 4 + 1],
                        chunk[i * 4 + 2],
                        chunk[i * 4 + 3],
                    ]);
                    out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                }
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Gray8) => {
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(16) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                out.push((luma * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Rgba16) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for chunk in pixels.chunks_exact(16) {
                for i in 0..4 {
                    let v = f32::from_le_bytes([
                        chunk[i * 4],
                        chunk[i * 4 + 1],
                        chunk[i * 4 + 2],
                        chunk[i * 4 + 3],
                    ]);
                    let u = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                    out.extend_from_slice(&u.to_le_bytes());
                }
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Rgb16) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for chunk in pixels.chunks_exact(16) {
                for i in 0..3 {
                    let v = f32::from_le_bytes([
                        chunk[i * 4],
                        chunk[i * 4 + 1],
                        chunk[i * 4 + 2],
                        chunk[i * 4 + 3],
                    ]);
                    let u = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                    out.extend_from_slice(&u.to_le_bytes());
                }
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Gray16) => {
            let mut out = Vec::with_capacity(pixel_count * 2);
            for chunk in pixels.chunks_exact(16) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let u = (luma * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                out.extend_from_slice(&u.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba32f, PixelFormat::Gray32f) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(16) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                out.extend_from_slice(&luma.to_le_bytes());
            }
            Ok(out)
        }

        // ── Cross-depth + cross-channel: two-step via intermediate ──
        _ => {
            // Step 1: convert to 8-bit same-channel-count
            let (intermediate, intermediate_fmt) = match src {
                PixelFormat::Gray16 => (
                    convert_pixels(pixels, src, PixelFormat::Gray8, pixel_count)?,
                    PixelFormat::Gray8,
                ),
                PixelFormat::Rgb16 => (
                    convert_pixels(pixels, src, PixelFormat::Rgb8, pixel_count)?,
                    PixelFormat::Rgb8,
                ),
                PixelFormat::Rgba16 => (
                    convert_pixels(pixels, src, PixelFormat::Rgba8, pixel_count)?,
                    PixelFormat::Rgba8,
                ),
                PixelFormat::Gray16f => (
                    convert_pixels(pixels, src, PixelFormat::Gray8, pixel_count)?,
                    PixelFormat::Gray8,
                ),
                PixelFormat::Rgb16f => (
                    convert_pixels(pixels, src, PixelFormat::Rgb8, pixel_count)?,
                    PixelFormat::Rgb8,
                ),
                PixelFormat::Rgba16f => (
                    convert_pixels(pixels, src, PixelFormat::Rgba8, pixel_count)?,
                    PixelFormat::Rgba8,
                ),
                PixelFormat::Gray32f => (
                    convert_pixels(pixels, src, PixelFormat::Gray8, pixel_count)?,
                    PixelFormat::Gray8,
                ),
                PixelFormat::Rgb32f => (
                    convert_pixels(pixels, src, PixelFormat::Rgb8, pixel_count)?,
                    PixelFormat::Rgb8,
                ),
                PixelFormat::Rgba32f => (
                    convert_pixels(pixels, src, PixelFormat::Rgba8, pixel_count)?,
                    PixelFormat::Rgba8,
                ),
                other => (pixels.to_vec(), other),
            };
            // Step 2: convert channels at 8-bit
            let (channel_converted, channel_fmt) = if intermediate_fmt == dst {
                return Ok(intermediate);
            } else {
                // Get 8-bit target
                let target_8 = match dst {
                    PixelFormat::Gray8
                    | PixelFormat::Gray16
                    | PixelFormat::Gray16f
                    | PixelFormat::Gray32f => PixelFormat::Gray8,
                    PixelFormat::Rgb8
                    | PixelFormat::Rgb16
                    | PixelFormat::Rgb16f
                    | PixelFormat::Rgb32f => PixelFormat::Rgb8,
                    PixelFormat::Rgba8
                    | PixelFormat::Rgba16
                    | PixelFormat::Rgba16f
                    | PixelFormat::Rgba32f => PixelFormat::Rgba8,
                    _ => {
                        return Err(ImageError::UnsupportedFormat(format!(
                            "conversion from {src:?} to {dst:?} not supported"
                        )));
                    }
                };
                if intermediate_fmt == target_8 {
                    (intermediate, target_8)
                } else {
                    (
                        convert_pixels(&intermediate, intermediate_fmt, target_8, pixel_count)?,
                        target_8,
                    )
                }
            };
            // Step 3: promote to 16-bit or f32 if needed
            if channel_fmt == dst {
                Ok(channel_converted)
            } else {
                convert_pixels(&channel_converted, channel_fmt, dst, pixel_count)
            }
        }
    }
}
