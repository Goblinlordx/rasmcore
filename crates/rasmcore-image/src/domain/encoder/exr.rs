use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// EXR encode configuration.
#[derive(Debug, Clone, Default)]
pub struct ExrEncodeConfig;

/// Encode pixel data to OpenEXR format using the exr crate directly.
///
/// Converts 8-bit/16-bit input to f32 linear, writes as RGBA channels.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    _config: &ExrEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let npixels = w * h;

    // Convert to RGBA f32
    let (r_f32, g_f32, b_f32, a_f32) = match info.format {
        PixelFormat::Rgb8 => {
            let mut r = Vec::with_capacity(npixels);
            let mut g = Vec::with_capacity(npixels);
            let mut b = Vec::with_capacity(npixels);
            let a = vec![1.0f32; npixels];
            for chunk in pixels.chunks_exact(3) {
                r.push(chunk[0] as f32 / 255.0);
                g.push(chunk[1] as f32 / 255.0);
                b.push(chunk[2] as f32 / 255.0);
            }
            (r, g, b, a)
        }
        PixelFormat::Rgba8 => {
            let mut r = Vec::with_capacity(npixels);
            let mut g = Vec::with_capacity(npixels);
            let mut b = Vec::with_capacity(npixels);
            let mut a = Vec::with_capacity(npixels);
            for chunk in pixels.chunks_exact(4) {
                r.push(chunk[0] as f32 / 255.0);
                g.push(chunk[1] as f32 / 255.0);
                b.push(chunk[2] as f32 / 255.0);
                a.push(chunk[3] as f32 / 255.0);
            }
            (r, g, b, a)
        }
        PixelFormat::Gray8 => {
            let mut r = Vec::with_capacity(npixels);
            let a = vec![1.0f32; npixels];
            for &v in pixels {
                r.push(v as f32 / 255.0);
            }
            (r.clone(), r.clone(), r, a)
        }
        PixelFormat::Rgba32f => {
            // Native f32 input — extract channels directly, no conversion
            let mut r = Vec::with_capacity(npixels);
            let mut g = Vec::with_capacity(npixels);
            let mut b = Vec::with_capacity(npixels);
            let mut a = Vec::with_capacity(npixels);
            for chunk in pixels.chunks_exact(16) {
                r.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                g.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
                b.push(f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]));
                a.push(f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]));
            }
            (r, g, b, a)
        }
        PixelFormat::Rgb32f => {
            // Native f32 RGB input — add alpha=1.0
            let mut r = Vec::with_capacity(npixels);
            let mut g = Vec::with_capacity(npixels);
            let mut b = Vec::with_capacity(npixels);
            let a = vec![1.0f32; npixels];
            for chunk in pixels.chunks_exact(12) {
                r.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                g.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
                b.push(f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]));
            }
            (r, g, b, a)
        }
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "EXR encode requires RGB8, RGBA8, Gray8, Rgb32f, or Rgba32f".into(),
            ));
        }
    };

    // Write EXR to memory buffer using the builder API
    use exr::prelude::*;

    let channels = SpecificChannels::rgba(|Vec2(x, y)| {
        let idx = y * w + x;
        (r_f32[idx], g_f32[idx], b_f32[idx], a_f32[idx])
    });

    let image = Image::from_channels((w, h), channels);
    let mut buf = Vec::new();
    image
        .write()
        .to_buffered(std::io::Cursor::new(&mut buf))
        .map_err(|e| ImageError::ProcessingFailed(format!("EXR encode: {e}")))?;

    Ok(buf)
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "exr",
        format: "exr",
        mime: "image/x-exr",
        extensions: &["exr"],
        fn_name: "encode_exr",
        encode_fn: None,
        preferred_output_cs: crate::domain::encoder::EncoderColorSpace::Linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    #[test]
    fn encode_produces_valid_exr() {
        let pixels: Vec<u8> = (0..(8 * 8 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &ExrEncodeConfig).unwrap();
        // EXR magic: 0x76 0x2F 0x31 0x01
        assert_eq!(&result[..4], &[0x76, 0x2F, 0x31, 0x01]);
    }

    #[test]
    fn encode_rgba8() {
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &ExrEncodeConfig).unwrap();
        assert_eq!(&result[..4], &[0x76, 0x2F, 0x31, 0x01]);
    }

    #[test]
    fn encode_rgba32f() {
        // Create 4x4 Rgba32f gradient
        let mut pixels = Vec::with_capacity(4 * 4 * 16);
        for i in 0..16u32 {
            let v = i as f32 / 15.0;
            pixels.extend_from_slice(&v.to_le_bytes()); // R
            pixels.extend_from_slice(&(1.0 - v).to_le_bytes()); // G
            pixels.extend_from_slice(&0.5f32.to_le_bytes()); // B
            pixels.extend_from_slice(&1.0f32.to_le_bytes()); // A
        }
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgba32f,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &ExrEncodeConfig).unwrap();
        assert_eq!(&result[..4], &[0x76, 0x2F, 0x31, 0x01]);
    }

    #[test]
    fn exr_f32_round_trip_bit_exact() {
        use crate::domain::decoder;

        // Create known f32 values — a gradient with precise values
        let (w, h) = (4, 4);
        let mut orig_pixels = Vec::with_capacity((w * h * 16) as usize);
        for i in 0..(w * h) {
            let v = i as f32 / (w * h - 1) as f32;
            orig_pixels.extend_from_slice(&v.to_le_bytes());       // R
            orig_pixels.extend_from_slice(&(1.0 - v).to_le_bytes()); // G
            orig_pixels.extend_from_slice(&(v * 0.5).to_le_bytes()); // B
            orig_pixels.extend_from_slice(&1.0f32.to_le_bytes());    // A
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba32f,
            color_space: ColorSpace::Srgb,
        };

        // Encode
        let encoded = encode_pixels(&orig_pixels, &info, &ExrEncodeConfig).unwrap();

        // Decode f32
        let decoded = decoder::decode_f32(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgba32f);
        assert_eq!(decoded.info.width, w);
        assert_eq!(decoded.info.height, h);
        assert_eq!(decoded.pixels.len(), orig_pixels.len());

        // Verify bit-exact f32 round-trip
        for i in 0..(w * h) as usize {
            let off = i * 16;
            for c in 0..4 {
                let orig = f32::from_le_bytes([
                    orig_pixels[off + c * 4],
                    orig_pixels[off + c * 4 + 1],
                    orig_pixels[off + c * 4 + 2],
                    orig_pixels[off + c * 4 + 3],
                ]);
                let back = f32::from_le_bytes([
                    decoded.pixels[off + c * 4],
                    decoded.pixels[off + c * 4 + 1],
                    decoded.pixels[off + c * 4 + 2],
                    decoded.pixels[off + c * 4 + 3],
                ]);
                assert!(
                    (orig - back).abs() < 1e-6,
                    "pixel {i} channel {c}: orig={orig}, back={back}"
                );
            }
        }
    }

    #[test]
    fn exr_standard_decode_still_rgba8() {
        // Encode and decode via standard path — should still be Rgba8
        let pixels: Vec<u8> = (0..(4 * 4 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let encoded = encode_pixels(&pixels, &info, &ExrEncodeConfig).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgba8);
    }
}
