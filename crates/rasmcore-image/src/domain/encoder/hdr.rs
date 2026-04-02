use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// HDR (Radiance RGBE) encode configuration.
#[derive(Debug, Clone, Default)]
pub struct HdrEncodeConfig;

/// Encode pixel data to Radiance HDR (.hdr) format.
///
/// Converts 8-bit input to RGBE (shared exponent) encoding.
/// Uses uncompressed scanline encoding (no RLE for simplicity).
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    _config: &HdrEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let mut buf = Vec::new();

    // Radiance HDR header
    buf.extend_from_slice(b"#?RADIANCE\n");
    buf.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n");
    buf.extend_from_slice(b"\n");

    // Resolution string: -Y height +X width (top-to-bottom, left-to-right)
    let res = format!("-Y {h} +X {w}\n");
    buf.extend_from_slice(res.as_bytes());

    // Determine pixel layout and convert each pixel to RGBE
    match info.format {
        PixelFormat::Rgb32f => {
            // Native f32 RGB — use directly, no u8→f32 conversion
            for chunk in pixels.chunks_exact(12) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                buf.extend_from_slice(&to_rgbe(r, g, b));
            }
        }
        PixelFormat::Rgba32f => {
            // Native f32 RGBA — use RGB, ignore alpha
            for chunk in pixels.chunks_exact(16) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                buf.extend_from_slice(&to_rgbe(r, g, b));
            }
        }
        PixelFormat::Rgb8 => {
            for chunk in pixels.chunks_exact(3) {
                let r = chunk[0] as f32 / 255.0;
                let g = chunk[1] as f32 / 255.0;
                let b = chunk[2] as f32 / 255.0;
                buf.extend_from_slice(&to_rgbe(r, g, b));
            }
        }
        PixelFormat::Rgba8 => {
            for chunk in pixels.chunks_exact(4) {
                let r = chunk[0] as f32 / 255.0;
                let g = chunk[1] as f32 / 255.0;
                let b = chunk[2] as f32 / 255.0;
                buf.extend_from_slice(&to_rgbe(r, g, b));
            }
        }
        PixelFormat::Gray8 => {
            for &v in pixels {
                let f = v as f32 / 255.0;
                buf.extend_from_slice(&to_rgbe(f, f, f));
            }
        }
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "HDR encode requires RGB8, RGBA8, Gray8, Rgb32f, or Rgba32f".into(),
            ));
        }
    }

    Ok(buf)
}

/// Convert linear RGB to RGBE (shared exponent) encoding.
///
/// Uses the standard Radiance RGBE encoding: find the exponent such that
/// the largest channel, when scaled by 256 / 2^exp, fits in [1, 255].
fn to_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
    let max_val = r.max(g).max(b);
    if max_val < 1e-32 {
        return [0, 0, 0, 0];
    }
    // frexp-style: find e such that max_val * 256 / 2^e is in [128, 256)
    // This gives e = floor(log2(max_val)) + 1, i.e., ceil(log2(max_val)) when not exact
    let e = max_val.log2().floor() as i32 + 1;
    let inv_scale = 2.0f32.powi(e);
    let scale = 256.0 / inv_scale;
    [
        (r * scale + 0.5).min(255.0) as u8,
        (g * scale + 0.5).min(255.0) as u8,
        (b * scale + 0.5).min(255.0) as u8,
        (e + 128) as u8,
    ]
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "hdr",
        format: "hdr",
        mime: "image/vnd.radiance",
        extensions: &["hdr"],
        fn_name: "encode_hdr",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    #[test]
    fn encode_produces_valid_hdr() {
        let pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &HdrEncodeConfig).unwrap();
        assert!(result.starts_with(b"#?RADIANCE\n"));
    }

    #[test]
    fn rgbe_black() {
        assert_eq!(to_rgbe(0.0, 0.0, 0.0), [0, 0, 0, 0]);
    }

    #[test]
    fn rgbe_white() {
        let rgbe = to_rgbe(1.0, 1.0, 1.0);
        // 1.0 → e=1, scale=128 → R=G=B=128, E=129
        // Decodes: 128 * 2^(129-128-8) = 128/128 = 1.0
        assert_eq!(rgbe, [128, 128, 128, 129]);
    }

    #[test]
    fn encode_rgb32f() {
        // Create 4x4 Rgb32f gradient
        let mut pixels = Vec::with_capacity(4 * 4 * 12);
        for i in 0..16u32 {
            let v = i as f32 / 15.0;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&(1.0 - v).to_le_bytes());
            pixels.extend_from_slice(&0.5f32.to_le_bytes());
        }
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb32f,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &HdrEncodeConfig).unwrap();
        assert!(result.starts_with(b"#?RADIANCE\n"));
    }

    #[test]
    fn hdr_f32_round_trip_rgbe_precision() {
        use crate::domain::decoder;

        // Create known f32 values
        let (w, h) = (4, 4);
        let mut orig_pixels = Vec::with_capacity((w * h * 12) as usize);
        let mut orig_values = Vec::new();
        for i in 0..(w * h) {
            // Use values in [0.01, 1.0] range to avoid RGBE edge cases near zero
            let v = 0.01 + (i as f32 / (w * h) as f32) * 0.99;
            let r = v;
            let g = 1.0 - v;
            let b = v * 0.5 + 0.25;
            orig_pixels.extend_from_slice(&r.to_le_bytes());
            orig_pixels.extend_from_slice(&g.to_le_bytes());
            orig_pixels.extend_from_slice(&b.to_le_bytes());
            orig_values.push((r, g, b));
        }

        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb32f,
            color_space: ColorSpace::Srgb,
        };

        // Encode
        let encoded = encode_pixels(&orig_pixels, &info, &HdrEncodeConfig).unwrap();

        // Decode f32
        let decoded = decoder::decode_f32(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgb32f);
        assert_eq!(decoded.info.width, w);
        assert_eq!(decoded.info.height, h);

        // RGBE has ~1% relative precision. Verify within tolerance.
        for (i, &(orig_r, orig_g, orig_b)) in orig_values.iter().enumerate() {
            let off = i * 12;
            let back_r = f32::from_le_bytes([
                decoded.pixels[off], decoded.pixels[off+1], decoded.pixels[off+2], decoded.pixels[off+3],
            ]);
            let back_g = f32::from_le_bytes([
                decoded.pixels[off+4], decoded.pixels[off+5], decoded.pixels[off+6], decoded.pixels[off+7],
            ]);
            let back_b = f32::from_le_bytes([
                decoded.pixels[off+8], decoded.pixels[off+9], decoded.pixels[off+10], decoded.pixels[off+11],
            ]);

            // RGBE tolerance: ~1/128 relative error per channel
            let tol = |orig: f32| (orig * 0.02).max(0.005);
            assert!(
                (orig_r - back_r).abs() <= tol(orig_r),
                "pixel {i} R: orig={orig_r}, back={back_r}, diff={}",
                (orig_r - back_r).abs()
            );
            assert!(
                (orig_g - back_g).abs() <= tol(orig_g),
                "pixel {i} G: orig={orig_g}, back={back_g}, diff={}",
                (orig_g - back_g).abs()
            );
            assert!(
                (orig_b - back_b).abs() <= tol(orig_b),
                "pixel {i} B: orig={orig_b}, back={back_b}, diff={}",
                (orig_b - back_b).abs()
            );
        }
    }

    #[test]
    fn hdr_standard_decode_still_rgb8() {
        let pixels: Vec<u8> = (0..(4 * 4 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let encoded = encode_pixels(&pixels, &info, &HdrEncodeConfig).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn cross_format_exr_f32_to_hdr_f32() {
        use crate::domain::decoder;
        use crate::domain::encoder::exr as exr_enc;

        // Create Rgba32f data
        let (w, h) = (4, 4);
        let mut rgba_f32 = Vec::with_capacity((w * h * 16) as usize);
        for i in 0..(w * h) {
            let v = 0.1 + (i as f32 / (w * h) as f32) * 0.8;
            rgba_f32.extend_from_slice(&v.to_le_bytes());
            rgba_f32.extend_from_slice(&(1.0 - v).to_le_bytes());
            rgba_f32.extend_from_slice(&(v * 0.5).to_le_bytes());
            rgba_f32.extend_from_slice(&1.0f32.to_le_bytes());
        }

        let info_rgba = ImageInfo {
            width: w, height: h,
            format: PixelFormat::Rgba32f,
            color_space: ColorSpace::Srgb,
        };

        // EXR f32 encode
        let exr_bytes = exr_enc::encode_pixels(&rgba_f32, &info_rgba, &exr_enc::ExrEncodeConfig).unwrap();

        // EXR f32 decode
        let decoded_exr = decoder::decode_f32(&exr_bytes).unwrap();
        assert_eq!(decoded_exr.info.format, PixelFormat::Rgba32f);

        // HDR f32 encode (from Rgba32f — HDR accepts it)
        let hdr_bytes = encode_pixels(&decoded_exr.pixels, &decoded_exr.info, &HdrEncodeConfig).unwrap();

        // HDR f32 decode
        let decoded_hdr = decoder::decode_f32(&hdr_bytes).unwrap();
        assert_eq!(decoded_hdr.info.format, PixelFormat::Rgb32f);

        // Verify values within RGBE tolerance
        for i in 0..(w * h) as usize {
            let exr_off = i * 16;
            let hdr_off = i * 12;
            let exr_r = f32::from_le_bytes([
                decoded_exr.pixels[exr_off], decoded_exr.pixels[exr_off+1],
                decoded_exr.pixels[exr_off+2], decoded_exr.pixels[exr_off+3],
            ]);
            let hdr_r = f32::from_le_bytes([
                decoded_hdr.pixels[hdr_off], decoded_hdr.pixels[hdr_off+1],
                decoded_hdr.pixels[hdr_off+2], decoded_hdr.pixels[hdr_off+3],
            ]);
            let tol = (exr_r * 0.02).max(0.005);
            assert!(
                (exr_r - hdr_r).abs() <= tol,
                "pixel {i} R: exr={exr_r}, hdr={hdr_r}"
            );
        }
    }
}
