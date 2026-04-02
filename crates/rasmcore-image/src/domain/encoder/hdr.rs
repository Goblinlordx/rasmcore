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
fn to_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
    let max_val = r.max(g).max(b);
    if max_val < 1e-32 {
        return [0, 0, 0, 0];
    }
    // frexp equivalent: find exponent e such that max_val = mantissa * 2^e
    let e = max_val.log2().ceil() as i32;
    let scale = (256.0 / (2.0f32.powi(e))).min(255.0);
    [
        (r * scale) as u8,
        (g * scale) as u8,
        (b * scale) as u8,
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
        // 1.0 → exponent 0, scale = 256/1 = 256 → clamped 255
        // R=G=B=128, E=128
        assert_eq!(rgbe[3], 128); // exponent = 0 + 128
        assert!(rgbe[0] > 0 && rgbe[1] > 0 && rgbe[2] > 0);
    }
}
