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
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "HDR encode requires RGB8, RGBA8, or Gray8".into(),
            ));
        }
    };

    let mut buf = Vec::new();

    // Radiance HDR header
    buf.extend_from_slice(b"#?RADIANCE\n");
    buf.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n");
    buf.extend_from_slice(b"\n");

    // Resolution string: -Y height +X width (top-to-bottom, left-to-right)
    let res = format!("-Y {h} +X {w}\n");
    buf.extend_from_slice(res.as_bytes());

    // Convert each scanline to RGBE
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * channels;
            let (r, g, b) = match channels {
                3 => (
                    pixels[idx] as f32 / 255.0,
                    pixels[idx + 1] as f32 / 255.0,
                    pixels[idx + 2] as f32 / 255.0,
                ),
                4 => (
                    pixels[idx] as f32 / 255.0,
                    pixels[idx + 1] as f32 / 255.0,
                    pixels[idx + 2] as f32 / 255.0,
                ),
                1 => {
                    let v = pixels[idx] as f32 / 255.0;
                    (v, v, v)
                }
                _ => unreachable!(),
            };

            let rgbe = to_rgbe(r, g, b);
            buf.extend_from_slice(&rgbe);
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
