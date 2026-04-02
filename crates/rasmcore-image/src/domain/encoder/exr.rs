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
}
