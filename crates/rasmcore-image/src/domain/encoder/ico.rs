use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// ICO encode configuration. ICO embeds PNG sub-images, no meaningful params.
#[derive(Debug, Clone, Default)]
pub struct IcoEncodeConfig;

/// Encode pixel data to ICO using PNG-in-ICO format.
///
/// ICO format: 6-byte header + 16-byte directory entry + embedded PNG data.
/// Input images should ideally be square and no larger than 256x256
/// (Windows icon convention), but any dimensions are accepted.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    _config: &IcoEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    // First, encode the pixel data as PNG
    let png_config = super::png::PngEncodeConfig::default();
    let png_data = super::png::encode(pixels, info, &png_config)?;

    // ICO header: 6 bytes
    // - reserved: u16 = 0
    // - type: u16 = 1 (icon)
    // - count: u16 = 1 (one entry)
    let mut buf = Vec::with_capacity(6 + 16 + png_data.len());

    // Header
    buf.extend_from_slice(&[0x00, 0x00]); // reserved
    buf.extend_from_slice(&1u16.to_le_bytes()); // type = icon
    buf.extend_from_slice(&1u16.to_le_bytes()); // count = 1

    // Directory entry: 16 bytes
    // Width/height: 0 means 256
    let w = if info.width >= 256 {
        0u8
    } else {
        info.width as u8
    };
    let h = if info.height >= 256 {
        0u8
    } else {
        info.height as u8
    };

    buf.push(w); // width
    buf.push(h); // height
    buf.push(0); // color count (0 = truecolor)
    buf.push(0); // reserved

    let bpp = match info.format {
        PixelFormat::Rgba8 => 32u16,
        PixelFormat::Rgb8 => 24,
        PixelFormat::Gray8 => 8,
        _ => 32,
    };
    buf.extend_from_slice(&1u16.to_le_bytes()); // color planes
    buf.extend_from_slice(&bpp.to_le_bytes()); // bits per pixel
    buf.extend_from_slice(&(png_data.len() as u32).to_le_bytes()); // image data size
    buf.extend_from_slice(&22u32.to_le_bytes()); // offset to image data (6 header + 16 entry)

    // PNG data
    buf.extend_from_slice(&png_data);

    Ok(buf)
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "ico",
        format: "ico",
        mime: "image/x-icon",
        extensions: &["ico"],
        fn_name: "encode_ico",
        encode_fn: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    #[test]
    fn encode_produces_valid_ico() {
        let pixels: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &IcoEncodeConfig).unwrap();
        // ICO magic: 00 00 01 00 (reserved=0, type=1=icon)
        assert_eq!(&result[..4], &[0x00, 0x00, 0x01, 0x00]);
    }

    #[test]
    fn encode_rgb8() {
        let pixels: Vec<u8> = (0..(32 * 32 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &IcoEncodeConfig);
        assert!(result.is_ok());
    }

    #[test]
    fn decode_roundtrip_preserves_dimensions() {
        let pixels: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let encoded = encode_pixels(&pixels, &info, &IcoEncodeConfig).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 16);
        assert_eq!(decoded.info.height, 16);
    }
}
