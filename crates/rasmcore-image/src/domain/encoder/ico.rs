use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// ICO encode configuration. ICO embeds PNG sub-images, no meaningful params.
#[derive(Debug, Clone, Default)]
pub struct IcoEncodeConfig;

/// Encode pixel data to ICO.
///
/// ICO format embeds entries as PNG. Input images should ideally be square
/// and no larger than 256x256 (Windows icon convention), but the encoder
/// will accept any dimensions.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    _config: &IcoEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Ico)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::encoder::pixels_to_dynamic_image;
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
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        let result = encode(&img, &info, &IcoEncodeConfig).unwrap();
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
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        let result = encode(&img, &info, &IcoEncodeConfig);
        assert!(result.is_ok());
    }
}
