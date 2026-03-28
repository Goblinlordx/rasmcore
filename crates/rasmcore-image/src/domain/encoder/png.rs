use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// PNG encode configuration.
#[derive(Debug, Clone)]
pub struct PngEncodeConfig {
    /// Compression level 0-9 where 0=none, 9=max (default: 6).
    pub compression_level: u8,
}

impl Default for PngEncodeConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
        }
    }
}

/// Encode pixel data to PNG with the given configuration.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    _config: &PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::encoder::pixels_to_dynamic_image;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn make_test_image() -> (DynamicImage, ImageInfo) {
        let pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        (img, info)
    }

    #[test]
    fn encode_produces_valid_png() {
        let (img, info) = make_test_image();
        let result = encode(&img, &info, &PngEncodeConfig::default()).unwrap();
        assert_eq!(&result[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn default_compression_level_is_6() {
        assert_eq!(PngEncodeConfig::default().compression_level, 6);
    }
}
