use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// JPEG encode configuration.
#[derive(Debug, Clone)]
pub struct JpegEncodeConfig {
    /// Quality level 1-100 (default: 85).
    pub quality: u8,
}

impl Default for JpegEncodeConfig {
    fn default() -> Self {
        Self { quality: 85 }
    }
}

/// Encode pixel data to JPEG with the given configuration.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    config: &JpegEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, config.quality);
    img.write_with_encoder(encoder)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::encoder::pixels_to_dynamic_image;
    use crate::domain::types::{ColorSpace, PixelFormat};

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
    fn encode_produces_valid_jpeg() {
        let (img, info) = make_test_image();
        let result = encode(&img, &info, &JpegEncodeConfig::default()).unwrap();
        assert_eq!(&result[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encode_with_custom_quality() {
        let (img, info) = make_test_image();
        let result = encode(&img, &info, &JpegEncodeConfig { quality: 50 });
        assert!(result.is_ok());
    }

    #[test]
    fn default_config_quality_is_85() {
        assert_eq!(JpegEncodeConfig::default().quality, 85);
    }
}
