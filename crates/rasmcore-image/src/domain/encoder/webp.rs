use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// WebP encode configuration.
#[derive(Debug, Clone)]
pub struct WebpEncodeConfig {
    /// Quality level 1-100 for lossy mode (default: 75).
    pub quality: u8,
    /// Use lossless encoding (default: false).
    pub lossless: bool,
}

impl Default for WebpEncodeConfig {
    fn default() -> Self {
        Self {
            quality: 75,
            lossless: false,
        }
    }
}

/// Encode pixel data to WebP with the given configuration.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    _config: &WebpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::WebP)
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
    fn encode_produces_output() {
        let (img, info) = make_test_image();
        let result = encode(&img, &info, &WebpEncodeConfig::default()).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn default_config() {
        let config = WebpEncodeConfig::default();
        assert_eq!(config.quality, 75);
        assert!(!config.lossless);
    }
}
