use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// GIF encode configuration.
#[derive(Debug, Clone, Default)]
pub struct GifEncodeConfig {
    /// Repeat count: 0 = infinite loop, n = repeat n times (default: 0 = infinite).
    pub repeat: u16,
}

/// Encode pixel data to GIF with the given configuration.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    _config: &GifEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Gif)
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
    fn encode_produces_valid_gif() {
        let (img, info) = make_test_image();
        let result = encode(&img, &info, &GifEncodeConfig::default()).unwrap();
        // GIF magic bytes: GIF89a or GIF87a
        assert_eq!(&result[..3], b"GIF");
    }

    #[test]
    fn encode_rgba8_produces_valid_gif() {
        let pixels: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        let result = encode(&img, &info, &GifEncodeConfig::default()).unwrap();
        assert_eq!(&result[..3], b"GIF");
    }

    #[test]
    fn default_config_repeat_is_infinite() {
        assert_eq!(GifEncodeConfig::default().repeat, 0);
    }
}
