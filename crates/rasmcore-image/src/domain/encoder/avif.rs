use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

use super::pixels_to_dynamic_image;

/// AVIF encode configuration.
#[derive(Debug, Clone)]
pub struct AvifEncodeConfig {
    /// Quality level 1-100 (default: 75). Higher = better quality, larger file.
    pub quality: u8,
    /// Speed preset 1-10 (default: 6). Higher = faster encode, lower quality.
    /// 1 = slowest/best, 10 = fastest/worst.
    pub speed: u8,
}

impl Default for AvifEncodeConfig {
    fn default() -> Self {
        Self {
            quality: 75,
            speed: 6,
        }
    }
}

/// Encode pixel data to AVIF with the given configuration.
///
/// Uses the image crate's AvifEncoder backed by rav1e (pure Rust AV1 encoder).
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    config: &AvifEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let img = pixels_to_dynamic_image(pixels, info)?;

    let mut buf = Vec::new();
    let cursor = std::io::Cursor::new(&mut buf);
    let encoder = image::codecs::avif::AvifEncoder::new_with_speed_quality(
        cursor,
        config.speed,
        config.quality,
    );
    img.write_with_encoder(encoder)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, PixelFormat};

    fn make_test_pixels(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn encode_produces_valid_avif() {
        let (pixels, info) = make_test_pixels(16, 16);
        let result = encode(&pixels, &info, &AvifEncodeConfig::default()).unwrap();
        // AVIF files start with ftyp box
        assert!(!result.is_empty());
        // Check for "ftyp" box (bytes 4-8 in ISO BMFF)
        assert_eq!(&result[4..8], b"ftyp");
    }

    #[test]
    fn encode_rgba8_produces_valid_avif() {
        let pixels: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode(&pixels, &info, &AvifEncodeConfig::default()).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn encode_with_custom_quality() {
        let (pixels, info) = make_test_pixels(16, 16);
        let config = AvifEncodeConfig {
            quality: 90,
            speed: 10,
        };
        let result = encode(&pixels, &info, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn default_config() {
        let config = AvifEncodeConfig::default();
        assert_eq!(config.quality, 75);
        assert_eq!(config.speed, 6);
    }
}
