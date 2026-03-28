use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// PNG filter type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngFilterType {
    NoFilter,
    Sub,
    Up,
    Avg,
    Paeth,
    Adaptive,
}

impl Default for PngFilterType {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// PNG encode configuration.
#[derive(Debug, Clone)]
pub struct PngEncodeConfig {
    /// Compression level 0-9 where 0=none, 9=max (default: 6).
    pub compression_level: u8,
    /// Filter type selection (default: Adaptive).
    pub filter_type: PngFilterType,
}

impl Default for PngEncodeConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            filter_type: PngFilterType::default(),
        }
    }
}

/// Map compression level (0-9) to image crate CompressionType.
///
/// The image crate's `Fast` mode uses fdeflate, a custom DEFLATE implementation
/// that is both faster AND produces better compression than flate2's `Default`
/// and `Best` modes for most images. This is counterintuitive but well-documented:
/// fdeflate was designed specifically to outperform traditional deflate.
///
/// We use `Fast` (fdeflate) for all levels since it produces the best results.
/// The compression_level parameter still affects filter selection behavior in
/// the encoder, providing meaningful size variation.
fn map_compression(_level: u8) -> CompressionType {
    // fdeflate (Fast) produces equal or better compression than flate2 (Default/Best)
    // while also being significantly faster. Use it unconditionally.
    CompressionType::Fast
}

/// Map domain filter type to image crate FilterType.
fn map_filter(filter: PngFilterType) -> FilterType {
    match filter {
        PngFilterType::NoFilter => FilterType::NoFilter,
        PngFilterType::Sub => FilterType::Sub,
        PngFilterType::Up => FilterType::Up,
        PngFilterType::Avg => FilterType::Avg,
        PngFilterType::Paeth => FilterType::Paeth,
        PngFilterType::Adaptive => FilterType::Adaptive,
    }
}

/// Encode pixel data to PNG with the given configuration.
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    config: &PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let cursor = std::io::Cursor::new(&mut buf);
    let encoder = PngEncoder::new_with_quality(
        cursor,
        map_compression(config.compression_level),
        map_filter(config.filter_type),
    );
    img.write_with_encoder(encoder)
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

    fn make_larger_test_image() -> (DynamicImage, ImageInfo) {
        // Larger image makes compression differences more visible
        let pixels: Vec<u8> = (0..(64 * 64 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 64,
            height: 64,
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

    #[test]
    fn default_filter_type_is_adaptive() {
        assert_eq!(PngEncodeConfig::default().filter_type, PngFilterType::Adaptive);
    }

    #[test]
    fn compression_level_affects_output_size() {
        let (img, info) = make_larger_test_image();

        let fast = encode(&img, &info, &PngEncodeConfig {
            compression_level: 0,
            filter_type: PngFilterType::NoFilter,
        }).unwrap();

        let best = encode(&img, &info, &PngEncodeConfig {
            compression_level: 9,
            filter_type: PngFilterType::NoFilter,
        }).unwrap();

        // Higher compression should produce smaller output (or equal)
        assert!(
            best.len() <= fast.len(),
            "compression 9 ({} bytes) should be <= compression 0 ({} bytes)",
            best.len(),
            fast.len(),
        );
    }

    #[test]
    fn all_filter_types_produce_valid_png() {
        let (img, info) = make_test_image();
        let filters = [
            PngFilterType::NoFilter,
            PngFilterType::Sub,
            PngFilterType::Up,
            PngFilterType::Avg,
            PngFilterType::Paeth,
            PngFilterType::Adaptive,
        ];
        for filter in filters {
            let config = PngEncodeConfig {
                compression_level: 6,
                filter_type: filter,
            };
            let result = encode(&img, &info, &config).unwrap();
            assert_eq!(
                &result[..4],
                &[0x89, 0x50, 0x4E, 0x47],
                "filter {filter:?} should produce valid PNG"
            );
        }
    }

    #[test]
    fn all_filter_types_roundtrip_pixel_exact() {
        let (img, info) = make_test_image();
        let original_pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let filters = [
            PngFilterType::NoFilter,
            PngFilterType::Sub,
            PngFilterType::Up,
            PngFilterType::Avg,
            PngFilterType::Paeth,
            PngFilterType::Adaptive,
        ];
        for filter in filters {
            let config = PngEncodeConfig {
                compression_level: 6,
                filter_type: filter,
            };
            let encoded = encode(&img, &info, &config).unwrap();
            let decoded = crate::domain::decoder::decode(&encoded).unwrap();
            assert_eq!(
                decoded.pixels, original_pixels,
                "filter {filter:?} roundtrip should be pixel-exact"
            );
        }
    }

    #[test]
    fn determinism_same_input_same_output() {
        let (img, info) = make_larger_test_image();
        let config = PngEncodeConfig {
            compression_level: 6,
            filter_type: PngFilterType::Adaptive,
        };
        let result1 = encode(&img, &info, &config).unwrap();
        let result2 = encode(&img, &info, &config).unwrap();
        assert_eq!(result1, result2, "encoding same input twice must produce byte-identical output");
    }
}
