use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// QOI encode configuration. QOI is a fixed lossless format with no parameters.
#[derive(Debug, Clone, Default)]
pub struct QoiEncodeConfig;

/// Encode pixel data to QOI (Quite OK Image).
pub fn encode(
    img: &DynamicImage,
    _info: &ImageInfo,
    _config: &QoiEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Qoi)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::encoder::pixels_to_dynamic_image;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    #[test]
    fn encode_produces_valid_qoi() {
        let pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        let result = encode(&img, &info, &QoiEncodeConfig).unwrap();
        // QOI magic: "qoif"
        assert_eq!(&result[..4], b"qoif");
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
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        let result = encode(&img, &info, &QoiEncodeConfig);
        assert!(result.is_ok());
    }
}
