use image::{DynamicImage, ImageFormat};

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

const SUPPORTED_FORMATS: &[&str] = &["png", "jpeg", "webp"];

/// Encode pixel data to a specific image format
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, ImageError> {
    let img = pixels_to_dynamic_image(pixels, info)?;
    let image_format = str_to_format(format)?;

    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);

    match image_format {
        ImageFormat::Jpeg => {
            let q = quality.unwrap_or(85);
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, q);
            img.write_with_encoder(encoder)
                .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
        }
        other => {
            img.write_to(&mut cursor, other)
                .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
        }
    }

    Ok(buf)
}

/// List supported encode formats
pub fn supported_formats() -> Vec<String> {
    SUPPORTED_FORMATS.iter().map(|s| String::from(*s)).collect()
}

fn str_to_format(s: &str) -> Result<ImageFormat, ImageError> {
    match s {
        "png" => Ok(ImageFormat::Png),
        "jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
        "webp" => Ok(ImageFormat::WebP),
        other => Err(ImageError::UnsupportedFormat(format!(
            "encode format '{other}' not supported"
        ))),
    }
}

fn pixels_to_dynamic_image(pixels: &[u8], info: &ImageInfo) -> Result<DynamicImage, ImageError> {
    match info.format {
        PixelFormat::Rgb8 => {
            let img = image::RgbImage::from_raw(info.width, info.height, pixels.to_vec())
                .ok_or_else(|| {
                    ImageError::InvalidInput("pixel data size mismatch for RGB8".into())
                })?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        PixelFormat::Rgba8 => {
            let img = image::RgbaImage::from_raw(info.width, info.height, pixels.to_vec())
                .ok_or_else(|| {
                    ImageError::InvalidInput("pixel data size mismatch for RGBA8".into())
                })?;
            Ok(DynamicImage::ImageRgba8(img))
        }
        PixelFormat::Gray8 => {
            let img = image::GrayImage::from_raw(info.width, info.height, pixels.to_vec())
                .ok_or_else(|| {
                    ImageError::InvalidInput("pixel data size mismatch for Gray8".into())
                })?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "encoding from {other:?} pixel format not supported"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_rgb8_pixels(width: u32, height: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width,
            height,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_rgba8_pixels(width: u32, height: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(width * height * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width,
            height,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn encode_png_produces_valid_png() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "png", None).unwrap();
        // PNG magic bytes
        assert_eq!(&result[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn encode_jpeg_produces_valid_jpeg() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "jpeg", Some(90)).unwrap();
        // JPEG magic bytes
        assert_eq!(&result[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encode_jpeg_default_quality() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "jpeg", None);
        assert!(result.is_ok());
    }

    #[test]
    fn encode_webp_produces_output() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "webp", None);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn encode_rgba8_to_png() {
        let (pixels, info) = make_rgba8_pixels(8, 8);
        let result = encode(&pixels, &info, "png", None);
        assert!(result.is_ok());
    }

    #[test]
    fn encode_unsupported_format_returns_error() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let result = encode(&pixels, &info, "bmp", None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::UnsupportedFormat(_) => {}
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn encode_mismatched_pixel_data_returns_error() {
        let info = ImageInfo {
            width: 100,
            height: 100,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Too few pixels
        let result = encode(&[0u8; 10], &info, "png", None);
        assert!(result.is_err());
    }

    #[test]
    fn encode_unsupported_pixel_format_returns_error() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![0u8; 8 * 8 * 3];
        let result = encode(&pixels, &info, "png", None);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_png_preserves_pixels() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "png", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn supported_formats_lists_expected() {
        let fmts = supported_formats();
        assert!(fmts.contains(&"png".to_string()));
        assert!(fmts.contains(&"jpeg".to_string()));
        assert!(fmts.contains(&"webp".to_string()));
    }
}
