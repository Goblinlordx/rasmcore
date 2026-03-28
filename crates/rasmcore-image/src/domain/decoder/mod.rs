use image::ImageFormat;

use super::color;
use super::error::ImageError;
use super::types::{ColorSpace, DecodedImage, ImageInfo, PixelFormat};

/// Supported decode formats
const SUPPORTED_FORMATS: &[&str] = &[
    "png", "jpeg", "gif", "webp", "bmp", "tiff", "avif", "qoi", "ico",
];

/// Detect image format from header bytes
pub fn detect_format(header: &[u8]) -> Option<String> {
    image::guess_format(header)
        .ok()
        .and_then(|fmt| format_to_str(fmt))
        .map(String::from)
}

/// Decode an image from raw bytes
pub fn decode(data: &[u8]) -> Result<DecodedImage, ImageError> {
    let img = image::load_from_memory(data).map_err(|e| ImageError::InvalidInput(e.to_string()))?;

    let format = detect_pixel_format(&img);
    let pixels = match format {
        PixelFormat::Rgba8 => img.to_rgba8().into_raw(),
        PixelFormat::Rgb8 => img.to_rgb8().into_raw(),
        PixelFormat::Gray8 => img.to_luma8().into_raw(),
        PixelFormat::Gray16 => {
            let luma16 = img.to_luma16();
            luma16
                .as_raw()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
        _ => img.to_rgba8().into_raw(),
    };

    // Extract ICC profile from raw bytes (before the image crate strips metadata)
    let icc_profile = extract_icc_profile(data);

    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width: img.width(),
            height: img.height(),
            format,
            color_space: ColorSpace::Srgb,
        },
        icc_profile,
    })
}

/// Extract ICC profile from raw image bytes based on detected format.
fn extract_icc_profile(data: &[u8]) -> Option<Vec<u8>> {
    match detect_format(data)?.as_str() {
        "jpeg" => color::extract_icc_from_jpeg(data),
        "png" => color::extract_icc_from_png(data),
        _ => None,
    }
}

/// Decode and convert to a specific pixel format
pub fn decode_as(data: &[u8], target_format: PixelFormat) -> Result<DecodedImage, ImageError> {
    let img = image::load_from_memory(data).map_err(|e| ImageError::InvalidInput(e.to_string()))?;

    let (pixels, format) = match target_format {
        PixelFormat::Rgb8 => (img.to_rgb8().into_raw(), PixelFormat::Rgb8),
        PixelFormat::Rgba8 => (img.to_rgba8().into_raw(), PixelFormat::Rgba8),
        PixelFormat::Gray8 => (img.to_luma8().into_raw(), PixelFormat::Gray8),
        PixelFormat::Gray16 => {
            let luma16 = img.to_luma16();
            let bytes = luma16
                .as_raw()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            (bytes, PixelFormat::Gray16)
        }
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "conversion to {other:?} not supported"
            )));
        }
    };

    let icc_profile = extract_icc_profile(data);

    Ok(DecodedImage {
        pixels,
        info: ImageInfo {
            width: img.width(),
            height: img.height(),
            format,
            color_space: ColorSpace::Srgb,
        },
        icc_profile,
    })
}

/// List supported decode formats
pub fn supported_formats() -> Vec<String> {
    SUPPORTED_FORMATS.iter().map(|s| String::from(*s)).collect()
}

fn detect_pixel_format(img: &image::DynamicImage) -> PixelFormat {
    match img.color() {
        image::ColorType::Rgb8 => PixelFormat::Rgb8,
        image::ColorType::Rgba8 => PixelFormat::Rgba8,
        image::ColorType::L8 => PixelFormat::Gray8,
        image::ColorType::L16 => PixelFormat::Gray16,
        _ => PixelFormat::Rgba8,
    }
}

fn format_to_str(fmt: ImageFormat) -> Option<&'static str> {
    match fmt {
        ImageFormat::Png => Some("png"),
        ImageFormat::Jpeg => Some("jpeg"),
        ImageFormat::Gif => Some("gif"),
        ImageFormat::WebP => Some("webp"),
        ImageFormat::Bmp => Some("bmp"),
        ImageFormat::Tiff => Some("tiff"),
        ImageFormat::Avif => Some("avif"),
        ImageFormat::Qoi => Some("qoi"),
        ImageFormat::Ico => Some("ico"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_png(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageFormat::Png).unwrap();
        buf
    }

    fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageFormat::Jpeg).unwrap();
        buf
    }

    #[test]
    fn detect_format_png() {
        let data = make_png(8, 8);
        assert_eq!(detect_format(&data), Some("png".to_string()));
    }

    #[test]
    fn detect_format_jpeg() {
        let data = make_jpeg(8, 8);
        assert_eq!(detect_format(&data), Some("jpeg".to_string()));
    }

    #[test]
    fn detect_format_empty() {
        assert_eq!(detect_format(&[]), None);
    }

    #[test]
    fn detect_format_garbage() {
        assert_eq!(detect_format(&[0xFF, 0x00, 0x42, 0x99]), None);
    }

    #[test]
    fn decode_png_returns_correct_dimensions() {
        let data = make_png(64, 32);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 32);
    }

    #[test]
    fn decode_png_returns_rgb8_format() {
        let data = make_png(8, 8);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn decode_png_pixel_data_length_matches() {
        let data = make_png(10, 10);
        let result = decode(&data).unwrap();
        let expected_len = 10 * 10 * 3;
        assert_eq!(result.pixels.len(), expected_len);
    }

    #[test]
    fn decode_jpeg_returns_correct_dimensions() {
        let data = make_jpeg(100, 50);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.width, 100);
        assert_eq!(result.info.height, 50);
    }

    #[test]
    fn decode_invalid_data_returns_error() {
        let result = decode(&[0x00, 0x01, 0x02]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn decode_as_rgba8() {
        let data = make_png(8, 8);
        let result = decode_as(&data, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn decode_as_gray8() {
        let data = make_png(8, 8);
        let result = decode_as(&data, PixelFormat::Gray8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 8 * 8 * 1);
    }

    #[test]
    fn decode_as_unsupported_returns_error() {
        let data = make_png(8, 8);
        let result = decode_as(&data, PixelFormat::Yuv420p);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::UnsupportedFormat(_) => {}
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn supported_formats_includes_common_formats() {
        let fmts = supported_formats();
        assert!(fmts.contains(&"png".to_string()));
        assert!(fmts.contains(&"jpeg".to_string()));
        assert!(fmts.contains(&"webp".to_string()));
        assert!(fmts.contains(&"gif".to_string()));
        assert!(fmts.contains(&"bmp".to_string()));
        assert!(fmts.contains(&"tiff".to_string()));
        assert!(fmts.contains(&"avif".to_string()));
        assert!(fmts.contains(&"qoi".to_string()));
        assert!(fmts.contains(&"ico".to_string()));
    }

    #[test]
    fn decode_color_space_defaults_to_srgb() {
        let data = make_png(8, 8);
        let result = decode(&data).unwrap();
        assert_eq!(result.info.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn decode_png_no_icc_profile() {
        let data = make_png(8, 8);
        let result = decode(&data).unwrap();
        assert!(result.icc_profile.is_none());
    }

    #[test]
    fn decode_jpeg_no_icc_profile() {
        let data = make_jpeg(8, 8);
        let result = decode(&data).unwrap();
        assert!(result.icc_profile.is_none());
    }

    #[test]
    fn decode_jpeg_with_icc_extracts_profile() {
        let jpeg = make_jpeg(8, 8);
        // Embed a fake ICC profile into the JPEG
        let fake_icc = vec![42u8; 128];
        let jpeg_with_icc =
            crate::domain::encoder::jpeg::embed_icc_profile(&jpeg, &fake_icc).unwrap();
        let result = decode(&jpeg_with_icc).unwrap();
        assert_eq!(result.icc_profile, Some(fake_icc));
    }

    #[test]
    fn decode_png_with_icc_extracts_profile() {
        let png = make_png(8, 8);
        // Embed a fake ICC profile into the PNG
        let fake_icc = vec![99u8; 200];
        let png_with_icc =
            crate::domain::encoder::png::embed_icc_profile(&png, &fake_icc).unwrap();
        let result = decode(&png_with_icc).unwrap();
        assert_eq!(result.icc_profile, Some(fake_icc));
    }

    #[test]
    fn decode_as_preserves_icc_profile() {
        let jpeg = make_jpeg(8, 8);
        let fake_icc = vec![55u8; 64];
        let jpeg_with_icc =
            crate::domain::encoder::jpeg::embed_icc_profile(&jpeg, &fake_icc).unwrap();
        let result = decode_as(&jpeg_with_icc, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.icc_profile, Some(fake_icc));
    }
}
