pub mod avif;
pub mod dds;
pub mod gif;
pub mod jpeg;
pub mod png;
pub mod tiff;
pub mod webp;

use image::DynamicImage;

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

const SUPPORTED_FORMATS: &[&str] = &[
    "png", "jpeg", "webp", "gif", "tiff", "avif", "tga", "hdr", "pnm", "exr", "dds",
];

/// Encode pixel data to a specific image format (convenience wrapper).
///
/// Dispatches to per-format encoders with default configs. For fine-grained
/// control, use the per-format encode functions directly (e.g., `jpeg::encode`).
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, ImageError> {
    match format {
        "jpeg" | "jpg" => {
            let config = jpeg::JpegEncodeConfig {
                quality: quality.unwrap_or(85),
                progressive: false,
            };
            jpeg::encode_pixels(pixels, info, &config)
        }
        "png" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            let config = png::PngEncodeConfig::default();
            png::encode(&img, info, &config)
        }
        "webp" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            let config = webp::WebpEncodeConfig::default();
            webp::encode(&img, info, &config)
        }
        "gif" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            let config = gif::GifEncodeConfig::default();
            gif::encode(&img, info, &config)
        }
        "avif" => {
            let config = avif::AvifEncodeConfig {
                quality: quality.unwrap_or(75),
                ..Default::default()
            };
            avif::encode(pixels, info, &config)
        }
        "tiff" | "tif" => {
            let config = tiff::TiffEncodeConfig::default();
            tiff::encode(pixels, info, &config)
        }
        "tga" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            encode_via_image_format(&img, image::ImageFormat::Tga)
        }
        "hdr" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            let rgb32f = DynamicImage::ImageRgb32F(img.to_rgb32f());
            encode_via_image_format(&rgb32f, image::ImageFormat::Hdr)
        }
        "pnm" | "ppm" | "pgm" | "pbm" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            encode_via_image_format(&img, image::ImageFormat::Pnm)
        }
        "exr" | "openexr" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            let rgba32f = DynamicImage::ImageRgba32F(img.to_rgba32f());
            encode_via_image_format(&rgba32f, image::ImageFormat::OpenExr)
        }
        "dds" => dds::encode_dds(pixels, info),
        other => Err(ImageError::UnsupportedFormat(format!(
            "encode format '{other}' not supported"
        ))),
    }
}

/// List supported encode formats.
pub fn supported_formats() -> Vec<String> {
    SUPPORTED_FORMATS.iter().map(|s| String::from(*s)).collect()
}

/// Encode a DynamicImage using the image crate's built-in format encoder.
/// Used for formats that don't need custom config (TGA, HDR, PNM, EXR).
fn encode_via_image_format(
    img: &DynamicImage,
    format: image::ImageFormat,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, format)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}

/// Convert raw pixel data to a DynamicImage for encoding.
pub fn pixels_to_dynamic_image(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<DynamicImage, ImageError> {
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
        assert_eq!(&result[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn encode_jpeg_produces_valid_jpeg() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "jpeg", Some(90)).unwrap();
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
    fn encode_gif_produces_valid_gif() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "gif", None).unwrap();
        assert_eq!(&result[..3], b"GIF");
    }

    #[test]
    fn encode_gif_rgba8() {
        let (pixels, info) = make_rgba8_pixels(8, 8);
        let result = encode(&pixels, &info, "gif", None);
        assert!(result.is_ok());
    }

    #[test]
    fn roundtrip_gif_preserves_dimensions() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "gif", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
    }

    #[test]
    fn supported_formats_lists_expected() {
        let fmts = supported_formats();
        for f in [
            "png", "jpeg", "webp", "gif", "tga", "hdr", "pnm", "exr", "dds",
        ] {
            assert!(fmts.contains(&f.to_string()), "missing format: {f}");
        }
    }

    // ---- New trivial format tests ----

    #[test]
    fn encode_tga_produces_output() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "tga", None);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn roundtrip_tga_preserves_dimensions() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "tga", None).unwrap();
        // TGA has no magic bytes — image::guess_format may fail.
        // Use image crate directly with explicit format hint.
        let img = image::load_from_memory_with_format(&encoded, image::ImageFormat::Tga).unwrap();
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
    }

    #[test]
    fn encode_hdr_produces_output() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "hdr", None);
        assert!(result.is_ok());
        let data = result.unwrap();
        // HDR files start with "#?RADIANCE" or "#?RGBE"
        assert!(
            data.starts_with(b"#?RADIANCE") || data.starts_with(b"#?RGBE"),
            "HDR should start with Radiance signature"
        );
    }

    #[test]
    fn roundtrip_hdr_preserves_dimensions() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "hdr", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
    }

    #[test]
    fn encode_pnm_produces_output() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "pnm", None);
        assert!(result.is_ok());
        let data = result.unwrap();
        // PPM starts with "P6" (binary) or "P3" (ASCII)
        assert!(data[0] == b'P', "PNM should start with 'P'");
    }

    #[test]
    fn roundtrip_pnm_preserves_pixels() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "pnm", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
        // PNM is lossless
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn encode_exr_produces_output() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "exr", None);
        assert!(result.is_ok());
        let data = result.unwrap();
        // EXR magic: 0x76, 0x2F, 0x31, 0x01
        assert_eq!(&data[..4], &[0x76, 0x2F, 0x31, 0x01]);
    }

    #[test]
    fn roundtrip_exr_preserves_dimensions() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "exr", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
    }

    #[test]
    fn encode_dds_produces_output() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "dds", None);
        assert!(result.is_ok());
        let data = result.unwrap();
        assert_eq!(&data[..4], b"DDS ");
    }

    #[test]
    fn roundtrip_dds_header_and_pixels() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "dds", None).unwrap();
        // Verify DDS magic
        assert_eq!(&encoded[..4], b"DDS ");
        // Read dimensions from header (offset 12 = height, 16 = width)
        let h = u32::from_le_bytes([encoded[12], encoded[13], encoded[14], encoded[15]]);
        let w = u32::from_le_bytes([encoded[16], encoded[17], encoded[18], encoded[19]]);
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        // Pixel data starts at offset 128, should be 8*8*4 = 256 bytes of BGRA
        assert_eq!(encoded.len(), 128 + 8 * 8 * 4);
        // Verify first pixel: RGB input (0,1,2) -> BGRA (2,1,0,255)
        assert_eq!(encoded[128], 2); // B
        assert_eq!(encoded[129], 1); // G
        assert_eq!(encoded[130], 0); // R
        assert_eq!(encoded[131], 255); // A
    }

    #[test]
    fn encode_tga_rgba8() {
        let (pixels, info) = make_rgba8_pixels(8, 8);
        let result = encode(&pixels, &info, "tga", None);
        assert!(result.is_ok());
    }

    #[test]
    fn encode_dds_rgba8() {
        let (pixels, info) = make_rgba8_pixels(8, 8);
        let result = encode(&pixels, &info, "dds", None);
        assert!(result.is_ok());
    }
}
