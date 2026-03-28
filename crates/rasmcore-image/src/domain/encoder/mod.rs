pub mod avif;
pub mod bmp;
pub mod dds;
pub mod fits;
pub mod gif;
pub mod ico;
pub mod jp2;
pub mod jpeg;
pub mod png;
pub mod qoi;
pub mod tiff;
pub mod webp;

use image::DynamicImage;

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

const SUPPORTED_FORMATS: &[&str] = &[
    "png", "jpeg", "webp", "gif", "tiff", "avif", "bmp", "ico", "qoi", "tga", "hdr", "pnm", "exr",
    "dds", "jp2", "fits",
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
            let config = webp::WebpEncodeConfig {
                quality: quality.unwrap_or(75),
                lossless: false,
            };
            webp::encode_pixels(pixels, info, &config)
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
        "bmp" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            bmp::encode(&img, info, &bmp::BmpEncodeConfig)
        }
        "ico" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            ico::encode(&img, info, &ico::IcoEncodeConfig)
        }
        "qoi" => {
            let img = pixels_to_dynamic_image(pixels, info)?;
            qoi::encode(&img, info, &qoi::QoiEncodeConfig)
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
        "jp2" | "j2k" | "jpeg2000" => {
            let config = jp2::Jp2EncodeConfig::default();
            jp2::encode(pixels, info, &config)
        }
        "fits" | "fit" => fits::encode_pixels(pixels, info),
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
        let result = encode(&pixels, &info, "svg", None);
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
    #[test]
    fn encode_avif_produces_valid_avif() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "avif", Some(50)).unwrap();
        assert_eq!(&result[4..8], b"ftyp");
    }

    #[test]
    fn encode_avif_rgba8() {
        let (pixels, info) = make_rgba8_pixels(16, 16);
        let result = encode(&pixels, &info, "avif", None);
        assert!(result.is_ok());
    }

    #[test]
    fn roundtrip_avif_preserves_dimensions() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let encoded = encode(&pixels, &info, "avif", Some(80)).unwrap();
        // AVIF decode may not be available in all builds (needs avif-native feature)
        match crate::domain::decoder::decode(&encoded) {
            Ok(decoded) => {
                assert_eq!(decoded.info.width, 16);
                assert_eq!(decoded.info.height, 16);
            }
            Err(_) => {
                // AVIF decode not available in this build, just verify encode worked
                assert_eq!(&encoded[4..8], b"ftyp");
            }
        }
    }

    #[test]
    fn avif_roundtrip_quality_psnr() {
        let (pixels, info) = make_rgb8_pixels(32, 32);
        let encoded = encode(&pixels, &info, "avif", Some(80)).unwrap();
        // AVIF decode may not be available
        let decoded = match crate::domain::decoder::decode_as(&encoded, PixelFormat::Rgb8) {
            Ok(d) => d,
            Err(_) => return, // Skip PSNR test if decode unavailable
        };
        let mse: f64 = pixels
            .iter()
            .zip(decoded.pixels.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / pixels.len() as f64;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        };
        assert!(
            psnr > 25.0,
            "AVIF roundtrip PSNR too low: {psnr:.1}dB (expected >25dB at quality 80)"
        );
    }

    #[test]
    fn supported_formats_lists_expected() {
        let fmts = supported_formats();
        for f in [
            "png", "jpeg", "webp", "gif", "tiff", "avif", "bmp", "ico", "qoi", "tga", "hdr", "pnm",
            "exr", "dds", "jp2",
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

    // ---- BMP/ICO/QOI encoder tests ----

    #[test]
    fn encode_bmp_produces_valid_bmp() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "bmp", None).unwrap();
        assert_eq!(&result[..2], b"BM");
    }

    #[test]
    fn roundtrip_bmp_preserves_pixels() {
        let (pixels, info) = make_rgb8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "bmp", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
        // BMP is lossless — pixels should match exactly
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn encode_ico_produces_valid_ico() {
        let (pixels, info) = make_rgba8_pixels(16, 16);
        let result = encode(&pixels, &info, "ico", None).unwrap();
        // ICO magic: 00 00 01 00
        assert_eq!(&result[..4], &[0x00, 0x00, 0x01, 0x00]);
    }

    #[test]
    fn roundtrip_ico_preserves_dimensions() {
        let (pixels, info) = make_rgba8_pixels(32, 32);
        let encoded = encode(&pixels, &info, "ico", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 32);
        assert_eq!(decoded.info.height, 32);
    }

    #[test]
    fn encode_qoi_produces_valid_qoi() {
        let (pixels, info) = make_rgb8_pixels(16, 16);
        let result = encode(&pixels, &info, "qoi", None).unwrap();
        assert_eq!(&result[..4], b"qoif");
    }

    #[test]
    fn roundtrip_qoi_preserves_pixels() {
        let (pixels, info) = make_rgba8_pixels(8, 8);
        let encoded = encode(&pixels, &info, "qoi", None).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 8);
        assert_eq!(decoded.info.height, 8);
        // QOI is lossless
        assert_eq!(decoded.pixels, pixels);
    }
}
