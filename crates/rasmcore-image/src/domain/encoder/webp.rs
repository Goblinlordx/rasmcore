use image::DynamicImage;

use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

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

/// Encode raw pixel data to WebP.
///
/// Dispatches to rasmcore-webp (lossy) or image crate (lossless) based on config.
/// Accepts RGB8 or RGBA8 pixel data directly.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    config: &WebpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    if config.lossless {
        // Lossless: use image crate's WebP encoder (handles alpha, pixel-exact)
        let img = super::pixels_to_dynamic_image(pixels, info)?;
        encode_lossless(&img)
    } else {
        // Lossy: use our rasmcore-webp VP8 encoder
        encode_lossy(pixels, info, config)
    }
}

/// Encode a DynamicImage to WebP (convenience wrapper for pipeline sink compatibility).
pub fn encode(
    img: &DynamicImage,
    info: &ImageInfo,
    config: &WebpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    if config.lossless {
        encode_lossless(img)
    } else {
        // Extract raw pixels and dispatch to lossy encoder
        let (pixels, pixel_format) = match img {
            image::DynamicImage::ImageRgb8(buf) => (buf.as_raw().as_slice(), PixelFormat::Rgb8),
            image::DynamicImage::ImageRgba8(buf) => (buf.as_raw().as_slice(), PixelFormat::Rgba8),
            _ => {
                let rgb = img.to_rgb8();
                let adjusted_info = ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..*info
                };
                return encode_lossy(rgb.as_raw(), &adjusted_info, config);
            }
        };
        let adjusted_info = ImageInfo {
            format: pixel_format,
            ..*info
        };
        encode_lossy(pixels, &adjusted_info, config)
    }
}

/// Lossy encoding via rasmcore-webp (our pure Rust VP8 encoder).
fn encode_lossy(
    pixels: &[u8],
    info: &ImageInfo,
    config: &WebpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let webp_format = match info.format {
        PixelFormat::Rgb8 => rasmcore_webp::PixelFormat::Rgb8,
        PixelFormat::Rgba8 => rasmcore_webp::PixelFormat::Rgba8,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "WebP lossy encoding from {other:?} not supported — convert to RGB8 or RGBA8 first"
            )));
        }
    };

    let webp_config = rasmcore_webp::EncodeConfig {
        quality: config.quality,
    };

    rasmcore_webp::encode(pixels, info.width, info.height, webp_format, &webp_config)
        .map_err(|e| ImageError::ProcessingFailed(format!("WebP lossy encode failed: {e}")))
}

/// Lossless encoding via image crate (preserves all pixel data exactly).
fn encode_lossless(img: &DynamicImage) -> Result<Vec<u8>, ImageError> {
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
    use crate::domain::types::ColorSpace;

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

    fn make_test_image(w: u32, h: u32) -> (DynamicImage, ImageInfo) {
        let (pixels, info) = make_test_pixels(w, h);
        let img = pixels_to_dynamic_image(&pixels, &info).unwrap();
        (img, info)
    }

    #[test]
    fn encode_lossy_produces_valid_webp() {
        let (pixels, info) = make_test_pixels(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let result = encode_pixels(&pixels, &info, &config).unwrap();
        // WebP RIFF header: "RIFF" ... "WEBP"
        assert_eq!(&result[..4], b"RIFF");
        assert_eq!(&result[8..12], b"WEBP");
    }

    #[test]
    fn encode_lossless_produces_output() {
        let (pixels, info) = make_test_pixels(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: true,
        };
        let result = encode_pixels(&pixels, &info, &config).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn encode_via_dynamic_image_lossy() {
        let (img, info) = make_test_image(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let result = encode(&img, &info, &config).unwrap();
        assert_eq!(&result[..4], b"RIFF");
        assert_eq!(&result[8..12], b"WEBP");
    }

    #[test]
    fn encode_via_dynamic_image_lossless() {
        let (img, info) = make_test_image(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: true,
        };
        let result = encode(&img, &info, &config).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn lossy_quality_affects_size() {
        let (pixels, info) = make_test_pixels(64, 64);
        let low_q = encode_pixels(
            &pixels,
            &info,
            &WebpEncodeConfig {
                quality: 10,
                lossless: false,
            },
        )
        .unwrap();
        let high_q = encode_pixels(
            &pixels,
            &info,
            &WebpEncodeConfig {
                quality: 95,
                lossless: false,
            },
        )
        .unwrap();
        // Higher quality should produce larger output
        assert!(
            high_q.len() > low_q.len(),
            "q95 ({}) should be larger than q10 ({})",
            high_q.len(),
            low_q.len()
        );
    }

    #[test]
    fn lossy_output_decodable() {
        let (pixels, info) = make_test_pixels(32, 32);
        let encoded = encode_pixels(
            &pixels,
            &info,
            &WebpEncodeConfig {
                quality: 75,
                lossless: false,
            },
        )
        .unwrap();

        // Decode with image crate to verify it's valid WebP
        let decoded = image::load_from_memory(&encoded).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    #[test]
    fn lossy_deterministic() {
        let (pixels, info) = make_test_pixels(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let out1 = encode_pixels(&pixels, &info, &config).unwrap();
        let out2 = encode_pixels(&pixels, &info, &config).unwrap();
        assert_eq!(out1, out2, "lossy encoding should be deterministic");
    }

    #[test]
    fn encode_rgba8_lossy() {
        let pixels: Vec<u8> = (0..(16 * 16 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let result = encode_pixels(&pixels, &info, &config).unwrap();
        assert_eq!(&result[..4], b"RIFF");
    }

    #[test]
    fn encode_1x1_lossy() {
        let pixels = vec![128u8, 64, 32];
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let result = encode_pixels(&pixels, &info, &config).unwrap();
        assert_eq!(&result[..4], b"RIFF");
    }

    #[test]
    fn default_config() {
        let config = WebpEncodeConfig::default();
        assert_eq!(config.quality, 75);
        assert!(!config.lossless);
    }
}
