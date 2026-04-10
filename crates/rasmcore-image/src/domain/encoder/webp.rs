use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// WebP encode configuration.
#[derive(Debug, Clone)]
pub struct WebpEncodeConfig {
    /// Quality level 1-100 for lossy mode (default: 75).
    pub quality: u8,
    /// Use lossless encoding (default: false).
    /// Note: lossless WebP is not yet natively implemented.
    /// Falls back to lossy at quality 100 when requested.
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

/// Encode raw pixel data to WebP using rasmcore-webp (VP8 lossy).
///
/// Accepts RGB8 or RGBA8 pixel data directly.
/// Lossless mode is not yet natively implemented — falls back to
/// lossy at quality 100.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    config: &WebpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let quality = if config.lossless { 100 } else { config.quality };
    encode_lossy(pixels, info, quality)
}

fn encode_lossy(pixels: &[u8], info: &ImageInfo, quality: u8) -> Result<Vec<u8>, ImageError> {
    let webp_format = match info.format {
        PixelFormat::Rgb8 => rasmcore_webp::PixelFormat::Rgb8,
        PixelFormat::Rgba8 => rasmcore_webp::PixelFormat::Rgba8,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "WebP encode requires RGB8 or RGBA8".into(),
            ));
        }
    };

    let webp_config = rasmcore_webp::EncodeConfig { quality };

    rasmcore_webp::encode(pixels, info.width, info.height, webp_format, &webp_config)
        .map_err(|e| ImageError::ProcessingFailed(format!("WebP lossy encode failed: {e}")))
}

// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "webp",
        format: "webp",
        mime: "image/webp",
        extensions: &["webp"],
        fn_name: "encode_webp",
        encode_fn: None,
        preferred_output_cs: crate::domain::encoder::EncoderColorSpace::Srgb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn default_config() {
        let config = WebpEncodeConfig::default();
        assert_eq!(config.quality, 75);
        assert!(!config.lossless);
    }

    #[test]
    fn encode_1x1_lossy() {
        let pixels = vec![128u8, 128, 128];
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
        assert!(result.starts_with(b"RIFF"));
    }

    #[test]
    fn encode_lossy_produces_valid_webp() {
        let (pixels, info) = make_test_pixels(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let result = encode_pixels(&pixels, &info, &config).unwrap();
        assert!(result.starts_with(b"RIFF"));
        assert_eq!(&result[8..12], b"WEBP");
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
            quality: 85,
            lossless: false,
        };
        let result = encode_pixels(&pixels, &info, &config).unwrap();
        assert!(result.starts_with(b"RIFF"));
    }

    #[test]
    fn lossy_deterministic() {
        let (pixels, info) = make_test_pixels(16, 16);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let a = encode_pixels(&pixels, &info, &config).unwrap();
        let b = encode_pixels(&pixels, &info, &config).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn lossy_quality_affects_size() {
        let (pixels, info) = make_test_pixels(32, 32);
        let low = encode_pixels(
            &pixels,
            &info,
            &WebpEncodeConfig {
                quality: 30,
                lossless: false,
            },
        )
        .unwrap();
        let high = encode_pixels(
            &pixels,
            &info,
            &WebpEncodeConfig {
                quality: 95,
                lossless: false,
            },
        )
        .unwrap();
        assert!(
            high.len() > low.len(),
            "higher quality should produce larger output"
        );
    }

    #[test]
    fn lossy_output_decodable() {
        let (pixels, info) = make_test_pixels(32, 32);
        let config = WebpEncodeConfig {
            quality: 75,
            lossless: false,
        };
        let encoded = encode_pixels(&pixels, &info, &config).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 32);
        assert_eq!(decoded.info.height, 32);
    }
}
