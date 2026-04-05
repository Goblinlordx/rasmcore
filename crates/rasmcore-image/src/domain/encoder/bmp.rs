use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// BMP encode configuration. BMP has no meaningful encode parameters.
#[derive(Debug, Clone, Default)]
pub struct BmpEncodeConfig;

/// Encode pixel data to BMP using the native rasmcore-bmp encoder.
pub fn encode_pixels(
    pixels: &[u8],
    info: &ImageInfo,
    _config: &BmpEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    super::native_trivial::encode_bmp(pixels, info)
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "bmp",
        format: "bmp",
        mime: "image/bmp",
        extensions: &["bmp"],
        fn_name: "encode_bmp",
        encode_fn: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    #[test]
    fn encode_produces_valid_bmp() {
        let pixels: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &BmpEncodeConfig).unwrap();
        assert_eq!(&result[..2], b"BM");
    }

    #[test]
    fn encode_rgba8_unsupported() {
        // BMP native encoder does not support RGBA8 — returns error
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_pixels(&pixels, &info, &BmpEncodeConfig);
        assert!(result.is_err());
    }
}
