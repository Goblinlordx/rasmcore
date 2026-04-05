use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

/// AVIF encode configuration.
#[derive(Debug, Clone)]
pub struct AvifEncodeConfig {
    /// Quality level 1-100 (default: 75).
    pub quality: u8,
    /// Speed preset 1-10 (default: 6).
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

/// AVIF encoding is not currently available.
///
/// The rav1e/ravif dependency (50+ crates) has been removed to reduce binary
/// size. Native AV1 encoding is a planned future capability.
pub fn encode(
    _pixels: &[u8],
    _info: &ImageInfo,
    _config: &AvifEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    Err(ImageError::UnsupportedFormat(
        "AVIF encoding not available — rav1e dependency removed for binary size. \
         Native AV1 encoding is a planned future capability."
            .into(),
    ))
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "avif",
        format: "avif",
        mime: "image/avif",
        extensions: &["avif"],
        fn_name: "encode_avif",
        encode_fn: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, PixelFormat};

    #[test]
    fn encode_returns_unsupported() {
        let pixels = vec![0u8; 16 * 16 * 3];
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode(&pixels, &info, &AvifEncodeConfig::default());
        assert!(result.is_err());
    }
}
