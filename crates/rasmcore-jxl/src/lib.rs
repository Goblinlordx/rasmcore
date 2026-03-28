//! rasmcore-jxl — Pure Rust JPEG XL encoder (ISO/IEC 18181).
//!
//! Scaffold for a future pure-Rust JXL encoder, following the same pattern
//! as rasmcore-webp. Decode is handled by jxl-oxide in rasmcore-image.
//!
//! The encoder will implement:
//! - ANS entropy coding
//! - Modular mode (lossless)
//! - VarDCT mode (lossy)
//! - Progressive encoding

pub mod config;
pub mod error;

pub use config::JxlEncodeConfig;
pub use error::EncodeError;

/// Pixel format of input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Gray8,
}

/// Encode raw pixels to JPEG XL.
///
/// # Status
///
/// This encoder is not yet implemented. Returns `EncodeError::NotYetImplemented`.
/// Future tracks will implement the encoding pipeline incrementally.
pub fn encode(
    _pixels: &[u8],
    _width: u32,
    _height: u32,
    _format: PixelFormat,
    _config: &JxlEncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    Err(EncodeError::NotYetImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_returns_not_yet_implemented() {
        let pixels = vec![0u8; 16 * 16 * 3];
        let config = JxlEncodeConfig::default();
        let result = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config);
        assert!(matches!(result, Err(EncodeError::NotYetImplemented)));
    }
}
