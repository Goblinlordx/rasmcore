//! Pure Rust JPEG encoder/decoder — ITU-T T.81.
//!
//! Implements baseline, extended, and progressive JPEG encoding/decoding
//! with both Huffman and arithmetic entropy coding.
//!
//! Uses shared infrastructure:
//! - `rasmcore-bitio` for bit-level I/O
//! - `rasmcore-deflate::huffman` for Huffman coding
//! - `rasmcore-color` for RGB/YCbCr conversion
//!
//! # Architecture
//!
//! ```text
//! Encode pipeline:
//!   pixels → color convert → block split → DCT → quantize → entropy code → bitstream → markers
//!
//! Decode pipeline:
//!   markers → bitstream → entropy decode → dequantize → IDCT → block merge → color convert → pixels
//! ```

mod color;
mod dct;
pub mod entropy;
mod error;
mod markers;
mod quantize;
mod types;

pub use error::EncodeError;
pub use types::*;

/// Encode raw pixels to JPEG.
///
/// Returns the encoded JPEG byte stream, or an error if encoding fails.
///
/// # Example
///
/// ```
/// use rasmcore_jpeg::{encode, EncodeConfig, PixelFormat};
///
/// let pixels = vec![128u8; 64 * 64 * 3]; // 64x64 gray RGB
/// let config = EncodeConfig::default();
/// // let jpeg_bytes = encode(&pixels, 64, 64, PixelFormat::Rgb8, &config).unwrap();
/// // (not yet implemented — returns EncodeError::NotYetImplemented)
/// ```
pub fn encode(
    _pixels: &[u8],
    _width: u32,
    _height: u32,
    _format: PixelFormat,
    _config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    Err(EncodeError::NotYetImplemented)
}

/// Decode JPEG data to raw pixels.
///
/// Returns decoded pixel data with image info, or an error.
pub fn decode(_data: &[u8]) -> Result<DecodedOutput, EncodeError> {
    Err(EncodeError::NotYetImplemented)
}

/// Decoded JPEG output.
pub struct DecodedOutput {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_returns_not_yet_implemented() {
        let pixels = vec![0u8; 8 * 8 * 3];
        let result = encode(&pixels, 8, 8, PixelFormat::Rgb8, &EncodeConfig::default());
        assert!(matches!(result, Err(EncodeError::NotYetImplemented)));
    }

    #[test]
    fn decode_returns_not_yet_implemented() {
        let result = decode(&[0xFF, 0xD8, 0xFF, 0xD9]);
        assert!(matches!(result, Err(EncodeError::NotYetImplemented)));
    }

    #[test]
    fn default_config_values() {
        let config = EncodeConfig::default();
        assert_eq!(config.quality, 85);
        assert!(!config.progressive);
        assert_eq!(config.subsampling, ChromaSubsampling::Quarter420);
        assert!(!config.arithmetic_coding);
        assert!(config.restart_interval.is_none());
        assert!(!config.optimize_huffman);
        assert!(!config.trellis);
        assert_eq!(config.sample_precision, SamplePrecision::Eight);
        assert!(config.custom_quant_tables.is_none());
    }
}
