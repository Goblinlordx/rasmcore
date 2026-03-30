//! Pure Rust DNG RAW image decoder.
//!
//! Decodes DNG (Digital Negative) files — Adobe's open RAW format based on TIFF/EP.
//! Implements the full pipeline: TIFF container parsing → raw sensor extraction →
//! Bayer CFA demosaicing → color matrix application → sRGB output.
//!
//! Zero external dependencies. WASM-ready.
//!
//! # Supported features
//!
//! - Uncompressed and lossless JPEG (ITU-T T.81 Annex H) DNG files
//! - All 4 Bayer CFA patterns: RGGB, BGGR, GRBG, GBRG
//! - 8/12/14/16-bit sensor data
//! - ColorMatrix1 → XYZ → sRGB color pipeline
//! - AsShotNeutral white balance
//! - BlackLevel/WhiteLevel normalization
//! - Strip and tile data layouts
//!
//! # Out of scope
//!
//! - Proprietary RAW formats (CR2, NEF, ARW, etc.)
//! - DNG opcodes, profiles, lens correction
//! - HDR/floating-point DNG
//! - DNG encoding (write)

use std::fmt;

mod color;
mod demosaic;
mod dng;
mod ljpeg;
mod tiff;

/// RAW decode error.
#[derive(Debug)]
pub enum RawError {
    InvalidFormat(String),
    UnsupportedCompression(u16),
    DataTruncated,
}

impl fmt::Display for RawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(m) => write!(f, "invalid DNG: {m}"),
            Self::UnsupportedCompression(c) => write!(f, "unsupported compression: {c}"),
            Self::DataTruncated => write!(f, "DNG data truncated"),
        }
    }
}

impl std::error::Error for RawError {}

/// Decoded DNG image data.
pub struct DngImage {
    /// RGB pixel data (interleaved). RGB8 for 8-bit output, RGB16 LE for 16-bit.
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Original sensor bit depth (8, 12, 14, or 16).
    pub bits_per_sample: u16,
    /// True if pixels are 16-bit (6 bytes per pixel), false if 8-bit (3 bytes per pixel).
    pub is_16bit: bool,
}

/// Check if data is a DNG file (TIFF header + DNGVersion tag).
pub fn is_dng(data: &[u8]) -> bool {
    dng::is_dng(data)
}

/// Decode a DNG file to 8-bit sRGB.
///
/// Returns `DngImage` with RGB8 pixel data.
pub fn decode(data: &[u8]) -> Result<DngImage, RawError> {
    let result = dng::decode_dng(data)?;
    Ok(DngImage {
        pixels: result.pixels,
        width: result.width,
        height: result.height,
        bits_per_sample: result.bits_per_sample,
        is_16bit: false,
    })
}

/// Decode a DNG file to 16-bit sRGB.
///
/// Returns `DngImage` with RGB16 pixel data (little-endian).
pub fn decode_16bit(data: &[u8]) -> Result<DngImage, RawError> {
    let result = dng::decode_dng_16bit(data)?;
    Ok(DngImage {
        pixels: result.pixels,
        width: result.width,
        height: result.height,
        bits_per_sample: result.bits_per_sample,
        is_16bit: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_dng_rejects_empty() {
        assert!(!is_dng(&[]));
    }

    #[test]
    fn is_dng_rejects_short() {
        assert!(!is_dng(&[0xFF, 0xD8]));
    }

    #[test]
    fn error_display() {
        let e = RawError::InvalidFormat("test".into());
        assert_eq!(format!("{e}"), "invalid DNG: test");
        let e = RawError::UnsupportedCompression(42);
        assert_eq!(format!("{e}"), "unsupported compression: 42");
        let e = RawError::DataTruncated;
        assert_eq!(format!("{e}"), "DNG data truncated");
    }
}
