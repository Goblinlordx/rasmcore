//! rasmcore-webp — Pure Rust VP8 lossy encoder for WebP images.
//!
//! Implements the VP8 bitstream format (RFC 6386) and WebP RIFF container.
//! No C dependencies — compiles to any target including wasm32-wasip2.
//!
//! # Architecture
//!
//! Each module is a reusable component with a clean public API:
//! - [`boolcoder`] — Boolean arithmetic encoder (general-purpose)
//! - [`dct`] — 4×4 integer DCT/IDCT and Walsh-Hadamard Transform
//! - [`color`] — RGB↔YUV420 conversion
//! - [`predict`] — VP8 intra-prediction modes
//! - [`quant`] — Quantization tables and quality mapping
//!
//! # Usage
//!
//! ```no_run
//! use rasmcore_webp::{encode, EncodeConfig, PixelFormat};
//!
//! let pixels: Vec<u8> = vec![0; 64 * 64 * 3]; // 64x64 RGB8
//! let config = EncodeConfig::default(); // quality 75
//! let webp = encode(&pixels, 64, 64, PixelFormat::Rgb8, &config).unwrap();
//! ```

// Encoder pipeline modules (public for reuse)
pub mod boolcoder;
pub mod color;
pub mod dct;
pub mod predict;
pub mod quant;
pub mod tables;

// Encoder assembly modules (internal for now)
pub mod bitstream;
pub mod block;
pub mod config;
pub mod container;
pub mod error;
pub mod filter;
pub mod ratecontrol;
pub mod token;

// Re-export public API
pub use config::EncodeConfig;
pub use error::EncodeError;

/// Pixel format of input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 3 bytes per pixel: R, G, B
    Rgb8,
    /// 4 bytes per pixel: R, G, B, A (alpha is discarded for lossy VP8)
    Rgba8,
}

/// Encode raw pixels to lossy WebP (VP8).
///
/// Accepts RGB8 or RGBA8 pixel data. Returns a complete WebP file
/// (RIFF container with VP8 chunk).
///
/// # Errors
///
/// Returns [`EncodeError::InvalidDimensions`] if width or height is zero.
/// Returns [`EncodeError::InvalidPixelData`] if pixel buffer length doesn't
/// match expected size for the given dimensions and format.
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(EncodeError::InvalidDimensions { width, height });
    }

    let bpp = match format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
    };
    let expected = (width as usize) * (height as usize) * bpp;
    if pixels.len() != expected {
        return Err(EncodeError::InvalidPixelData {
            expected,
            actual: pixels.len(),
        });
    }

    // Validate quality
    let _quality = config.quality.clamp(1, 100);

    // TODO: Full encoding pipeline will be implemented in subsequent tracks:
    // 1. RGB → YUV420 conversion (color.rs)
    // 2. Macroblock partitioning (block.rs)
    // 3. Per-macroblock: prediction → residual → DCT → quantize → bool-encode
    // 4. Bitstream assembly (bitstream.rs)
    // 5. RIFF/WebP container wrapping (container.rs)
    Err(EncodeError::EncodeFailed(
        "VP8 encoder not yet implemented — scaffold only".into(),
    ))
}
