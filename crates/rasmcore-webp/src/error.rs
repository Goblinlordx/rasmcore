use std::fmt;

/// VP8 encoding error.
#[derive(Debug)]
pub enum EncodeError {
    /// Image dimensions are invalid (zero width or height).
    InvalidDimensions { width: u32, height: u32 },
    /// Pixel buffer size doesn't match expected size for dimensions and format.
    InvalidPixelData { expected: usize, actual: usize },
    /// Encoding failed.
    EncodeFailed(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
            Self::InvalidPixelData { expected, actual } => {
                write!(
                    f,
                    "pixel data mismatch: expected {expected} bytes, got {actual}"
                )
            }
            Self::EncodeFailed(msg) => write!(f, "encode failed: {msg}"),
        }
    }
}

impl std::error::Error for EncodeError {}
