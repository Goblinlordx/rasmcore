use std::fmt;

/// JPEG XL encoding error.
#[derive(Debug)]
pub enum EncodeError {
    /// Encoder not yet implemented.
    NotYetImplemented,
    /// Image dimensions are invalid.
    InvalidDimensions { width: u32, height: u32 },
    /// Pixel buffer size mismatch.
    InvalidPixelData { expected: usize, actual: usize },
    /// Encoding failed.
    EncodeFailed(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotYetImplemented => write!(f, "JXL encoder not yet implemented"),
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
