//! JPEG encoder/decoder error types.

/// Errors from JPEG encode/decode operations.
#[derive(Debug, Clone)]
pub enum EncodeError {
    /// Feature not yet implemented (stub).
    NotYetImplemented,
    /// Invalid input data or parameters.
    InvalidInput(String),
    /// Encoding failed.
    EncodeFailed(String),
    /// Decoding failed.
    DecodeFailed(String),
    /// Unsupported JPEG feature (e.g., 12-bit, arithmetic in baseline).
    Unsupported(String),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotYetImplemented => write!(f, "not yet implemented"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::EncodeFailed(msg) => write!(f, "encode failed: {msg}"),
            Self::DecodeFailed(msg) => write!(f, "decode failed: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for EncodeError {}
