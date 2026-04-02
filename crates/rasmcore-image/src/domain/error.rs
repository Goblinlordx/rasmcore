/// Domain-level errors for image processing operations.
/// These are independent of WIT — the adapter layer translates them.
#[derive(Debug)]
pub enum ImageError {
    /// Input data is invalid or corrupt
    InvalidInput(String),
    /// Requested format is not supported
    UnsupportedFormat(String),
    /// Operation is not yet implemented
    NotImplemented,
    /// Encoding/decoding failed
    ProcessingFailed(String),
    /// Invalid parameters (e.g., crop out of bounds)
    InvalidParameters(String),
    /// Script plugin error (Rhai runtime error, validation failure, OOB access)
    ScriptError(String),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::NotImplemented => write!(f, "not implemented"),
            Self::ProcessingFailed(msg) => write!(f, "processing failed: {msg}"),
            Self::InvalidParameters(msg) => write!(f, "invalid parameters: {msg}"),
            Self::ScriptError(msg) => write!(f, "script error: {msg}"),
        }
    }
}

impl std::error::Error for ImageError {}
