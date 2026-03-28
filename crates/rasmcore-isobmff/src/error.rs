//! Error types for ISOBMFF parsing.

use core::fmt;

/// Errors that can occur during ISOBMFF parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsobmffError {
    /// Data is too short to contain the expected structure.
    Truncated {
        expected: usize,
        available: usize,
    },
    /// Box header has an invalid size (e.g., size < 8 for a normal box).
    InvalidBoxSize {
        box_type: [u8; 4],
        size: u64,
    },
    /// Required box is missing from the file.
    MissingBox {
        box_type: [u8; 4],
    },
    /// Box version is not supported by this parser.
    UnsupportedVersion {
        box_type: [u8; 4],
        version: u8,
    },
    /// The file does not start with an ftyp box or has no recognized brand.
    NotIsobmff,
    /// A box nesting or structure constraint was violated.
    InvalidStructure(String),
}

impl fmt::Display for IsobmffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                expected,
                available,
            } => {
                write!(f, "truncated: expected {expected} bytes, got {available}")
            }
            Self::InvalidBoxSize { box_type, size } => {
                let name = fourcc_str(box_type);
                write!(f, "invalid box size for '{name}': {size}")
            }
            Self::MissingBox { box_type } => {
                let name = fourcc_str(box_type);
                write!(f, "missing required box: '{name}'")
            }
            Self::UnsupportedVersion {
                box_type,
                version,
            } => {
                let name = fourcc_str(box_type);
                write!(f, "unsupported version {version} for box '{name}'")
            }
            Self::NotIsobmff => write!(f, "not an ISOBMFF file"),
            Self::InvalidStructure(msg) => write!(f, "invalid structure: {msg}"),
        }
    }
}

impl std::error::Error for IsobmffError {}

/// Format a FourCC as a string, replacing non-ASCII with '?'.
fn fourcc_str(cc: &[u8; 4]) -> String {
    cc.iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '?'
            }
        })
        .collect()
}
