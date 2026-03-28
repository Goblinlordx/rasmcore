#![allow(clippy::needless_range_loop)]
//! Pure Rust deflate/inflate implementation with shared Huffman coding.
//!
//! Provides:
//! - `pub mod huffman` — Huffman tree build/encode/decode, reusable by JPEG and other codecs
//! - `inflate()` — RFC 1951 decompression
//! - `deflate()` — RFC 1951 compression with configurable level
//! - `checksum` — CRC32 and Adler32 for PNG and zlib
//!
//! Zero external dependencies beyond `rasmcore-bitio`.

pub mod checksum;
mod deflate;
pub mod huffman;
mod inflate;

pub use deflate::deflate;
pub use inflate::inflate;

/// Deflate compression level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionLevel {
    /// No compression — stored blocks only.
    None,
    /// Fast compression — fixed Huffman tables, minimal LZ77 matching.
    Fast,
    #[default]
    /// Default compression — dynamic Huffman tables, hash-chain LZ77.
    Default,
    /// Best compression — exhaustive LZ77 matching, optimal Huffman.
    Best,
}


/// Errors from inflate/deflate operations.
#[derive(Debug, Clone)]
pub enum DeflateError {
    /// Input data is truncated or malformed.
    InvalidData(String),
    /// Checksum mismatch.
    ChecksumMismatch { expected: u32, actual: u32 },
    /// Unsupported feature in the deflate stream.
    Unsupported(String),
}

impl std::fmt::Display for DeflateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidData(msg) => write!(f, "invalid deflate data: {msg}"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "checksum mismatch: expected {expected:#010x}, got {actual:#010x}"
                )
            }
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for DeflateError {}
