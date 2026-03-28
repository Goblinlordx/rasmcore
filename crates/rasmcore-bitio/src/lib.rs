//! Bit-level I/O for codec bitstreams.
//!
//! Provides `BitWriter` and `BitReader` with configurable bit ordering:
//! - **MSB-first**: used by Huffman coding (JPEG, Deflate/PNG)
//! - **LSB-first**: used by LZW (GIF, TIFF)
//!
//! Designed as shared infrastructure for all rasmcore native codecs.
//! Zero dependencies, WASM-ready.
//!
//! # Performance
//!
//! Uses a u64 accumulator with batch byte flushing for throughput.
//! Minimizes branch mispredictions in the inner write loop.

mod reader;
mod writer;

pub use reader::BitReader;
pub use writer::BitWriter;

/// Bit ordering within bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    /// Most significant bit first. Used by Huffman (JPEG, Deflate/PNG).
    MsbFirst,
    /// Least significant bit first. Used by LZW (GIF, TIFF).
    LsbFirst,
}
