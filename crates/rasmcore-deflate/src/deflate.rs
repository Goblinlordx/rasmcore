//! RFC 1951 deflate (compression).
//!
//! Supports multiple compression levels:
//! - None: stored blocks (no compression)
//! - Fast: fixed Huffman with minimal LZ77
//! - Default: dynamic Huffman with hash-chain LZ77
//! - Best: exhaustive LZ77 matching

use rasmcore_bitio::{BitOrder, BitWriter};

use crate::huffman::HuffmanEncoder;
use crate::{CompressionLevel, DeflateError};

/// Deflate (compress) data according to RFC 1951.
///
/// Returns the raw deflate bitstream (not zlib or gzip wrapped).
pub fn deflate(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>, DeflateError> {
    match level {
        CompressionLevel::None => deflate_stored(data),
        CompressionLevel::Fast => deflate_fixed(data),
        CompressionLevel::Default | CompressionLevel::Best => deflate_fixed(data), // TODO: dynamic Huffman + LZ77
    }
}

/// Stored blocks — no compression (BTYPE=00).
fn deflate_stored(data: &[u8]) -> Result<Vec<u8>, DeflateError> {
    let mut writer = BitWriter::new(BitOrder::LsbFirst);

    let chunks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(65535).collect()
    };

    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;

        // BFINAL + BTYPE=00
        writer.write_bit(is_last);
        writer.write_bits(2, 0); // stored

        // Align to byte boundary
        writer.align_to_byte();

        // LEN and NLEN
        let len = chunk.len() as u16;
        writer.write_bits(16, len as u32);
        writer.write_bits(16, (!len) as u32);

        // Raw data
        for &byte in *chunk {
            writer.write_bits(8, byte as u32);
        }
    }

    Ok(writer.finish())
}

/// Fixed Huffman — BTYPE=01, uses standard fixed code tables.
///
/// This is a middle ground: better than stored, simpler than dynamic.
/// No LZ77 matching — just literal encoding with fixed Huffman codes.
fn deflate_fixed(data: &[u8]) -> Result<Vec<u8>, DeflateError> {
    let mut writer = BitWriter::new(BitOrder::LsbFirst);

    // Build fixed Huffman encoder
    let litlen_encoder = build_fixed_litlen_encoder();

    // Single block: BFINAL=1, BTYPE=01 (fixed)
    writer.write_bit(true); // BFINAL
    writer.write_bits(2, 1); // BTYPE = fixed

    // Encode each byte as a literal
    // Deflate Huffman codes are canonical (MSB) but packed LSB-first in the stream.
    // We reverse each code before writing to the LSB-first BitWriter.
    for &byte in data {
        let (code, len) = litlen_encoder.encode(byte as u16);
        writer.write_bits(len, reverse_code(code, len));
    }

    // End of block (symbol 256)
    let (code, len) = litlen_encoder.encode(256);
    writer.write_bits(len, reverse_code(code, len));

    Ok(writer.finish())
}

/// Reverse the bit order of a `len`-bit value.
/// Canonical Huffman codes are MSB-first but deflate packs them LSB-first.
fn reverse_code(value: u32, len: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..len {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Build fixed literal/length Huffman encoder (RFC 1951 Section 3.2.6).
fn build_fixed_litlen_encoder() -> HuffmanEncoder {
    let mut lengths = vec![0u8; 288];
    lengths[0..=143].fill(8);
    lengths[144..=255].fill(9);
    lengths[256..=279].fill(7);
    lengths[280..=287].fill(8);
    HuffmanEncoder::from_code_lengths(&lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflate_stored_roundtrip() {
        let original = b"Hello, World! This is a test of stored deflate blocks.";
        let compressed = deflate(original, CompressionLevel::None).unwrap();
        let decompressed = crate::inflate(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn deflate_stored_empty() {
        let compressed = deflate(b"", CompressionLevel::None).unwrap();
        let decompressed = crate::inflate(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn deflate_stored_large() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let compressed = deflate(&original, CompressionLevel::None).unwrap();
        let decompressed = crate::inflate(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn deflate_fixed_roundtrip() {
        let original = b"Hello, World!";
        let compressed = deflate(original, CompressionLevel::Fast).unwrap();
        let decompressed = crate::inflate(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn deflate_fixed_all_byte_values() {
        let original: Vec<u8> = (0..=255).collect();
        let compressed = deflate(&original, CompressionLevel::Fast).unwrap();
        let decompressed = crate::inflate(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn deflate_fixed_empty() {
        let compressed = deflate(b"", CompressionLevel::Fast).unwrap();
        let decompressed = crate::inflate(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn deflate_fixed_smaller_than_stored_for_text() {
        let text = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; // repetitive
        let stored = deflate(text, CompressionLevel::None).unwrap();
        let fixed = deflate(text, CompressionLevel::Fast).unwrap();
        // Fixed should be smaller than stored for repetitive text
        // (even without LZ77, Huffman coding of 'a' is more efficient than raw bytes)
        assert!(
            fixed.len() <= stored.len(),
            "fixed ({}) should be <= stored ({})",
            fixed.len(),
            stored.len()
        );
    }
}
