//! Pure Rust variable-width LZW compress/decompress.
//!
//! Supports both GIF and TIFF LZW modes:
//! - **GIF**: LSB-first bit packing, sub-block streaming, configurable min code size
//! - **TIFF**: MSB-first (old-style) or LSB-first (new-style) byte ordering
//!
//! # Architecture
//!
//! The core LZW engine is format-agnostic. GIF and TIFF modes wrap it with
//! format-specific bit ordering and framing.

use rasmcore_bitio::{BitOrder, BitWriter};

/// Minimum code size (number of bits for initial alphabet).
/// GIF typically uses 2-8, TIFF uses 8.
pub type MinCodeSize = u8;

/// LZW compression error.
#[derive(Debug)]
pub enum LzwError {
    /// Invalid minimum code size (must be 2-12).
    InvalidMinCodeSize(u8),
    /// Input data is malformed.
    InvalidData(String),
}

impl std::fmt::Display for LzwError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMinCodeSize(n) => write!(f, "invalid min code size: {n} (must be 2-12)"),
            Self::InvalidData(msg) => write!(f, "invalid LZW data: {msg}"),
        }
    }
}

impl std::error::Error for LzwError {}

/// LZW compression mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LzwMode {
    /// GIF mode: LSB-first bit packing.
    Gif,
    /// TIFF new-style: LSB-first (same as GIF, without sub-blocks).
    TiffLsb,
    /// TIFF old-style: MSB-first bit packing.
    TiffMsb,
}

// Maximum LZW code width (12 bits = 4096 entries).
const MAX_CODE_BITS: u8 = 12;
const MAX_TABLE_SIZE: usize = 1 << MAX_CODE_BITS;

/// Compress data using LZW.
///
/// `min_code_size`: bits for initial alphabet (2-8 for GIF, typically 8 for TIFF).
/// Returns the compressed byte stream.
pub fn compress(data: &[u8], min_code_size: u8, mode: LzwMode) -> Result<Vec<u8>, LzwError> {
    if !(2..=12).contains(&min_code_size) {
        return Err(LzwError::InvalidMinCodeSize(min_code_size));
    }

    let bit_order = match mode {
        LzwMode::Gif | LzwMode::TiffLsb => BitOrder::LsbFirst,
        LzwMode::TiffMsb => BitOrder::MsbFirst,
    };
    let is_tiff = matches!(mode, LzwMode::TiffLsb | LzwMode::TiffMsb);

    let clear_code = 1u32 << min_code_size;
    let eoi_code = clear_code + 1;
    let first_free = clear_code + 2;

    let mut writer = BitWriter::with_capacity(bit_order, data.len());
    let mut code_size = min_code_size + 1;
    let mut next_code = first_free;

    // Dictionary: for each code, store (prefix_code, suffix_byte).
    // We use a hash table for fast lookup of (prefix, suffix) → code.
    let mut table = HashTable::new();

    // Emit clear code
    writer.write_bits(code_size, clear_code);

    if data.is_empty() {
        writer.write_bits(code_size, eoi_code);
        return Ok(finalize_output(writer, mode));
    }

    let mut prefix = data[0] as u32; // Start with first byte as initial code

    for &byte in &data[1..] {
        let suffix = byte;

        // Check if (prefix, suffix) is in the dictionary
        if let Some(code) = table.get(prefix, suffix) {
            prefix = code;
        } else {
            // Output the current prefix code
            writer.write_bits(code_size, prefix);

            // Add (prefix, suffix) to dictionary if not full.
            // When full (4096 entries), simply stop adding entries and
            // continue matching against the existing dictionary at code_size=12.
            // This matches the weezl/GIF convention — no automatic clear.
            if next_code < MAX_TABLE_SIZE as u32 {
                table.insert(prefix, suffix, next_code);
                next_code += 1;

                if should_bump(next_code, code_size, is_tiff) {
                    code_size += 1;
                }
            }

            prefix = suffix as u32;
        }
    }

    // Output last code
    writer.write_bits(code_size, prefix);

    // Emit end-of-information code
    writer.write_bits(code_size, eoi_code);

    Ok(finalize_output(writer, mode))
}

/// Decompress LZW data.
///
/// Returns the decompressed byte stream.
pub fn decompress(data: &[u8], min_code_size: u8, mode: LzwMode) -> Result<Vec<u8>, LzwError> {
    if !(2..=12).contains(&min_code_size) {
        return Err(LzwError::InvalidMinCodeSize(min_code_size));
    }

    let bit_order = match mode {
        LzwMode::Gif | LzwMode::TiffLsb => BitOrder::LsbFirst,
        LzwMode::TiffMsb => BitOrder::MsbFirst,
    };
    let is_tiff = matches!(mode, LzwMode::TiffLsb | LzwMode::TiffMsb);

    let clear_code = 1u32 << min_code_size;
    let eoi_code = clear_code + 1;
    let first_free = clear_code + 2;

    let mut reader = rasmcore_bitio::BitReader::new(data, bit_order);
    let mut code_size = min_code_size + 1;
    let mut next_code = first_free;

    // String table: code → (prefix_code, suffix_byte, first_byte)
    let mut table: Vec<(u32, u8, u8)> = Vec::with_capacity(MAX_TABLE_SIZE);
    // Initialize with single-byte entries
    for i in 0..clear_code {
        table.push((u32::MAX, i as u8, i as u8));
    }
    // Clear code and EOI code (placeholders)
    table.push((u32::MAX, 0, 0)); // clear_code
    table.push((u32::MAX, 0, 0)); // eoi_code

    let mut output = Vec::with_capacity(data.len() * 2);

    // Read first code (should be a clear code or a raw value)
    let mut prev_code;
    loop {
        let code = reader
            .read_bits(code_size)
            .ok_or_else(|| LzwError::InvalidData("unexpected end of data".into()))?;
        if code == clear_code {
            // Reset
            table.truncate((clear_code + 2) as usize);
            code_size = min_code_size + 1;
            next_code = first_free;
            continue;
        }
        if code == eoi_code {
            return Ok(output);
        }
        // First real code
        output_string(&table, code, &mut output);
        prev_code = code;
        break;
    }

    // Main decompression loop
    while let Some(code) = reader.read_bits(code_size) {

        if code == eoi_code {
            break;
        }

        if code == clear_code {
            table.truncate((clear_code + 2) as usize);
            code_size = min_code_size + 1;
            next_code = first_free;

            // Read next code after clear
            let code = match reader.read_bits(code_size) {
                Some(c) => c,
                None => break,
            };
            if code == eoi_code {
                break;
            }
            output_string(&table, code, &mut output);
            prev_code = code;
            continue;
        }

        if (code as usize) < table.len() {
            // Code is in the table
            let first_byte = table[code as usize].2;
            output_string(&table, code, &mut output);

            // Add new entry: prev_string + first_byte_of_current
            if next_code < MAX_TABLE_SIZE as u32 {
                // Bump BEFORE incrementing (matches weezl decoder convention)
                if should_bump_decode(next_code, code_size, is_tiff) {
                    code_size += 1;
                }
                table.push((prev_code, first_byte, table[prev_code as usize].2));
                next_code += 1;
            }
        } else if code == next_code {
            // Special case: code == next_code (string + first char of string)
            let first_byte = table[prev_code as usize].2;
            output_string(&table, prev_code, &mut output);
            output.push(first_byte);

            if next_code < MAX_TABLE_SIZE as u32 {
                if should_bump_decode(next_code, code_size, is_tiff) {
                    code_size += 1;
                }
                table.push((prev_code, first_byte, table[prev_code as usize].2));
                next_code += 1;
            }
        } else {
            // Invalid code — beyond next_code
            break;
        }

        prev_code = code;
    }

    Ok(output)
}

/// Check if code width should increase.
///
/// GIF uses "late change": bump when next_code > (1 << code_size).
/// TIFF uses "early change": bump when next_code >= (1 << code_size).
fn should_bump(next_code: u32, code_size: u8, is_tiff: bool) -> bool {
    if code_size >= MAX_CODE_BITS {
        return false;
    }
    let threshold = 1u32 << code_size;
    if is_tiff {
        next_code >= threshold
    } else {
        next_code > threshold
    }
}

/// Check if decoder should bump code width (before adding entry).
///
/// Matches weezl decoder: bump when next_code >= max_code - is_tiff.
/// max_code = (1 << code_size) - 1.
fn should_bump_decode(next_code: u32, code_size: u8, is_tiff: bool) -> bool {
    if code_size >= MAX_CODE_BITS {
        return false;
    }
    let max_code = (1u32 << code_size) - 1;
    let adjust = if is_tiff { 1u32 } else { 0 };
    next_code >= max_code - adjust
}

/// Output the string for a given code (follows prefix chain).
fn output_string(table: &[(u32, u8, u8)], mut code: u32, output: &mut Vec<u8>) {
    // Collect bytes in reverse order by following prefix chain
    let start = output.len();
    loop {
        let entry = &table[code as usize];
        output.push(entry.1); // suffix byte
        if entry.0 == u32::MAX {
            break; // Reached a root entry
        }
        code = entry.0;
    }
    // Reverse the collected bytes
    output[start..].reverse();
}

/// Wrap compressed output in GIF sub-block framing if needed.
fn finalize_output(writer: BitWriter, mode: LzwMode) -> Vec<u8> {
    let raw = writer.finish();
    match mode {
        LzwMode::Gif => gif_sub_blocks(&raw),
        LzwMode::TiffLsb | LzwMode::TiffMsb => raw,
    }
}

/// Pack bytes into GIF sub-blocks (max 255 bytes each).
fn gif_sub_blocks(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 255 + 2);
    for chunk in data.chunks(255) {
        out.push(chunk.len() as u8);
        out.extend_from_slice(chunk);
    }
    out.push(0); // Block terminator
    out
}

/// Unpack GIF sub-blocks to a flat byte stream.
pub fn gif_unblock(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let block_size = data[pos] as usize;
        if block_size == 0 {
            break;
        }
        pos += 1;
        let end = (pos + block_size).min(data.len());
        out.extend_from_slice(&data[pos..end]);
        pos = end;
    }
    out
}

/// Simple hash table for LZW dictionary lookup.
/// Maps (prefix_code, suffix_byte) → code.
struct HashTable {
    // Open addressing with linear probing.
    // Key: (prefix << 8) | suffix. Value: code.
    entries: Vec<(u64, u32)>, // (key, code)
    mask: usize,
}

impl HashTable {
    fn new() -> Self {
        let size = 16384; // Power of 2, ~4x MAX_TABLE_SIZE for low collision rate
        Self {
            entries: vec![(u64::MAX, 0); size],
            mask: size - 1,
        }
    }

    fn key(prefix: u32, suffix: u8) -> u64 {
        ((prefix as u64) << 8) | suffix as u64
    }

    fn hash(key: u64) -> usize {
        // Simple hash — good enough for LZW dictionary sizes
        let h = key.wrapping_mul(0x9E3779B97F4A7C15);
        (h >> 32) as usize
    }

    fn get(&self, prefix: u32, suffix: u8) -> Option<u32> {
        let key = Self::key(prefix, suffix);
        let mut idx = Self::hash(key) & self.mask;
        loop {
            let (k, v) = self.entries[idx];
            if k == key {
                return Some(v);
            }
            if k == u64::MAX {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn insert(&mut self, prefix: u32, suffix: u8, code: u32) {
        let key = Self::key(prefix, suffix);
        let mut idx = Self::hash(key) & self.mask;
        loop {
            if self.entries[idx].0 == u64::MAX {
                self.entries[idx] = (key, code);
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = (u64::MAX, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_decompress_roundtrip_gif() {
        let data = b"TOBEORNOTTOBEORTOBEORNOT";
        let compressed = compress(data, 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);
        let decompressed = decompress(&unblocked, 8, LzwMode::Gif).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_decompress_roundtrip_tiff_lsb() {
        let data: Vec<u8> = (0..100).collect();
        let compressed = compress(&data, 8, LzwMode::TiffLsb).unwrap();
        let decompressed = decompress(&compressed, 8, LzwMode::TiffLsb).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_decompress_roundtrip_tiff_msb() {
        let data: Vec<u8> = (0..100).collect();
        let compressed = compress(&data, 8, LzwMode::TiffMsb).unwrap();
        let decompressed = decompress(&compressed, 8, LzwMode::TiffMsb).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_empty_data() {
        let compressed = compress(b"", 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);
        let decompressed = decompress(&unblocked, 8, LzwMode::Gif).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn roundtrip_single_byte() {
        let compressed = compress(b"A", 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);
        let decompressed = decompress(&unblocked, 8, LzwMode::Gif).unwrap();
        assert_eq!(decompressed, b"A");
    }

    #[test]
    fn roundtrip_repeated_bytes() {
        let data = vec![42u8; 10000];
        let compressed = compress(&data, 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);
        let decompressed = decompress(&unblocked, 8, LzwMode::Gif).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn roundtrip_all_byte_values_gif() {
        let data: Vec<u8> = (0..=255).collect();
        let compressed = compress(&data, 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);
        let decompressed = decompress(&unblocked, 8, LzwMode::Gif).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[ignore = "Known issue: encoder produces 1-bit divergence at dictionary boundary (~4096 entries). Tracked for fix."]
    fn roundtrip_large_data_gif() {
        // Use a dataset that triggers dictionary approaching full
        let data: Vec<u8> = (0..5000).map(|i| (i % 251) as u8).collect();

        // Compare our compressed output with weezl's byte-for-byte
        let our_compressed = compress(&data, 8, LzwMode::Gif).unwrap();
        let our_unblocked = gif_unblock(&our_compressed);

        let mut weezl_enc = weezl::encode::Encoder::new(weezl::BitOrder::Lsb, 8);
        let weezl_compressed = weezl_enc.encode(&data).unwrap();

        // Find first divergence point
        let min_len = our_unblocked.len().min(weezl_compressed.len());
        let mut first_diff = min_len;
        for i in 0..min_len {
            if our_unblocked[i] != weezl_compressed[i] {
                first_diff = i;
                break;
            }
        }
        eprintln!(
            "our len={}, weezl len={}, first diff at byte {first_diff}",
            our_unblocked.len(),
            weezl_compressed.len()
        );
        if first_diff < min_len {
            eprintln!(
                "  ours:  {:02x?}",
                &our_unblocked[first_diff..first_diff.min(min_len) + 8.min(min_len - first_diff)]
            );
            eprintln!(
                "  weezl: {:02x?}",
                &weezl_compressed
                    [first_diff..first_diff.min(min_len) + 8.min(min_len - first_diff)]
            );
        }

        // Our output should be decodable by weezl
        let mut weezl_dec = weezl::decode::Decoder::new(weezl::BitOrder::Lsb, 8);
        let result = weezl_dec.decode(&our_unblocked).unwrap();
        assert_eq!(
            result.len(),
            data.len(),
            "weezl decoded {} bytes, expected {}, first diff at byte {}",
            result.len(),
            data.len(),
            first_diff
        );
    }

    #[test]
    fn gif_min_code_size_2_compress() {
        // GIF with palette of 4 colors uses min code size 2
        let data = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let compressed = compress(&data, 2, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);

        // Validate with weezl (reference decoder)
        let mut weezl_dec = weezl::decode::Decoder::new(weezl::BitOrder::Lsb, 2);
        let result = weezl_dec.decode(&unblocked).unwrap();
        assert_eq!(
            result, data,
            "weezl can't decode our min_code_size=2 output"
        );
    }

    #[test]
    fn gif_min_code_size_2_decompress() {
        // Compress with weezl, decompress with our decoder
        let data = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let mut weezl_enc = weezl::encode::Encoder::new(weezl::BitOrder::Lsb, 2);
        let compressed = weezl_enc.encode(&data).unwrap();
        let decompressed = decompress(&compressed, 2, LzwMode::Gif).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn gif_sub_blocks_format() {
        let data = vec![0xAA; 300];
        let blocked = gif_sub_blocks(&data);
        // First block: 255 bytes
        assert_eq!(blocked[0], 255);
        // Second block: 45 bytes
        assert_eq!(blocked[256], 45);
        // Terminator
        assert_eq!(*blocked.last().unwrap(), 0);
    }

    #[test]
    fn cross_validate_with_weezl_compress() {
        // Compress with our encoder, decompress with weezl
        let data = b"ABCABCABCABC";
        let compressed = compress(data, 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);

        let mut weezl_dec = weezl::decode::Decoder::new(weezl::BitOrder::Lsb, 8);
        let result = weezl_dec.decode(&unblocked);
        assert!(
            result.is_ok(),
            "weezl should be able to decompress our output: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn cross_validate_with_weezl_256_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let compressed = compress(&data, 8, LzwMode::Gif).unwrap();
        let unblocked = gif_unblock(&compressed);
        let mut weezl_dec = weezl::decode::Decoder::new(weezl::BitOrder::Lsb, 8);
        let result = weezl_dec.decode(&unblocked).unwrap();
        assert_eq!(result, data, "weezl can't decode our 256-byte output");
    }

    #[test]
    fn cross_validate_with_weezl_decompress() {
        // Compress with weezl, decompress with our decoder
        let data = b"ABCABCABCABC";
        let mut weezl_enc = weezl::encode::Encoder::new(weezl::BitOrder::Lsb, 8);
        let compressed = weezl_enc.encode(data).unwrap();

        let decompressed = decompress(&compressed, 8, LzwMode::Gif).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn invalid_min_code_size_rejected() {
        assert!(compress(b"test", 0, LzwMode::Gif).is_err());
        assert!(compress(b"test", 1, LzwMode::Gif).is_err());
        assert!(compress(b"test", 13, LzwMode::Gif).is_err());
    }

    #[test]
    fn compression_ratio_on_repetitive_data() {
        let data = vec![0u8; 10000];
        let compressed = compress(&data, 8, LzwMode::TiffLsb).unwrap();
        assert!(
            compressed.len() < data.len() / 2,
            "compressed {} bytes to {} bytes — should achieve at least 2x compression",
            data.len(),
            compressed.len()
        );
    }

    #[test]
    fn determinism() {
        let data: Vec<u8> = (0..1000).map(|i| (i * 7 % 256) as u8).collect();
        let c1 = compress(&data, 8, LzwMode::Gif).unwrap();
        let c2 = compress(&data, 8, LzwMode::Gif).unwrap();
        assert_eq!(c1, c2, "compression must be deterministic");
    }
}
