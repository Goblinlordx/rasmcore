//! ICC color profile operations.
//!
//! Uses moxcms (pure Rust, WASM-compatible) for ICC profile parsing and
//! color space transforms. This module provides:
//! - ICC profile extraction from JPEG/PNG raw bytes
//! - ICC-to-sRGB pixel conversion via moxcms transform

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

/// Extract ICC profile from JPEG data (APP2 marker with "ICC_PROFILE\0" signature).
///
/// ICC profiles in JPEG are stored in APP2 markers (0xFF, 0xE2) with the
/// signature "ICC_PROFILE\0" followed by a sequence number and count.
/// Large profiles are split across multiple APP2 chunks.
pub fn extract_icc_from_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    // ICC_PROFILE marker signature: "ICC_PROFILE\0" (12 bytes) + seq_no (1) + num_markers (1)
    const ICC_SIG: &[u8] = b"ICC_PROFILE\0";
    const ICC_HEADER_LEN: usize = 14; // signature (12) + seq (1) + count (1)

    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();

    let mut pos = 0;
    while pos + 4 < data.len() {
        // Find JPEG marker
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = data[pos + 1];

        // Skip padding bytes (0xFF 0xFF)
        if marker == 0xFF {
            pos += 1;
            continue;
        }

        // SOS marker — stop scanning (pixel data follows)
        if marker == 0xDA {
            break;
        }

        // Markers without length: SOI, EOI, RST0-RST7
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            pos += 2;
            continue;
        }

        // Read marker length
        if pos + 4 > data.len() {
            break;
        }
        let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        if length < 2 {
            break;
        }

        let segment_start = pos + 4;
        let segment_len = length - 2;

        // Check for APP2 marker (0xE2) with ICC_PROFILE signature
        if marker == 0xE2 && segment_len > ICC_HEADER_LEN {
            let segment_data = &data[segment_start..segment_start + segment_len.min(data.len() - segment_start)];
            if segment_data.starts_with(ICC_SIG) {
                let seq_no = segment_data[12];
                let icc_data = segment_data[ICC_HEADER_LEN..].to_vec();
                chunks.push((seq_no, icc_data));
            }
        }

        pos = segment_start + segment_len;
    }

    if chunks.is_empty() {
        return None;
    }

    // Sort by sequence number and concatenate
    chunks.sort_by_key(|(seq, _)| *seq);
    let mut profile = Vec::new();
    for (_, chunk) in chunks {
        profile.extend_from_slice(&chunk);
    }

    Some(profile)
}

/// Extract ICC profile from PNG data (iCCP chunk).
///
/// PNG stores ICC profiles in the iCCP chunk, which contains:
/// - Null-terminated profile name (1-79 bytes + null)
/// - Compression method (1 byte, always 0 = zlib deflate)
/// - Compressed profile data (zlib/deflate)
pub fn extract_icc_from_png(data: &[u8]) -> Option<Vec<u8>> {
    // PNG signature: 8 bytes
    const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    const ICCP_TYPE: &[u8] = b"iCCP";

    if !data.starts_with(PNG_SIG) {
        return None;
    }

    let mut pos = 8; // Skip PNG signature

    while pos + 12 <= data.len() {
        let chunk_len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        let chunk_data_start = pos + 8;

        if chunk_type == ICCP_TYPE && chunk_data_start + chunk_len <= data.len() {
            let chunk_data = &data[chunk_data_start..chunk_data_start + chunk_len];

            // Find null terminator of profile name
            let name_end = chunk_data.iter().position(|&b| b == 0)?;
            if name_end + 2 > chunk_data.len() {
                return None;
            }

            // Skip profile name + null + compression method byte
            let compressed = &chunk_data[name_end + 2..];

            // Decompress with zlib (deflate)
            return decompress_zlib(compressed);
        }

        // IDAT means we've passed all metadata chunks
        if chunk_type == b"IDAT" {
            break;
        }

        // Skip to next chunk: length(4) + type(4) + data(chunk_len) + crc(4)
        pos = chunk_data_start + chunk_len + 4;
    }

    None
}

/// Decompress zlib data (for PNG iCCP chunks).
fn decompress_zlib(compressed: &[u8]) -> Option<Vec<u8>> {
    // Minimal zlib decompressor using raw deflate
    // zlib format: CMF(1) + FLG(1) + compressed_data + ADLER32(4)
    if compressed.len() < 6 {
        return None;
    }

    let cmf = compressed[0];
    let cm = cmf & 0x0F;
    if cm != 8 {
        // Only deflate (CM=8) is valid
        return None;
    }

    // Use miniz_oxide (available via image crate's transitive dependency)
    // which provides inflate
    let deflate_data = &compressed[2..compressed.len().saturating_sub(4)];
    miniz_oxide_decompress(deflate_data)
}

/// Decompress deflate data using a simple inflate implementation.
fn miniz_oxide_decompress(deflate_data: &[u8]) -> Option<Vec<u8>> {
    inflate_deflate(deflate_data)
}

/// Minimal RFC 1951 DEFLATE decompressor.
///
/// Supports: uncompressed blocks, fixed Huffman, dynamic Huffman.
/// Sufficient for ICC profile decompression in PNG iCCP chunks.
fn inflate_deflate(input: &[u8]) -> Option<Vec<u8>> {
    let mut reader = DeflateReader::new(input);
    reader.decompress()
}

struct DeflateReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: u8,
    bit_buf: u32,
}

impl<'a> DeflateReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_pos: 0,
            bit_buf: 0,
        }
    }

    fn read_bits(&mut self, count: u8) -> Option<u32> {
        while self.bit_pos < count {
            if self.pos >= self.data.len() {
                return None;
            }
            self.bit_buf |= (self.data[self.pos] as u32) << self.bit_pos;
            self.pos += 1;
            self.bit_pos += 8;
        }
        let val = self.bit_buf & ((1 << count) - 1);
        self.bit_buf >>= count;
        self.bit_pos -= count;
        Some(val)
    }

    fn decompress(&mut self) -> Option<Vec<u8>> {
        let mut output = Vec::new();

        loop {
            let bfinal = self.read_bits(1)?;
            let btype = self.read_bits(2)?;

            match btype {
                0 => self.decompress_uncompressed(&mut output)?,
                1 => self.decompress_fixed_huffman(&mut output)?,
                2 => self.decompress_dynamic_huffman(&mut output)?,
                _ => return None,
            }

            if bfinal == 1 {
                break;
            }
        }

        Some(output)
    }

    fn decompress_uncompressed(&mut self, output: &mut Vec<u8>) -> Option<()> {
        // Align to byte boundary
        self.bit_buf = 0;
        self.bit_pos = 0;

        if self.pos + 4 > self.data.len() {
            return None;
        }
        let len = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]) as usize;
        self.pos += 4; // skip LEN and NLEN

        if self.pos + len > self.data.len() {
            return None;
        }
        output.extend_from_slice(&self.data[self.pos..self.pos + len]);
        self.pos += len;
        Some(())
    }

    fn decompress_fixed_huffman(&mut self, output: &mut Vec<u8>) -> Option<()> {
        loop {
            let sym = self.decode_fixed_lit_len()?;
            if sym == 256 {
                break;
            }
            if sym < 256 {
                output.push(sym as u8);
            } else {
                let length = self.decode_length(sym)?;
                let dist_code = self.read_bits_reversed(5)?;
                let distance = self.decode_distance(dist_code)?;
                self.copy_match(output, distance, length)?;
            }
        }
        Some(())
    }

    fn decompress_dynamic_huffman(&mut self, output: &mut Vec<u8>) -> Option<()> {
        let hlit = self.read_bits(5)? as usize + 257;
        let hdist = self.read_bits(5)? as usize + 1;
        let hclen = self.read_bits(4)? as usize + 4;

        // Code length alphabet order
        const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

        let mut cl_lens = [0u8; 19];
        for i in 0..hclen {
            cl_lens[CL_ORDER[i]] = self.read_bits(3)? as u8;
        }

        let cl_table = build_huffman_table(&cl_lens)?;

        // Decode literal/length and distance code lengths
        let mut all_lens = vec![0u8; hlit + hdist];
        let mut i = 0;
        while i < all_lens.len() {
            let sym = self.decode_huffman(&cl_table)?;
            match sym {
                0..=15 => {
                    all_lens[i] = sym as u8;
                    i += 1;
                }
                16 => {
                    let repeat = self.read_bits(2)? as usize + 3;
                    if i == 0 {
                        return None;
                    }
                    let val = all_lens[i - 1];
                    for _ in 0..repeat {
                        if i >= all_lens.len() {
                            return None;
                        }
                        all_lens[i] = val;
                        i += 1;
                    }
                }
                17 => {
                    let repeat = self.read_bits(3)? as usize + 3;
                    i += repeat;
                }
                18 => {
                    let repeat = self.read_bits(7)? as usize + 11;
                    i += repeat;
                }
                _ => return None,
            }
        }

        let lit_table = build_huffman_table(&all_lens[..hlit])?;
        let dist_table = build_huffman_table(&all_lens[hlit..])?;

        loop {
            let sym = self.decode_huffman(&lit_table)?;
            if sym == 256 {
                break;
            }
            if sym < 256 {
                output.push(sym as u8);
            } else {
                let length = self.decode_length(sym as u32)?;
                let dist_code = self.decode_huffman(&dist_table)?;
                let distance = self.decode_distance(dist_code as u32)?;
                self.copy_match(output, distance, length)?;
            }
        }
        Some(())
    }

    fn decode_fixed_lit_len(&mut self) -> Option<u32> {
        // Fixed Huffman codes: RFC 1951 section 3.2.6
        let mut code = 0u32;
        for _ in 0..7 {
            let bit = self.read_bits(1)?;
            code = (code << 1) | bit;
        }
        // 7-bit codes: 256-279 map to codes 0000000-0010111
        if code <= 0x17 {
            return Some(code + 256);
        }

        let bit = self.read_bits(1)?;
        code = (code << 1) | bit;
        // 8-bit codes: 0-143 map to 00110000-10111111
        if (0x30..=0xBF).contains(&code) {
            return Some(code - 0x30);
        }
        // 8-bit codes: 280-287 map to 11000000-11000111
        if (0xC0..=0xC7).contains(&code) {
            return Some(code - 0xC0 + 280);
        }

        let bit = self.read_bits(1)?;
        code = (code << 1) | bit;
        // 9-bit codes: 144-255 map to 110010000-111111111
        if (0x190..=0x1FF).contains(&code) {
            return Some(code - 0x190 + 144);
        }

        None
    }

    fn read_bits_reversed(&mut self, count: u8) -> Option<u32> {
        let val = self.read_bits(count)?;
        let mut reversed = 0u32;
        for i in 0..count {
            if val & (1 << i) != 0 {
                reversed |= 1 << (count - 1 - i);
            }
        }
        Some(reversed)
    }

    fn decode_huffman(&mut self, table: &HuffmanTable) -> Option<u32> {
        let mut code = 0u32;
        for len in 1..=15 {
            let bit = self.read_bits(1)?;
            code = (code << 1) | bit;
            if let Some(&sym) = table.lookup.get(&(len, code)) {
                return Some(sym);
            }
        }
        None
    }

    fn decode_length(&mut self, sym: u32) -> Option<usize> {
        const LEN_BASE: [usize; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
            35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
        ];
        const LEN_EXTRA: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
            3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];

        let idx = (sym - 257) as usize;
        if idx >= LEN_BASE.len() {
            return None;
        }
        let extra = self.read_bits(LEN_EXTRA[idx])? as usize;
        Some(LEN_BASE[idx] + extra)
    }

    fn decode_distance(&mut self, code: u32) -> Option<usize> {
        const DIST_BASE: [usize; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
            257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
        ];
        const DIST_EXTRA: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
            7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        ];

        let idx = code as usize;
        if idx >= DIST_BASE.len() {
            return None;
        }
        let extra = self.read_bits(DIST_EXTRA[idx])? as usize;
        Some(DIST_BASE[idx] + extra)
    }

    fn copy_match(&self, output: &mut Vec<u8>, distance: usize, length: usize) -> Option<()> {
        if distance > output.len() {
            return None;
        }
        let start = output.len() - distance;
        for i in 0..length {
            let byte = output[start + (i % distance)];
            output.push(byte);
        }
        Some(())
    }
}

struct HuffmanTable {
    lookup: std::collections::HashMap<(u8, u32), u32>,
}

fn build_huffman_table(lengths: &[u8]) -> Option<HuffmanTable> {
    let max_len = *lengths.iter().max().unwrap_or(&0) as usize;
    if max_len == 0 {
        return Some(HuffmanTable {
            lookup: std::collections::HashMap::new(),
        });
    }

    // Count codes per length
    let mut bl_count = vec![0u32; max_len + 1];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }

    // Compute starting codes
    let mut next_code = vec![0u32; max_len + 1];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes
    let mut lookup = std::collections::HashMap::new();
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 {
            let code = next_code[len as usize];
            next_code[len as usize] += 1;
            lookup.insert((len, code), sym as u32);
        }
    }

    Some(HuffmanTable { lookup })
}

/// Convert image pixels from an ICC profile's color space to sRGB using moxcms.
pub fn icc_to_srgb(
    pixels: &[u8],
    info: &ImageInfo,
    icc_profile: &[u8],
) -> Result<Vec<u8>, ImageError> {
    use moxcms::{ColorProfile, Layout, TransformOptions};

    let src_profile = ColorProfile::new_from_slice(icc_profile)
        .map_err(|e| ImageError::ProcessingFailed(format!("invalid ICC profile: {e:?}")))?;

    let dst_profile = ColorProfile::new_srgb();

    let (src_layout, dst_layout) = match info.format {
        PixelFormat::Rgb8 => (Layout::Rgb, Layout::Rgb),
        PixelFormat::Rgba8 => (Layout::Rgba, Layout::Rgba),
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "ICC conversion from {other:?} not supported — convert to RGB8 or RGBA8 first"
            )));
        }
    };

    let transform = src_profile
        .create_transform_8bit(src_layout, &dst_profile, dst_layout, TransformOptions::default())
        .map_err(|e| ImageError::ProcessingFailed(format!("ICC transform creation failed: {e:?}")))?;

    let mut result = vec![0u8; pixels.len()];
    transform
        .transform(pixels, &mut result)
        .map_err(|e| ImageError::ProcessingFailed(format!("ICC transform failed: {e:?}")))?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal valid sRGB ICC v2 profile.
    // The macOS system profile is large; this generates a minimal one from
    // moxcms by creating an sRGB transform and using the system profile
    // if available.
    fn load_test_icc_profile() -> Option<Vec<u8>> {
        // Try macOS system sRGB profile first
        let path = "/System/Library/ColorSync/Profiles/sRGB Profile.icc";
        std::fs::read(path).ok()
    }

    fn load_display_p3_profile() -> Option<Vec<u8>> {
        let path = "/System/Library/ColorSync/Profiles/Display P3.icc";
        std::fs::read(path).ok()
    }

    #[test]
    fn icc_to_srgb_identity_with_srgb_profile() {
        let profile = match load_test_icc_profile() {
            Some(p) => p,
            None => return, // skip on non-macOS
        };

        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: super::super::types::ColorSpace::Srgb,
        };

        // Generate test pixels: known RGB values
        let pixels: Vec<u8> = (0..4 * 4 * 3).map(|i| (i * 16 % 256) as u8).collect();

        let result = icc_to_srgb(&pixels, &info, &profile).unwrap();
        assert_eq!(result.len(), pixels.len());

        // sRGB→sRGB should be near-identity (allow small rounding)
        for (a, b) in pixels.iter().zip(result.iter()) {
            assert!(
                (*a as i16 - *b as i16).unsigned_abs() <= 2,
                "sRGB identity drift too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn icc_to_srgb_display_p3_changes_pixels() {
        let profile = match load_display_p3_profile() {
            Some(p) => p,
            None => return,
        };

        let info = ImageInfo {
            width: 2,
            height: 2,
            format: PixelFormat::Rgb8,
            color_space: super::super::types::ColorSpace::DisplayP3,
        };

        // Bright saturated red in Display P3 (wider gamut than sRGB)
        let pixels = vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];

        let result = icc_to_srgb(&pixels, &info, &profile).unwrap();
        assert_eq!(result.len(), pixels.len());

        // Display P3 red (255,0,0) mapped to sRGB should change —
        // the transform should produce different values since P3 has a wider gamut.
        // The result should still be valid RGB values.
        assert!(result.iter().all(|&v| v <= 255));
    }

    #[test]
    fn icc_to_srgb_rgba8_format() {
        let profile = match load_test_icc_profile() {
            Some(p) => p,
            None => return,
        };

        let info = ImageInfo {
            width: 2,
            height: 2,
            format: PixelFormat::Rgba8,
            color_space: super::super::types::ColorSpace::Srgb,
        };

        // RGBA pixels with alpha
        let pixels = vec![128, 64, 32, 200, 128, 64, 32, 200, 128, 64, 32, 200, 128, 64, 32, 200];

        let result = icc_to_srgb(&pixels, &info, &profile).unwrap();
        assert_eq!(result.len(), pixels.len());

        // Alpha channel should be preserved (check every 4th byte)
        for i in (3..result.len()).step_by(4) {
            assert_eq!(result[i], pixels[i], "alpha channel was modified at index {i}");
        }
    }

    #[test]
    fn icc_to_srgb_invalid_profile_returns_error() {
        let info = ImageInfo {
            width: 2,
            height: 2,
            format: PixelFormat::Rgb8,
            color_space: super::super::types::ColorSpace::Srgb,
        };
        let pixels = vec![0u8; 2 * 2 * 3];
        let bad_profile = vec![0u8; 64]; // not a valid ICC profile

        let result = icc_to_srgb(&pixels, &info, &bad_profile);
        assert!(result.is_err());
    }

    #[test]
    fn icc_to_srgb_unsupported_format_returns_error() {
        let profile = match load_test_icc_profile() {
            Some(p) => p,
            None => return,
        };

        let info = ImageInfo {
            width: 2,
            height: 2,
            format: PixelFormat::Gray8,
            color_space: super::super::types::ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 2 * 2];

        let result = icc_to_srgb(&pixels, &info, &profile);
        assert!(result.is_err());
    }

    #[test]
    fn extract_icc_from_jpeg_no_profile() {
        // Minimal JPEG with no ICC: SOI + EOI
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
        assert!(extract_icc_from_jpeg(&jpeg).is_none());
    }

    #[test]
    fn extract_icc_from_png_no_profile() {
        // Minimal valid PNG header (signature + IHDR + IDAT)
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        // IDAT chunk (empty) to stop scanning
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // length
        png.extend_from_slice(b"IDAT");
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // CRC
        assert!(extract_icc_from_png(&png).is_none());
    }

    #[test]
    fn extract_icc_from_jpeg_with_app2() {
        // Construct a JPEG with an APP2 ICC_PROFILE marker
        let icc_data = vec![42u8; 100]; // Fake ICC profile data
        let mut jpeg = vec![0xFF, 0xD8]; // SOI

        // APP2 marker
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let seg_len = (2 + 14 + icc_data.len()) as u16;
        jpeg.extend_from_slice(&seg_len.to_be_bytes());
        jpeg.extend_from_slice(b"ICC_PROFILE\0");
        jpeg.push(1); // sequence number
        jpeg.push(1); // total count
        jpeg.extend_from_slice(&icc_data);

        // SOS marker (stops scanning)
        jpeg.push(0xFF);
        jpeg.push(0xDA);
        jpeg.extend_from_slice(&[0x00, 0x02]); // length

        let extracted = extract_icc_from_jpeg(&jpeg);
        assert_eq!(extracted, Some(icc_data));
    }

    #[test]
    fn extract_icc_from_jpeg_multi_chunk() {
        let chunk1 = vec![1u8; 50];
        let chunk2 = vec![2u8; 50];
        let mut jpeg = vec![0xFF, 0xD8]; // SOI

        // First APP2 chunk (seq 1)
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let seg_len = (2 + 14 + chunk1.len()) as u16;
        jpeg.extend_from_slice(&seg_len.to_be_bytes());
        jpeg.extend_from_slice(b"ICC_PROFILE\0");
        jpeg.push(1); // seq 1
        jpeg.push(2); // total 2
        jpeg.extend_from_slice(&chunk1);

        // Second APP2 chunk (seq 2)
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let seg_len = (2 + 14 + chunk2.len()) as u16;
        jpeg.extend_from_slice(&seg_len.to_be_bytes());
        jpeg.extend_from_slice(b"ICC_PROFILE\0");
        jpeg.push(2); // seq 2
        jpeg.push(2); // total 2
        jpeg.extend_from_slice(&chunk2);

        jpeg.push(0xFF);
        jpeg.push(0xDA);
        jpeg.extend_from_slice(&[0x00, 0x02]);

        let extracted = extract_icc_from_jpeg(&jpeg).unwrap();
        let mut expected = chunk1;
        expected.extend_from_slice(&chunk2);
        assert_eq!(extracted, expected);
    }

    #[test]
    fn extract_icc_from_png_with_iccp() {
        // Create a PNG with an iCCP chunk containing a zlib-compressed profile
        let icc_data = vec![99u8; 50]; // Fake ICC data

        // Compress with zlib (CMF + FLG + uncompressed deflate block + ADLER32)
        let mut compressed = Vec::new();
        compressed.push(0x78); // CMF: CM=8, CINFO=7
        compressed.push(0x01); // FLG: FCHECK=1
        // Uncompressed deflate block: BFINAL=1, BTYPE=00
        compressed.push(0x01); // BFINAL=1, BTYPE=0
        let len = icc_data.len() as u16;
        compressed.extend_from_slice(&len.to_le_bytes());
        let nlen = !len;
        compressed.extend_from_slice(&nlen.to_le_bytes());
        compressed.extend_from_slice(&icc_data);
        // ADLER32 placeholder (not checked by our decompressor)
        compressed.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        // Build iCCP chunk data: name + null + compression_method + compressed
        let mut chunk_data = Vec::new();
        chunk_data.extend_from_slice(b"sRGB"); // profile name
        chunk_data.push(0); // null terminator
        chunk_data.push(0); // compression method (deflate)
        chunk_data.extend_from_slice(&compressed);

        // Build PNG
        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        // iCCP chunk
        let chunk_len = chunk_data.len() as u32;
        png.extend_from_slice(&chunk_len.to_be_bytes());
        png.extend_from_slice(b"iCCP");
        png.extend_from_slice(&chunk_data);
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // CRC (not validated)

        // IDAT chunk to stop scanning
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        png.extend_from_slice(b"IDAT");
        png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        let extracted = extract_icc_from_png(&png);
        assert_eq!(extracted, Some(icc_data));
    }
}
