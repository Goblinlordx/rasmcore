//! Pure Rust PNM (Portable Any Map) encoder/decoder.
//!
//! Supports PBM (P1/P4), PGM (P2/P5), PPM (P3/P6) in both ASCII and binary.
//! Uses rasmcore-bitio for PBM bit-level I/O. WASM-ready.

use rasmcore_bitio::{BitOrder, BitReader, BitWriter};
use std::fmt;

/// PNM format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PnmFormat {
    /// P1 (ASCII) / P4 (binary) — bitmap (1-bit)
    Pbm,
    /// P2 (ASCII) / P5 (binary) — grayscale
    Pgm,
    /// P3 (ASCII) / P6 (binary) — RGB color
    Ppm,
}

/// PNM encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PnmMode {
    Ascii,
    Binary,
}

/// PNM image header.
#[derive(Debug, Clone)]
pub struct PnmHeader {
    pub format: PnmFormat,
    pub mode: PnmMode,
    pub width: u32,
    pub height: u32,
    pub maxval: u16,
}

/// PNM error.
#[derive(Debug)]
pub enum PnmError {
    InvalidMagic,
    InvalidHeader(String),
    InvalidData(String),
    BufferTooSmall,
}

impl fmt::Display for PnmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid PNM magic number"),
            Self::InvalidHeader(msg) => write!(f, "invalid PNM header: {msg}"),
            Self::InvalidData(msg) => write!(f, "invalid PNM data: {msg}"),
            Self::BufferTooSmall => write!(f, "pixel buffer too small"),
        }
    }
}

impl std::error::Error for PnmError {}

/// Encode grayscale pixels to PGM (P5 binary).
pub fn encode_pgm(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(PnmError::BufferTooSmall);
    }
    let header = format!("P5\n{width} {height}\n255\n");
    let mut out = Vec::with_capacity(header.len() + expected);
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&pixels[..expected]);
    Ok(out)
}

/// Encode RGB pixels to PPM (P6 binary).
pub fn encode_ppm(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let expected = width as usize * height as usize * 3;
    if pixels.len() < expected {
        return Err(PnmError::BufferTooSmall);
    }
    let header = format!("P6\n{width} {height}\n255\n");
    let mut out = Vec::with_capacity(header.len() + expected);
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&pixels[..expected]);
    Ok(out)
}

/// Encode bitmap to PBM (P4 binary).
///
/// `pixels` is one byte per pixel (0 = white, non-zero = black).
pub fn encode_pbm(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(PnmError::BufferTooSmall);
    }
    let header = format!("P4\n{width} {height}\n");
    let row_bytes = width.div_ceil(8) as usize;
    let mut out = Vec::with_capacity(header.len() + row_bytes * height as usize);
    out.extend_from_slice(header.as_bytes());

    // Use BitWriter (MSB-first) for PBM bit packing
    let mut bw = BitWriter::new(BitOrder::MsbFirst);
    for row in 0..height as usize {
        for col in 0..width as usize {
            bw.write_bit(pixels[row * width as usize + col] != 0);
        }
        // PBM rows are byte-aligned
        bw.align_to_byte();
    }
    out.extend_from_slice(&bw.finish());
    Ok(out)
}

/// Decode a PNM file to raw pixels.
///
/// Returns (header, pixels) where pixels format depends on the PNM type.
pub fn decode(data: &[u8]) -> Result<(PnmHeader, Vec<u8>), PnmError> {
    if data.len() < 3 {
        return Err(PnmError::InvalidMagic);
    }

    let (format, mode) = match &data[..2] {
        b"P1" => (PnmFormat::Pbm, PnmMode::Ascii),
        b"P2" => (PnmFormat::Pgm, PnmMode::Ascii),
        b"P3" => (PnmFormat::Ppm, PnmMode::Ascii),
        b"P4" => (PnmFormat::Pbm, PnmMode::Binary),
        b"P5" => (PnmFormat::Pgm, PnmMode::Binary),
        b"P6" => (PnmFormat::Ppm, PnmMode::Binary),
        _ => return Err(PnmError::InvalidMagic),
    };

    // Parse header: skip whitespace/comments, read width, height, maxval
    let mut pos = 2;
    let width = parse_number(data, &mut pos)?;
    let height = parse_number(data, &mut pos)?;
    let maxval = if format == PnmFormat::Pbm {
        1
    } else {
        parse_number(data, &mut pos)? as u16
    };

    // Skip single whitespace after header
    if pos < data.len() && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r') {
        pos += 1;
    }

    let header = PnmHeader {
        format,
        mode,
        width,
        height,
        maxval,
    };

    let pixels = match (format, mode) {
        (PnmFormat::Pgm, PnmMode::Binary) => {
            let len = width as usize * height as usize;
            if pos + len > data.len() {
                return Err(PnmError::InvalidData("not enough pixel data".into()));
            }
            data[pos..pos + len].to_vec()
        }
        (PnmFormat::Ppm, PnmMode::Binary) => {
            let len = width as usize * height as usize * 3;
            if pos + len > data.len() {
                return Err(PnmError::InvalidData("not enough pixel data".into()));
            }
            data[pos..pos + len].to_vec()
        }
        (PnmFormat::Pbm, PnmMode::Binary) => {
            let row_bytes = width.div_ceil(8) as usize;
            let total = row_bytes * height as usize;
            if pos + total > data.len() {
                return Err(PnmError::InvalidData("not enough pixel data".into()));
            }
            // Use BitReader (MSB-first) for PBM bit unpacking
            let mut pixels = Vec::with_capacity(width as usize * height as usize);
            let mut br = BitReader::new(&data[pos..pos + total], BitOrder::MsbFirst);
            for _row in 0..height as usize {
                for _col in 0..width as usize {
                    let bit = br.read_bit().unwrap_or(false);
                    pixels.push(if bit { 255 } else { 0 });
                }
                // PBM rows are byte-aligned — skip remaining bits in row
                br.align_to_byte();
            }
            pixels
        }
        (_, PnmMode::Ascii) => {
            // Parse ASCII values
            let mut pixels = Vec::new();
            let count = match format {
                PnmFormat::Pbm => width as usize * height as usize,
                PnmFormat::Pgm => width as usize * height as usize,
                PnmFormat::Ppm => width as usize * height as usize * 3,
            };
            for _ in 0..count {
                let val = parse_number(data, &mut pos)?;
                pixels.push(val.min(255) as u8);
            }
            pixels
        }
    };

    Ok((header, pixels))
}

/// Parse a decimal number from the data, skipping whitespace and comments.
fn parse_number(data: &[u8], pos: &mut usize) -> Result<u32, PnmError> {
    // Skip whitespace and comments
    while *pos < data.len() {
        match data[*pos] {
            b' ' | b'\t' | b'\n' | b'\r' => *pos += 1,
            b'#' => {
                while *pos < data.len() && data[*pos] != b'\n' {
                    *pos += 1;
                }
            }
            _ => break,
        }
    }

    if *pos >= data.len() || !data[*pos].is_ascii_digit() {
        return Err(PnmError::InvalidHeader("expected number".into()));
    }

    let mut n = 0u32;
    while *pos < data.len() && data[*pos].is_ascii_digit() {
        n = n * 10 + (data[*pos] - b'0') as u32;
        *pos += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ppm_binary() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_ppm(&pixels, 16, 16).unwrap();
        assert_eq!(&encoded[..2], b"P6");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 16);
        assert_eq!(header.height, 16);
        assert_eq!(header.format, PnmFormat::Ppm);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_pgm_binary() {
        let pixels: Vec<u8> = (0..8 * 8).map(|i| (i % 256) as u8).collect();
        let encoded = encode_pgm(&pixels, 8, 8).unwrap();
        assert_eq!(&encoded[..2], b"P5");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pgm);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_pbm_binary() {
        // 8x2 bitmap: alternating black/white
        let pixels: Vec<u8> = (0..16).map(|i| if i % 2 == 0 { 255 } else { 0 }).collect();
        let encoded = encode_pbm(&pixels, 8, 2).unwrap();
        assert_eq!(&encoded[..2], b"P4");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pbm);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_ppm_ascii() {
        let data = b"P3\n2 2\n255\n255 0 0 0 255 0 0 0 255 128 128 128\n";
        let (header, pixels) = decode(data).unwrap();
        assert_eq!(header.format, PnmFormat::Ppm);
        assert_eq!(header.mode, PnmMode::Ascii);
        assert_eq!(pixels, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]);
    }

    #[test]
    fn decode_with_comments() {
        let data =
            b"P6\n# comment\n2 2\n# another\n255\n\x00\x00\x00\xFF\xFF\xFF\x80\x80\x80\x40\x40\x40";
        let (header, _pixels) = decode(data).unwrap();
        assert_eq!(header.width, 2);
        assert_eq!(header.height, 2);
    }

    #[test]
    fn invalid_magic_rejected() {
        assert!(decode(b"XX\n1 1\n255\n\x00").is_err());
    }

    #[test]
    fn buffer_too_small_rejected() {
        assert!(encode_ppm(&[0; 2], 4, 4).is_err());
    }
}
