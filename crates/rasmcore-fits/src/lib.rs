//! Pure Rust FITS (Flexible Image Transport System) codec.
//!
//! Implements read and write for FITS image HDUs per the FITS 4.0 standard.
//! Supports BITPIX 8, 16, 32 (integer) and -32, -64 (IEEE float).
//! Zero external dependencies, WASM-ready.
//!
//! # Format overview
//!
//! FITS files consist of Header Data Units (HDUs). Each HDU has:
//! - **Header**: 80-character keyword=value "cards" in 2880-byte blocks
//! - **Data**: N-dimensional binary array in big-endian row-major order
//!
//! Common header keywords:
//! - SIMPLE: must be T for conforming FITS
//! - BITPIX: bits per pixel (8, 16, 32, -32, -64)
//! - NAXIS: number of dimensions
//! - NAXIS1, NAXIS2: width, height
//! - BZERO, BSCALE: linear transform (physical = BZERO + BSCALE * stored)

use std::fmt;

const FITS_BLOCK_SIZE: usize = 2880;
const FITS_CARD_SIZE: usize = 80;

/// FITS error.
#[derive(Debug)]
pub enum FitsError {
    InvalidFormat(String),
    UnsupportedBitpix(i32),
    InvalidHeader(String),
    DataTruncated,
    BufferTooSmall,
}

impl fmt::Display for FitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(m) => write!(f, "invalid FITS: {m}"),
            Self::UnsupportedBitpix(b) => write!(f, "unsupported BITPIX: {b}"),
            Self::InvalidHeader(m) => write!(f, "invalid FITS header: {m}"),
            Self::DataTruncated => write!(f, "FITS data truncated"),
            Self::BufferTooSmall => write!(f, "pixel buffer too small"),
        }
    }
}

impl std::error::Error for FitsError {}

/// FITS pixel data type (BITPIX keyword).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitpix {
    U8 = 8,
    I16 = 16,
    I32 = 32,
    F32 = -32,
    F64 = -64,
}

impl Bitpix {
    fn from_i32(v: i32) -> Result<Self, FitsError> {
        match v {
            8 => Ok(Self::U8),
            16 => Ok(Self::I16),
            32 => Ok(Self::I32),
            -32 => Ok(Self::F32),
            -64 => Ok(Self::F64),
            _ => Err(FitsError::UnsupportedBitpix(v)),
        }
    }

    fn bytes_per_pixel(self) -> usize {
        (self as i32).unsigned_abs() as usize / 8
    }
}

/// Decoded FITS header.
#[derive(Debug, Clone)]
pub struct FitsHeader {
    pub width: u32,
    pub height: u32,
    pub bitpix: Bitpix,
    pub bzero: f64,
    pub bscale: f64,
    /// All header keyword-value pairs.
    pub cards: Vec<(String, String)>,
}

// ─── Detect ────────────────────────────────────────────────────────────────

/// Check if data begins with a valid FITS header.
pub fn is_fits(data: &[u8]) -> bool {
    data.len() >= 30 && data.starts_with(b"SIMPLE  =")
}

// ─── Decode ────────────────────────────────────────────────────────────────

/// Decode a FITS file to a normalized pixel buffer.
///
/// Returns (header, pixels) where pixels are f64 values with BZERO/BSCALE applied.
/// For display, use `to_gray8()` or `to_gray16()` to map to integer pixels.
pub fn decode(data: &[u8]) -> Result<(FitsHeader, Vec<f64>), FitsError> {
    if !is_fits(data) {
        return Err(FitsError::InvalidFormat("missing SIMPLE keyword".into()));
    }

    let (header, data_offset) = parse_header(data)?;
    let pixel_count = header.width as usize * header.height as usize;
    let data_size = pixel_count * header.bitpix.bytes_per_pixel();

    if data_offset + data_size > data.len() {
        return Err(FitsError::DataTruncated);
    }

    let raw = &data[data_offset..data_offset + data_size];
    let mut pixels = Vec::with_capacity(pixel_count);

    match header.bitpix {
        Bitpix::U8 => {
            for &b in raw {
                pixels.push(header.bzero + header.bscale * b as f64);
            }
        }
        Bitpix::I16 => {
            for chunk in raw.chunks_exact(2) {
                let v = i16::from_be_bytes(chunk.try_into().unwrap());
                pixels.push(header.bzero + header.bscale * v as f64);
            }
        }
        Bitpix::I32 => {
            for chunk in raw.chunks_exact(4) {
                let v = i32::from_be_bytes(chunk.try_into().unwrap());
                pixels.push(header.bzero + header.bscale * v as f64);
            }
        }
        Bitpix::F32 => {
            for chunk in raw.chunks_exact(4) {
                let v = f32::from_be_bytes(chunk.try_into().unwrap());
                pixels.push(header.bzero + header.bscale * v as f64);
            }
        }
        Bitpix::F64 => {
            for chunk in raw.chunks_exact(8) {
                let v = f64::from_be_bytes(chunk.try_into().unwrap());
                pixels.push(header.bzero + header.bscale * v);
            }
        }
    }

    Ok((header, pixels))
}

/// Convert f64 pixel values to u8 (0-255) using min/max normalization.
pub fn to_gray8(pixels: &[f64]) -> Vec<u8> {
    if pixels.is_empty() {
        return Vec::new();
    }
    let min = pixels.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = pixels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < f64::EPSILON {
        1.0
    } else {
        max - min
    };
    pixels
        .iter()
        .map(|&v| ((v - min) / range * 255.0).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Convert f64 pixel values to u16 (0-65535) using min/max normalization.
pub fn to_gray16(pixels: &[f64]) -> Vec<u16> {
    if pixels.is_empty() {
        return Vec::new();
    }
    let min = pixels.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = pixels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < f64::EPSILON {
        1.0
    } else {
        max - min
    };
    pixels
        .iter()
        .map(|&v| ((v - min) / range * 65535.0).clamp(0.0, 65535.0) as u16)
        .collect()
}

// ─── Encode ────────────────────────────────────────────────────────────────

/// Encode u8 grayscale pixels to FITS (BITPIX=8).
pub fn encode_u8(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, FitsError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(FitsError::BufferTooSmall);
    }
    let header = build_header(width, height, Bitpix::U8, &[]);
    let mut out = Vec::with_capacity(header.len() + pad_to_block(expected));
    out.extend_from_slice(&header);
    out.extend_from_slice(&pixels[..expected]);
    pad_block(&mut out);
    Ok(out)
}

/// Encode i16 grayscale pixels to FITS (BITPIX=16).
pub fn encode_i16(pixels: &[i16], width: u32, height: u32) -> Result<Vec<u8>, FitsError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(FitsError::BufferTooSmall);
    }
    let header = build_header(width, height, Bitpix::I16, &[]);
    let mut out = Vec::with_capacity(header.len() + pad_to_block(expected * 2));
    out.extend_from_slice(&header);
    for &v in &pixels[..expected] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    pad_block(&mut out);
    Ok(out)
}

/// Encode i32 grayscale pixels to FITS (BITPIX=32).
pub fn encode_i32(pixels: &[i32], width: u32, height: u32) -> Result<Vec<u8>, FitsError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(FitsError::BufferTooSmall);
    }
    let header = build_header(width, height, Bitpix::I32, &[]);
    let mut out = Vec::with_capacity(header.len() + pad_to_block(expected * 4));
    out.extend_from_slice(&header);
    for &v in &pixels[..expected] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    pad_block(&mut out);
    Ok(out)
}

/// Encode f32 grayscale pixels to FITS (BITPIX=-32).
pub fn encode_f32(pixels: &[f32], width: u32, height: u32) -> Result<Vec<u8>, FitsError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(FitsError::BufferTooSmall);
    }
    let header = build_header(width, height, Bitpix::F32, &[]);
    let mut out = Vec::with_capacity(header.len() + pad_to_block(expected * 4));
    out.extend_from_slice(&header);
    for &v in &pixels[..expected] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    pad_block(&mut out);
    Ok(out)
}

/// Encode f64 grayscale pixels to FITS (BITPIX=-64).
pub fn encode_f64(pixels: &[f64], width: u32, height: u32) -> Result<Vec<u8>, FitsError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(FitsError::BufferTooSmall);
    }
    let header = build_header(width, height, Bitpix::F64, &[]);
    let mut out = Vec::with_capacity(header.len() + pad_to_block(expected * 8));
    out.extend_from_slice(&header);
    for &v in &pixels[..expected] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    pad_block(&mut out);
    Ok(out)
}

// ─── Header parsing ───────────────────────────────────────────────────────

fn parse_header(data: &[u8]) -> Result<(FitsHeader, usize), FitsError> {
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bitpix_val = 0i32;
    let mut naxis = 0u32;
    let mut bzero = 0.0f64;
    let mut bscale = 1.0f64;
    let mut cards = Vec::new();
    let mut pos = 0;
    let mut found_end = false;

    while pos + FITS_CARD_SIZE <= data.len() {
        let card = &data[pos..pos + FITS_CARD_SIZE];
        pos += FITS_CARD_SIZE;

        let card_str = std::str::from_utf8(card).unwrap_or("");
        let keyword = card_str[..8].trim();

        if keyword == "END" {
            found_end = true;
            // Advance to next block boundary
            pos = pos.div_ceil(FITS_BLOCK_SIZE) * FITS_BLOCK_SIZE;
            break;
        }

        if card_str.len() > 10 && &card_str[8..10] == "= " {
            let value_comment = &card_str[10..];
            let value = value_comment
                .split('/')
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('\'')
                .trim();

            cards.push((keyword.to_string(), value.to_string()));

            match keyword {
                "BITPIX" => bitpix_val = value.parse().unwrap_or(0),
                "NAXIS" => naxis = value.parse().unwrap_or(0),
                "NAXIS1" => width = value.parse().unwrap_or(0),
                "NAXIS2" => height = value.parse().unwrap_or(0),
                "BZERO" => bzero = value.parse().unwrap_or(0.0),
                "BSCALE" => bscale = value.parse().unwrap_or(1.0),
                _ => {}
            }
        }
    }

    if !found_end {
        return Err(FitsError::InvalidHeader("missing END card".into()));
    }
    if naxis < 2 {
        return Err(FitsError::InvalidHeader(format!("NAXIS={naxis}, need >=2")));
    }

    let bitpix = Bitpix::from_i32(bitpix_val)?;

    Ok((
        FitsHeader {
            width,
            height,
            bitpix,
            bzero,
            bscale,
            cards,
        },
        pos,
    ))
}

// ─── Header building ──────────────────────────────────────────────────────

fn build_header(width: u32, height: u32, bitpix: Bitpix, extra_cards: &[(&str, &str)]) -> Vec<u8> {
    let mut header = Vec::with_capacity(FITS_BLOCK_SIZE);

    write_card(&mut header, "SIMPLE", "T");
    write_card(&mut header, "BITPIX", &(bitpix as i32).to_string());
    write_card(&mut header, "NAXIS", "2");
    write_card(&mut header, "NAXIS1", &width.to_string());
    write_card(&mut header, "NAXIS2", &height.to_string());

    for &(key, val) in extra_cards {
        write_card(&mut header, key, val);
    }

    // END card
    let mut end_card = [b' '; FITS_CARD_SIZE];
    end_card[..3].copy_from_slice(b"END");
    header.extend_from_slice(&end_card);

    // Pad header to 2880-byte boundary
    pad_block(&mut header);
    header
}

fn write_card(buf: &mut Vec<u8>, keyword: &str, value: &str) {
    let mut card = [b' '; FITS_CARD_SIZE];
    let kw = keyword.as_bytes();
    card[..kw.len().min(8)].copy_from_slice(&kw[..kw.len().min(8)]);
    card[8] = b'=';
    card[9] = b' ';

    // Right-justify numeric values in columns 11-30
    let val_bytes = value.as_bytes();
    let start = 30usize.saturating_sub(val_bytes.len());
    let start = start.max(10);
    card[start..start + val_bytes.len().min(70)]
        .copy_from_slice(&val_bytes[..val_bytes.len().min(70)]);

    buf.extend_from_slice(&card);
}

fn pad_block(buf: &mut Vec<u8>) {
    let remainder = buf.len() % FITS_BLOCK_SIZE;
    if remainder != 0 {
        buf.extend(std::iter::repeat_n(0u8, FITS_BLOCK_SIZE - remainder));
    }
}

fn pad_to_block(size: usize) -> usize {
    size.div_ceil(FITS_BLOCK_SIZE) * FITS_BLOCK_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_u8() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let encoded = encode_u8(&pixels, 64, 64).unwrap();

        assert!(is_fits(&encoded));
        assert_eq!(encoded.len() % FITS_BLOCK_SIZE, 0);

        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 64);
        assert_eq!(header.height, 64);
        assert_eq!(header.bitpix, Bitpix::U8);

        // u8 roundtrip should be exact (no scaling by default)
        let gray8 = to_gray8(&decoded);
        assert_eq!(gray8, pixels);
    }

    #[test]
    fn roundtrip_i16() {
        let pixels: Vec<i16> = (0..32 * 32).map(|i| (i * 10 - 5000) as i16).collect();
        let encoded = encode_i16(&pixels, 32, 32).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bitpix, Bitpix::I16);
        // i16 → f64 is exact (all i16 values are representable in f64)
        for (i, &expected) in pixels.iter().enumerate() {
            assert_eq!(
                decoded[i], expected as f64,
                "pixel {i}: {} vs {}",
                decoded[i], expected
            );
        }
    }

    #[test]
    fn roundtrip_i32() {
        let pixels: Vec<i32> = (0..16 * 16).map(|i| i * 1000 - 128000).collect();
        let encoded = encode_i32(&pixels, 16, 16).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bitpix, Bitpix::I32);
        // i32 → f64 is exact (all i32 values are representable in f64)
        for (i, &expected) in pixels.iter().enumerate() {
            assert_eq!(
                decoded[i], expected as f64,
                "pixel {i}: {} vs {}",
                decoded[i], expected
            );
        }
    }

    #[test]
    fn roundtrip_f32() {
        let pixels: Vec<f32> = (0..16 * 16).map(|i| i as f32 * 0.5).collect();
        let encoded = encode_f32(&pixels, 16, 16).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bitpix, Bitpix::F32);
        // f32 → f64 has representational noise: f32 values like 0.5 are exact,
        // but values like 0.1 become 0.10000000149... in f64. The epsilon of
        // 1e-5 accounts for this precision loss in the f32→bytes→f32→f64 chain.
        for (i, &expected) in pixels.iter().enumerate() {
            assert!(
                (decoded[i] - expected as f64).abs() < 1e-5,
                "pixel {i}: {} vs {}",
                decoded[i],
                expected
            );
        }
    }

    #[test]
    fn roundtrip_f64() {
        let pixels: Vec<f64> = (0..8 * 8).map(|i| i as f64 * 1.23456789).collect();
        let encoded = encode_f64(&pixels, 8, 8).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bitpix, Bitpix::F64);
        // f64 → bytes → f64 is IEEE 754 lossless — must be bit-exact
        for (i, &expected) in pixels.iter().enumerate() {
            assert_eq!(
                decoded[i].to_bits(),
                expected.to_bits(),
                "pixel {i}: {} vs {} (bits: {:016x} vs {:016x})",
                decoded[i],
                expected,
                decoded[i].to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn bzero_bscale_applied() {
        // Manually construct FITS with BZERO=100, BSCALE=2
        let mut header_data = Vec::new();
        write_card(&mut header_data, "SIMPLE", "T");
        write_card(&mut header_data, "BITPIX", "8");
        write_card(&mut header_data, "NAXIS", "2");
        write_card(&mut header_data, "NAXIS1", "2");
        write_card(&mut header_data, "NAXIS2", "2");
        write_card(&mut header_data, "BZERO", "100.0");
        write_card(&mut header_data, "BSCALE", "2.0");
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        header_data.extend_from_slice(&end);
        pad_block(&mut header_data);

        // 4 pixels: [0, 1, 10, 50]
        header_data.extend_from_slice(&[0, 1, 10, 50]);
        pad_block(&mut header_data);

        let (header, pixels) = decode(&header_data).unwrap();
        assert_eq!(header.bzero, 100.0);
        assert_eq!(header.bscale, 2.0);
        // physical = BZERO + BSCALE * stored (integer inputs → exact f64)
        assert_eq!(pixels[0], 100.0); // 100 + 2*0
        assert_eq!(pixels[1], 102.0); // 100 + 2*1
        assert_eq!(pixels[2], 120.0); // 100 + 2*10
        assert_eq!(pixels[3], 200.0); // 100 + 2*50
    }

    #[test]
    fn block_alignment() {
        let encoded = encode_u8(&[0; 1], 1, 1).unwrap();
        assert_eq!(encoded.len() % FITS_BLOCK_SIZE, 0);
        // Header = 1 block, Data = 1 block (1 byte padded to 2880)
        assert_eq!(encoded.len(), FITS_BLOCK_SIZE * 2);
    }

    #[test]
    fn is_fits_detection() {
        assert!(is_fits(
            b"SIMPLE  =                    T / conforming FITS file"
        ));
        assert!(!is_fits(b"not a FITS file"));
        assert!(!is_fits(b""));
    }

    #[test]
    fn to_gray8_normalization() {
        let pixels = vec![0.0, 50.0, 100.0];
        let gray = to_gray8(&pixels);
        assert_eq!(gray[0], 0);
        assert_eq!(gray[1], 127); // 50/100 * 255 ≈ 127
        assert_eq!(gray[2], 255);
    }

    #[test]
    fn to_gray16_normalization() {
        let pixels = vec![0.0, 1.0];
        let gray = to_gray16(&pixels);
        assert_eq!(gray[0], 0);
        assert_eq!(gray[1], 65535);
    }

    #[test]
    fn header_keywords_preserved() {
        let encoded = encode_u8(&[128; 4], 2, 2).unwrap();
        let (header, _) = decode(&encoded).unwrap();
        assert!(header.cards.iter().any(|(k, _)| k == "SIMPLE"));
        assert!(header.cards.iter().any(|(k, _)| k == "NAXIS1"));
    }

    #[test]
    fn invalid_fits_rejected() {
        assert!(decode(b"not fits data at all").is_err());
    }

    #[test]
    fn buffer_too_small_rejected() {
        assert!(encode_u8(&[0; 2], 10, 10).is_err());
    }
}
