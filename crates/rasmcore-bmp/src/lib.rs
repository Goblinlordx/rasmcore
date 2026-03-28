//! Pure Rust BMP encoder/decoder.
//!
//! Supports uncompressed 24-bit (RGB) and 32-bit (RGBA) BMP files.
//! Uses BITMAPINFOHEADER (V1, 40 bytes). Zero external dependencies.

use std::fmt;

const BMP_MAGIC: [u8; 2] = *b"BM";
const BMP_FILE_HEADER_SIZE: usize = 14;
const BMP_INFO_HEADER_SIZE: u32 = 40;

/// BMP error.
#[derive(Debug)]
pub enum BmpError {
    InvalidMagic,
    InvalidHeader(String),
    UnsupportedFormat(String),
    BufferTooSmall,
}

impl fmt::Display for BmpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid BMP magic"),
            Self::InvalidHeader(m) => write!(f, "invalid BMP header: {m}"),
            Self::UnsupportedFormat(m) => write!(f, "unsupported BMP format: {m}"),
            Self::BufferTooSmall => write!(f, "pixel buffer too small"),
        }
    }
}

impl std::error::Error for BmpError {}

/// BMP header information.
#[derive(Debug, Clone)]
pub struct BmpHeader {
    pub width: u32,
    pub height: u32,
    pub bits_per_pixel: u16,
    pub top_down: bool,
}

/// Encode RGB pixels to 24-bit BMP.
///
/// BMP stores pixels bottom-up, BGR order, with row padding to 4-byte alignment.
pub fn encode_rgb(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, BmpError> {
    let expected = width as usize * height as usize * 3;
    if pixels.len() < expected {
        return Err(BmpError::BufferTooSmall);
    }
    encode_bmp(pixels, width, height, 3)
}

/// Encode RGBA pixels to 32-bit BMP.
pub fn encode_rgba(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, BmpError> {
    let expected = width as usize * height as usize * 4;
    if pixels.len() < expected {
        return Err(BmpError::BufferTooSmall);
    }
    encode_bmp(pixels, width, height, 4)
}

fn encode_bmp(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<Vec<u8>, BmpError> {
    let bpp = (channels * 8) as u16;
    let row_size = (width as usize * channels + 3) & !3; // 4-byte aligned
    let pixel_data_size = row_size * height as usize;
    let file_size = BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE as usize + pixel_data_size;
    let data_offset = (BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE as usize) as u32;

    let mut out = Vec::with_capacity(file_size);

    // File header (14 bytes)
    out.extend_from_slice(&BMP_MAGIC);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // reserved
    out.extend_from_slice(&data_offset.to_le_bytes());

    // Info header (40 bytes - BITMAPINFOHEADER)
    out.extend_from_slice(&BMP_INFO_HEADER_SIZE.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes()); // positive = bottom-up
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&bpp.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // compression = BI_RGB
    out.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes()); // h_res (72 DPI)
    out.extend_from_slice(&2835u32.to_le_bytes()); // v_res
    out.extend_from_slice(&0u32.to_le_bytes()); // colors used
    out.extend_from_slice(&0u32.to_le_bytes()); // important colors

    // Pixel data (bottom-up, BGR/BGRA order)
    let w = width as usize;
    let padding = row_size - w * channels;

    for row in (0..height as usize).rev() {
        for col in 0..w {
            let src = (row * w + col) * channels;
            // Convert RGB(A) → BGR(A)
            out.push(pixels[src + 2]); // B
            out.push(pixels[src + 1]); // G
            out.push(pixels[src]); // R
            if channels == 4 {
                out.push(pixels[src + 3]); // A
            }
        }
        out.extend(std::iter::repeat_n(0u8, padding));
    }

    Ok(out)
}

/// Decode a BMP file to raw RGB or RGBA pixels (top-down, RGB order).
pub fn decode(data: &[u8]) -> Result<(BmpHeader, Vec<u8>), BmpError> {
    if data.len() < BMP_FILE_HEADER_SIZE + 4 {
        return Err(BmpError::InvalidHeader("file too small".into()));
    }
    if data[0..2] != BMP_MAGIC {
        return Err(BmpError::InvalidMagic);
    }

    let data_offset = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
    let info_size = u32::from_le_bytes(data[14..18].try_into().unwrap());
    if info_size < 40 || data.len() < 14 + info_size as usize {
        return Err(BmpError::InvalidHeader("info header too small".into()));
    }

    let width = i32::from_le_bytes(data[18..22].try_into().unwrap());
    let height_raw = i32::from_le_bytes(data[22..26].try_into().unwrap());
    let bpp = u16::from_le_bytes(data[28..30].try_into().unwrap());
    let compression = u32::from_le_bytes(data[30..34].try_into().unwrap());

    if compression != 0 {
        return Err(BmpError::UnsupportedFormat(format!(
            "compression type {compression}"
        )));
    }

    let top_down = height_raw < 0;
    let width = width.unsigned_abs();
    let height = height_raw.unsigned_abs();
    let channels: usize = match bpp {
        24 => 3,
        32 => 4,
        _ => {
            return Err(BmpError::UnsupportedFormat(format!("{bpp} bits per pixel")));
        }
    };

    let row_size = (width as usize * channels + 3) & !3;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * channels);

    for row in 0..height as usize {
        let src_row = if top_down {
            row
        } else {
            height as usize - 1 - row
        };
        let row_start = data_offset + src_row * row_size;
        for col in 0..width as usize {
            let off = row_start + col * channels;
            if off + channels > data.len() {
                return Err(BmpError::InvalidHeader("pixel data truncated".into()));
            }
            // BGR(A) → RGB(A)
            pixels.push(data[off + 2]); // R
            pixels.push(data[off + 1]); // G
            pixels.push(data[off]); // B
            if channels == 4 {
                pixels.push(data[off + 3]); // A
            }
        }
    }

    Ok((
        BmpHeader {
            width,
            height,
            bits_per_pixel: bpp,
            top_down,
        },
        pixels,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_rgb() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgb(&pixels, 8, 8).unwrap();
        assert_eq!(&encoded[..2], b"BM");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 8);
        assert_eq!(header.height, 8);
        assert_eq!(header.bits_per_pixel, 24);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_rgba() {
        let pixels: Vec<u8> = (0..4 * 4 * 4).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgba(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bits_per_pixel, 32);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn row_padding_applied() {
        // 3-pixel wide RGB: 9 bytes/row → padded to 12
        let pixels = vec![128u8; 3 * 2 * 3];
        let encoded = encode_rgb(&pixels, 3, 2).unwrap();
        let (_, decoded) = decode(&encoded).unwrap();
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn invalid_magic_rejected() {
        assert!(
            decode(b"XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").is_err()
        );
    }
}
