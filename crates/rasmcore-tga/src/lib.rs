//! Pure Rust TGA (Truevision TARGA) encoder/decoder.
//!
//! Supports uncompressed and RLE-compressed 24-bit (RGB) and 32-bit (RGBA).
//! Zero external dependencies, WASM-ready.

use std::fmt;

const TGA_HEADER_SIZE: usize = 18;

/// TGA image type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TgaImageType {
    Uncompressed = 2,
    RleCompressed = 10,
}

/// TGA error.
#[derive(Debug)]
pub enum TgaError {
    InvalidHeader(String),
    UnsupportedFormat(String),
    BufferTooSmall,
    InvalidData(String),
}

impl fmt::Display for TgaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader(m) => write!(f, "invalid TGA header: {m}"),
            Self::UnsupportedFormat(m) => write!(f, "unsupported TGA format: {m}"),
            Self::BufferTooSmall => write!(f, "pixel buffer too small"),
            Self::InvalidData(m) => write!(f, "invalid TGA data: {m}"),
        }
    }
}

impl std::error::Error for TgaError {}

/// TGA header.
#[derive(Debug, Clone)]
pub struct TgaHeader {
    pub width: u16,
    pub height: u16,
    pub bits_per_pixel: u8,
    pub image_type: TgaImageType,
    pub top_down: bool,
}

/// Encode RGB pixels to uncompressed TGA.
pub fn encode_rgb(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 3;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_tga(pixels, width, height, 24, TgaImageType::Uncompressed)
}

/// Encode RGBA pixels to uncompressed TGA.
pub fn encode_rgba(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 4;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_tga(pixels, width, height, 32, TgaImageType::Uncompressed)
}

fn encode_tga(
    pixels: &[u8],
    width: u16,
    height: u16,
    bpp: u8,
    image_type: TgaImageType,
) -> Result<Vec<u8>, TgaError> {
    let channels = (bpp / 8) as usize;
    let pixel_count = width as usize * height as usize;
    let mut out = Vec::with_capacity(TGA_HEADER_SIZE + pixel_count * channels);

    // TGA header (18 bytes)
    out.push(0); // id_length
    out.push(0); // color_map_type
    out.push(image_type as u8);
    out.extend_from_slice(&[0u8; 5]); // color map spec
    out.extend_from_slice(&0u16.to_le_bytes()); // x_origin
    out.extend_from_slice(&0u16.to_le_bytes()); // y_origin
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.push(bpp);
    out.push(0x20); // image descriptor: top-down origin

    // Pixel data (top-down, BGR/BGRA order)
    for i in 0..pixel_count {
        let off = i * channels;
        out.push(pixels[off + 2]); // B
        out.push(pixels[off + 1]); // G
        out.push(pixels[off]); // R
        if channels == 4 {
            out.push(pixels[off + 3]); // A
        }
    }

    Ok(out)
}

/// Decode a TGA file to raw RGB or RGBA pixels (top-down, RGB order).
pub fn decode(data: &[u8]) -> Result<(TgaHeader, Vec<u8>), TgaError> {
    if data.len() < TGA_HEADER_SIZE {
        return Err(TgaError::InvalidHeader("file too small".into()));
    }

    let id_length = data[0] as usize;
    let image_type_byte = data[2];
    let width = u16::from_le_bytes(data[12..14].try_into().unwrap());
    let height = u16::from_le_bytes(data[14..16].try_into().unwrap());
    let bpp = data[16];
    let descriptor = data[17];
    let top_down = (descriptor & 0x20) != 0;

    let image_type = match image_type_byte {
        2 => TgaImageType::Uncompressed,
        10 => TgaImageType::RleCompressed,
        _ => {
            return Err(TgaError::UnsupportedFormat(format!(
                "image type {image_type_byte}"
            )));
        }
    };

    let channels = match bpp {
        24 => 3usize,
        32 => 4,
        _ => {
            return Err(TgaError::UnsupportedFormat(format!("{bpp} bpp")));
        }
    };

    let pixel_count = width as usize * height as usize;
    let data_start = TGA_HEADER_SIZE + id_length;

    let raw_pixels = match image_type {
        TgaImageType::Uncompressed => {
            let needed = data_start + pixel_count * channels;
            if data.len() < needed {
                return Err(TgaError::InvalidData("not enough pixel data".into()));
            }
            data[data_start..data_start + pixel_count * channels].to_vec()
        }
        TgaImageType::RleCompressed => decode_rle(&data[data_start..], pixel_count, channels)?,
    };

    // Convert BGR(A) → RGB(A) and handle scanline order
    let mut pixels = Vec::with_capacity(pixel_count * channels);
    for row in 0..height as usize {
        let src_row = if top_down {
            row
        } else {
            height as usize - 1 - row
        };
        for col in 0..width as usize {
            let off = (src_row * width as usize + col) * channels;
            pixels.push(raw_pixels[off + 2]); // R
            pixels.push(raw_pixels[off + 1]); // G
            pixels.push(raw_pixels[off]); // B
            if channels == 4 {
                pixels.push(raw_pixels[off + 3]); // A
            }
        }
    }

    Ok((
        TgaHeader {
            width,
            height,
            bits_per_pixel: bpp,
            image_type,
            top_down,
        },
        pixels,
    ))
}

/// Decode RLE-compressed TGA pixel data.
fn decode_rle(data: &[u8], pixel_count: usize, channels: usize) -> Result<Vec<u8>, TgaError> {
    let mut out = Vec::with_capacity(pixel_count * channels);
    let mut pos = 0;

    while out.len() < pixel_count * channels {
        if pos >= data.len() {
            return Err(TgaError::InvalidData("RLE data truncated".into()));
        }
        let header = data[pos];
        pos += 1;
        let count = (header & 0x7F) as usize + 1;

        if header & 0x80 != 0 {
            // RLE packet: one pixel repeated `count` times
            if pos + channels > data.len() {
                return Err(TgaError::InvalidData("RLE pixel data truncated".into()));
            }
            let pixel = &data[pos..pos + channels];
            pos += channels;
            for _ in 0..count {
                out.extend_from_slice(pixel);
            }
        } else {
            // Raw packet: `count` pixels
            let needed = count * channels;
            if pos + needed > data.len() {
                return Err(TgaError::InvalidData("raw packet data truncated".into()));
            }
            out.extend_from_slice(&data[pos..pos + needed]);
            pos += needed;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_rgb() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgb(&pixels, 8, 8).unwrap();
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
    fn top_down_flag() {
        let pixels = vec![128u8; 2 * 2 * 3];
        let encoded = encode_rgb(&pixels, 2, 2).unwrap();
        let (header, _) = decode(&encoded).unwrap();
        assert!(header.top_down);
    }

    #[test]
    fn file_too_small_rejected() {
        assert!(decode(&[0u8; 5]).is_err());
    }

    #[test]
    fn buffer_too_small_rejected() {
        assert!(encode_rgb(&[0; 2], 4, 4).is_err());
    }
}
