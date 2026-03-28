//! Pure Rust BMP encoder/decoder.
//!
//! Supports:
//! - Uncompressed 1/4/8/16/24/32-bit pixel data
//! - Palette/indexed color (1, 4, 8 bits per pixel)
//! - RLE8 and RLE4 compression (decode)
//! - BI_BITFIELDS 16/32-bit with arbitrary channel masks
//! - BITMAPINFOHEADER (V1) through BITMAPV5HEADER
//! - Top-down and bottom-up scanline order
//!
//! Uses rasmcore-bitio for sub-byte pixel unpacking.

use rasmcore_bitio::{BitOrder, BitReader};
use std::fmt;

const BMP_MAGIC: [u8; 2] = *b"BM";
const BMP_FILE_HEADER_SIZE: usize = 14;
const BMP_INFO_HEADER_SIZE: u32 = 40;

// Compression types
const BI_RGB: u32 = 0;
const BI_RLE8: u32 = 1;
const BI_RLE4: u32 = 2;
const BI_BITFIELDS: u32 = 3;

// RLE escape codes
const RLE_ESCAPE: u8 = 0;
const RLE_ESCAPE_EOL: u8 = 0;
const RLE_ESCAPE_EOF: u8 = 1;
const RLE_ESCAPE_DELTA: u8 = 2;

/// BMP error.
#[derive(Debug)]
pub enum BmpError {
    InvalidMagic,
    InvalidHeader(String),
    UnsupportedFormat(String),
    BufferTooSmall,
    CorruptData(String),
}

impl fmt::Display for BmpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid BMP magic"),
            Self::InvalidHeader(m) => write!(f, "invalid BMP header: {m}"),
            Self::UnsupportedFormat(m) => write!(f, "unsupported BMP format: {m}"),
            Self::BufferTooSmall => write!(f, "pixel buffer too small"),
            Self::CorruptData(m) => write!(f, "corrupt BMP data: {m}"),
        }
    }
}

impl std::error::Error for BmpError {}

/// Decoded BMP header information.
#[derive(Debug, Clone)]
pub struct BmpHeader {
    pub width: u32,
    pub height: u32,
    pub bits_per_pixel: u16,
    pub compression: u32,
    pub top_down: bool,
}

/// Encode grayscale pixels to 8-bit palette BMP.
///
/// Creates a 256-entry grayscale palette where index N maps to (N, N, N).
pub fn encode_gray(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, BmpError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(BmpError::BufferTooSmall);
    }

    let palette_size = 256 * 4; // 256 BGRA entries
    let row_size = (width as usize + 3) & !3; // 4-byte aligned
    let pixel_data_size = row_size * height as usize;
    let data_offset = BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE as usize + palette_size;
    let file_size = data_offset + pixel_data_size;

    let mut out = Vec::with_capacity(file_size);

    // File header
    out.extend_from_slice(&BMP_MAGIC);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&(data_offset as u32).to_le_bytes());

    // Info header
    out.extend_from_slice(&BMP_INFO_HEADER_SIZE.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&8u16.to_le_bytes()); // 8 bpp
    out.extend_from_slice(&BI_RGB.to_le_bytes());
    out.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&256u32.to_le_bytes()); // colors used
    out.extend_from_slice(&0u32.to_le_bytes());

    // Grayscale palette: 256 entries, each (i, i, i, 0)
    for i in 0..256u32 {
        let v = i as u8;
        out.extend_from_slice(&[v, v, v, 0]); // BGRA
    }

    // Pixel data: bottom-up, 1 byte per pixel (palette index = gray value)
    let w = width as usize;
    let padding = row_size - w;
    for row in (0..height as usize).rev() {
        out.extend_from_slice(&pixels[row * w..(row + 1) * w]);
        out.extend(std::iter::repeat_n(0u8, padding));
    }

    Ok(out)
}

/// Encode RGB pixels to 24-bit uncompressed BMP.
pub fn encode_rgb(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, BmpError> {
    let expected = width as usize * height as usize * 3;
    if pixels.len() < expected {
        return Err(BmpError::BufferTooSmall);
    }
    encode_bmp_uncompressed(pixels, width, height, 3)
}

/// Encode RGBA pixels to 32-bit uncompressed BMP.
pub fn encode_rgba(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, BmpError> {
    let expected = width as usize * height as usize * 4;
    if pixels.len() < expected {
        return Err(BmpError::BufferTooSmall);
    }
    encode_bmp_uncompressed(pixels, width, height, 4)
}

fn encode_bmp_uncompressed(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<Vec<u8>, BmpError> {
    let bpp = (channels * 8) as u16;
    let row_size = (width as usize * channels + 3) & !3;
    let pixel_data_size = row_size * height as usize;
    let file_size = BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE as usize + pixel_data_size;
    let data_offset = (BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE as usize) as u32;

    let mut out = Vec::with_capacity(file_size);

    // File header (14 bytes)
    out.extend_from_slice(&BMP_MAGIC);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&data_offset.to_le_bytes());

    // BITMAPINFOHEADER (40 bytes)
    out.extend_from_slice(&BMP_INFO_HEADER_SIZE.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&bpp.to_le_bytes());
    out.extend_from_slice(&BI_RGB.to_le_bytes());
    out.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes()); // h_res
    out.extend_from_slice(&2835u32.to_le_bytes()); // v_res
    out.extend_from_slice(&0u32.to_le_bytes()); // colors_used
    out.extend_from_slice(&0u32.to_le_bytes()); // colors_important

    // Pixel data: bottom-up, BGR(A) order
    let w = width as usize;
    let padding = row_size - w * channels;
    for row in (0..height as usize).rev() {
        for col in 0..w {
            let src = (row * w + col) * channels;
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

/// Decode a BMP file to raw RGBA pixels (top-down, RGBA order).
///
/// All input formats (1/4/8/16/24/32-bit, palette, RLE, bitfields)
/// are normalized to RGBA output.
pub fn decode(data: &[u8]) -> Result<(BmpHeader, Vec<u8>), BmpError> {
    if data.len() < BMP_FILE_HEADER_SIZE + 12 {
        return Err(BmpError::InvalidHeader("file too small".into()));
    }
    if data[0..2] != BMP_MAGIC {
        return Err(BmpError::InvalidMagic);
    }

    let data_offset = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
    let info_size = u32::from_le_bytes(data[14..18].try_into().unwrap());
    if data.len() < 14 + info_size as usize {
        return Err(BmpError::InvalidHeader("truncated info header".into()));
    }

    // Support BITMAPCOREHEADER (12 bytes) through BITMAPV5HEADER (124 bytes)
    let (width_raw, height_raw, bpp, compression) = if info_size == 12 {
        // BITMAPCOREHEADER
        let w = u16::from_le_bytes(data[18..20].try_into().unwrap()) as i32;
        let h = u16::from_le_bytes(data[20..22].try_into().unwrap()) as i32;
        let bpp = u16::from_le_bytes(data[24..26].try_into().unwrap());
        (w, h, bpp, BI_RGB)
    } else {
        // BITMAPINFOHEADER and later (>= 40 bytes)
        if info_size < 40 {
            return Err(BmpError::InvalidHeader(format!(
                "unsupported info header size {info_size}"
            )));
        }
        let w = i32::from_le_bytes(data[18..22].try_into().unwrap());
        let h = i32::from_le_bytes(data[22..26].try_into().unwrap());
        let bpp = u16::from_le_bytes(data[28..30].try_into().unwrap());
        let comp = u32::from_le_bytes(data[30..34].try_into().unwrap());
        (w, h, bpp, comp)
    };

    let top_down = height_raw < 0;
    let width = width_raw.unsigned_abs();
    let height = height_raw.unsigned_abs();

    let header = BmpHeader {
        width,
        height,
        bits_per_pixel: bpp,
        compression,
        top_down,
    };

    // Read palette if present (for 1/4/8 bpp)
    let palette = if bpp <= 8 {
        let palette_offset = 14 + info_size as usize;
        let num_colors = 1usize << bpp;
        let entry_size = if info_size == 12 { 3 } else { 4 }; // core=BGR, info+=BGRx
        let mut pal = Vec::with_capacity(num_colors);
        for i in 0..num_colors {
            let off = palette_offset + i * entry_size;
            if off + 3 > data.len() {
                break;
            }
            pal.push([data[off + 2], data[off + 1], data[off], 255]); // BGR→RGBA
        }
        Some(pal)
    } else {
        None
    };

    // Read bitfield masks if BI_BITFIELDS
    let bitfields = if compression == BI_BITFIELDS && info_size >= 40 {
        // Bitfield masks always follow the BITMAPINFOHEADER at offset 54 (14+40).
        let mo = 14 + 40;
        if mo + 12 <= data.len() {
            let r = u32::from_le_bytes(data[mo..mo + 4].try_into().unwrap());
            let g = u32::from_le_bytes(data[mo + 4..mo + 8].try_into().unwrap());
            let b = u32::from_le_bytes(data[mo + 8..mo + 12].try_into().unwrap());
            let a = if mo + 16 <= data.len() {
                u32::from_le_bytes(data[mo + 12..mo + 16].try_into().unwrap())
            } else {
                0
            };
            Some(BitfieldMasks { r, g, b, a })
        } else {
            None
        }
    } else {
        None
    };

    let pixels = match compression {
        BI_RGB => decode_uncompressed(data, data_offset, width, height, bpp, top_down, &palette)?,
        BI_RLE8 => decode_rle(
            data,
            data_offset,
            width,
            height,
            8,
            palette.as_ref().unwrap(),
        )?,
        BI_RLE4 => decode_rle(
            data,
            data_offset,
            width,
            height,
            4,
            palette.as_ref().unwrap(),
        )?,
        BI_BITFIELDS => {
            let bf = bitfields
                .ok_or_else(|| BmpError::InvalidHeader("missing bitfield masks".into()))?;
            decode_bitfields(data, data_offset, width, height, bpp, top_down, &bf)?
        }
        _ => {
            return Err(BmpError::UnsupportedFormat(format!(
                "compression type {compression}"
            )));
        }
    };

    Ok((header, pixels))
}

struct BitfieldMasks {
    r: u32,
    g: u32,
    b: u32,
    a: u32,
}

/// Extract a channel value using a bitmask, normalized to 0-255.
fn extract_channel(pixel: u32, mask: u32) -> u8 {
    if mask == 0 {
        return 255; // default alpha to opaque
    }
    let shift = mask.trailing_zeros();
    let bits = (mask >> shift).count_ones();
    let val = (pixel & mask) >> shift;
    // Scale to 8 bits
    if bits >= 8 {
        (val >> (bits - 8)) as u8
    } else {
        // Replicate bits to fill 8 bits (e.g., 5-bit: abcde → abcdeabc)
        let mut result = val << (8 - bits);
        result |= val >> (2 * bits).saturating_sub(8);
        result as u8
    }
}

fn decode_uncompressed(
    data: &[u8],
    data_offset: usize,
    width: u32,
    height: u32,
    bpp: u16,
    top_down: bool,
    palette: &Option<Vec<[u8; 4]>>,
) -> Result<Vec<u8>, BmpError> {
    let w = width as usize;
    let h = height as usize;
    let row_bits = w * bpp as usize;
    let row_size = row_bits.div_ceil(32) * 4; // 4-byte aligned
    let mut pixels = Vec::with_capacity(w * h * 4);

    for row in 0..h {
        let src_row = if top_down { row } else { h - 1 - row };
        let row_start = data_offset + src_row * row_size;

        match bpp {
            1 | 4 | 8 => {
                let pal = palette
                    .as_ref()
                    .ok_or_else(|| BmpError::InvalidHeader("missing palette".into()))?;

                if bpp == 8 {
                    // Direct byte index
                    for col in 0..w {
                        let off = row_start + col;
                        if off >= data.len() {
                            return Err(BmpError::CorruptData("pixel data truncated".into()));
                        }
                        let idx = data[off] as usize;
                        let color = pal.get(idx).copied().unwrap_or([0, 0, 0, 255]);
                        pixels.extend_from_slice(&color);
                    }
                } else {
                    // Use BitReader for 1-bit and 4-bit unpacking
                    let row_end = (row_start + row_size).min(data.len());
                    if row_start >= data.len() {
                        return Err(BmpError::CorruptData("pixel data truncated".into()));
                    }
                    let mut br = BitReader::new(&data[row_start..row_end], BitOrder::MsbFirst);
                    for _col in 0..w {
                        let idx = br.read_bits(bpp as u8).unwrap_or(0) as usize;
                        let color = pal.get(idx).copied().unwrap_or([0, 0, 0, 255]);
                        pixels.extend_from_slice(&color);
                    }
                }
            }
            16 => {
                // Default 16-bit: 5-5-5 (X1R5G5B5)
                for col in 0..w {
                    let off = row_start + col * 2;
                    if off + 2 > data.len() {
                        return Err(BmpError::CorruptData("pixel data truncated".into()));
                    }
                    let val = u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) as u32;
                    let r = extract_channel(val, 0x7C00);
                    let g = extract_channel(val, 0x03E0);
                    let b = extract_channel(val, 0x001F);
                    pixels.extend_from_slice(&[r, g, b, 255]);
                }
            }
            24 => {
                for col in 0..w {
                    let off = row_start + col * 3;
                    if off + 3 > data.len() {
                        return Err(BmpError::CorruptData("pixel data truncated".into()));
                    }
                    pixels.extend_from_slice(&[data[off + 2], data[off + 1], data[off], 255]);
                }
            }
            32 => {
                for col in 0..w {
                    let off = row_start + col * 4;
                    if off + 4 > data.len() {
                        return Err(BmpError::CorruptData("pixel data truncated".into()));
                    }
                    pixels.extend_from_slice(&[
                        data[off + 2],
                        data[off + 1],
                        data[off],
                        data[off + 3],
                    ]);
                }
            }
            _ => {
                return Err(BmpError::UnsupportedFormat(format!("{bpp} bits per pixel")));
            }
        }
    }

    Ok(pixels)
}

fn decode_bitfields(
    data: &[u8],
    data_offset: usize,
    width: u32,
    height: u32,
    bpp: u16,
    top_down: bool,
    bf: &BitfieldMasks,
) -> Result<Vec<u8>, BmpError> {
    let w = width as usize;
    let h = height as usize;
    let bytes_per_pixel = bpp as usize / 8;
    let row_size = (w * bytes_per_pixel + 3) & !3;
    let mut pixels = Vec::with_capacity(w * h * 4);

    for row in 0..h {
        let src_row = if top_down { row } else { h - 1 - row };
        let row_start = data_offset + src_row * row_size;

        for col in 0..w {
            let off = row_start + col * bytes_per_pixel;
            if off + bytes_per_pixel > data.len() {
                return Err(BmpError::CorruptData("pixel data truncated".into()));
            }
            let val = match bpp {
                16 => u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) as u32,
                32 => u32::from_le_bytes(data[off..off + 4].try_into().unwrap()),
                _ => {
                    return Err(BmpError::UnsupportedFormat(format!(
                        "bitfields with {bpp}bpp"
                    )));
                }
            };
            let r = extract_channel(val, bf.r);
            let g = extract_channel(val, bf.g);
            let b = extract_channel(val, bf.b);
            let a = if bf.a != 0 {
                extract_channel(val, bf.a)
            } else {
                255
            };
            pixels.extend_from_slice(&[r, g, b, a]);
        }
    }

    Ok(pixels)
}

fn decode_rle(
    data: &[u8],
    data_offset: usize,
    width: u32,
    height: u32,
    rle_bpp: u16,
    palette: &[[u8; 4]],
) -> Result<Vec<u8>, BmpError> {
    let w = width as usize;
    let h = height as usize;
    // RLE images are always bottom-up
    let mut pixels = vec![0u8; w * h * 4]; // RGBA, initialized to transparent black
    // Set alpha to 255 for all pixels (background)
    for px in pixels.chunks_exact_mut(4) {
        px[3] = 255;
    }

    let mut x = 0usize;
    let mut y = 0usize; // y=0 is the BOTTOM row in BMP
    let mut pos = data_offset;

    while pos + 1 < data.len() {
        let count = data[pos];
        let value = data[pos + 1];
        pos += 2;

        if count == RLE_ESCAPE {
            match value {
                RLE_ESCAPE_EOL => {
                    x = 0;
                    y += 1;
                }
                RLE_ESCAPE_EOF => break,
                RLE_ESCAPE_DELTA => {
                    if pos + 2 > data.len() {
                        return Err(BmpError::CorruptData("RLE delta truncated".into()));
                    }
                    x += data[pos] as usize;
                    y += data[pos + 1] as usize;
                    pos += 2;
                }
                abs_count => {
                    // Absolute mode: next `abs_count` pixels
                    if rle_bpp == 8 {
                        for _ in 0..abs_count {
                            if pos >= data.len() || x >= w || y >= h {
                                break;
                            }
                            let idx = data[pos] as usize;
                            pos += 1;
                            set_pixel(&mut pixels, w, h, x, y, palette, idx);
                            x += 1;
                        }
                        // Absolute runs are word-aligned
                        if abs_count % 2 == 1 {
                            pos += 1;
                        }
                    } else {
                        // RLE4: nibbles
                        let byte_count = (abs_count as usize).div_ceil(2);
                        let start = pos;
                        for i in 0..abs_count as usize {
                            if pos >= data.len() || x >= w || y >= h {
                                break;
                            }
                            let byte = data[start + i / 2];
                            let idx = if i % 2 == 0 {
                                (byte >> 4) as usize
                            } else {
                                (byte & 0x0F) as usize
                            };
                            set_pixel(&mut pixels, w, h, x, y, palette, idx);
                            x += 1;
                        }
                        pos += byte_count;
                        // Word-align
                        if byte_count % 2 == 1 {
                            pos += 1;
                        }
                    }
                }
            }
        } else {
            // Encoded mode: repeat pixel(s) `count` times
            if rle_bpp == 8 {
                let idx = value as usize;
                for _ in 0..count {
                    if x >= w || y >= h {
                        break;
                    }
                    set_pixel(&mut pixels, w, h, x, y, palette, idx);
                    x += 1;
                }
            } else {
                // RLE4: alternate between high and low nibble
                let hi = (value >> 4) as usize;
                let lo = (value & 0x0F) as usize;
                for i in 0..count as usize {
                    if x >= w || y >= h {
                        break;
                    }
                    let idx = if i % 2 == 0 { hi } else { lo };
                    set_pixel(&mut pixels, w, h, x, y, palette, idx);
                    x += 1;
                }
            }
        }
    }

    Ok(pixels)
}

/// Set a pixel in the output buffer (bottom-up → top-down conversion).
fn set_pixel(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    palette: &[[u8; 4]],
    idx: usize,
) {
    if x >= w || y >= h {
        return;
    }
    let out_row = h - 1 - y; // bottom-up to top-down
    let off = (out_row * w + x) * 4;
    if off + 4 <= pixels.len() {
        let color = palette.get(idx).copied().unwrap_or([0, 0, 0, 255]);
        pixels[off..off + 4].copy_from_slice(&color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_rgb24() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgb(&pixels, 8, 8).unwrap();
        assert_eq!(&encoded[..2], b"BM");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 8);
        assert_eq!(header.height, 8);
        assert_eq!(header.bits_per_pixel, 24);
        // Decoded is RGBA now, compare RGB channels
        for (i, chunk) in decoded.chunks_exact(4).enumerate() {
            assert_eq!(chunk[0], pixels[i * 3], "R mismatch at pixel {i}");
            assert_eq!(chunk[1], pixels[i * 3 + 1], "G mismatch at pixel {i}");
            assert_eq!(chunk[2], pixels[i * 3 + 2], "B mismatch at pixel {i}");
            assert_eq!(chunk[3], 255, "A should be 255");
        }
    }

    #[test]
    fn roundtrip_rgba32() {
        let pixels: Vec<u8> = (0..4 * 4 * 4).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgba(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bits_per_pixel, 32);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn row_padding_applied() {
        let pixels = vec![128u8; 3 * 2 * 3];
        let encoded = encode_rgb(&pixels, 3, 2).unwrap();
        let (_, decoded) = decode(&encoded).unwrap();
        for chunk in decoded.chunks_exact(4) {
            assert_eq!(chunk[0], 128);
            assert_eq!(chunk[1], 128);
            assert_eq!(chunk[2], 128);
        }
    }

    #[test]
    fn decode_palette_8bit() {
        // Manually construct an 8-bit palette BMP: 2x2, 4 colors
        let mut bmp = Vec::new();
        // File header
        bmp.extend_from_slice(b"BM");
        let palette_size = 256 * 4;
        let data_offset = 14 + 40 + palette_size;
        let row_size = 4; // 2 pixels/row, padded to 4
        let pixel_data_size = row_size * 2;
        let file_size = data_offset + pixel_data_size;
        bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 4]);
        bmp.extend_from_slice(&(data_offset as u32).to_le_bytes());
        // Info header
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&2i32.to_le_bytes()); // width
        bmp.extend_from_slice(&2i32.to_le_bytes()); // height (bottom-up)
        bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
        bmp.extend_from_slice(&8u16.to_le_bytes()); // bpp
        bmp.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
        bmp.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 16]); // res, colors
        // Palette: 256 entries (BGRA)
        for i in 0..256u32 {
            bmp.push((i * 1) as u8); // B
            bmp.push((i * 2) as u8); // G
            bmp.push((i * 3) as u8); // R
            bmp.push(0); // reserved
        }
        // Pixel data (bottom-up): row 1 (bottom) = [0, 1, pad, pad], row 0 (top) = [2, 3, pad, pad]
        bmp.extend_from_slice(&[0, 1, 0, 0]); // bottom row
        bmp.extend_from_slice(&[2, 3, 0, 0]); // top row

        let (header, decoded) = decode(&bmp).unwrap();
        assert_eq!(header.width, 2);
        assert_eq!(header.height, 2);
        assert_eq!(header.bits_per_pixel, 8);

        // Top-left pixel should be index 2 (top row in bottom-up = last row in file)
        // Index 2: R=6, G=4, B=2
        assert_eq!(decoded[0], 6); // R
        assert_eq!(decoded[1], 4); // G
        assert_eq!(decoded[2], 2); // B
    }

    #[test]
    fn decode_palette_4bit() {
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        let palette_size = 16 * 4;
        let data_offset = 14 + 40 + palette_size;
        let row_size = 4; // 4 pixels at 4bpp = 2 bytes, padded to 4
        let pixel_data_size = row_size * 1; // 1 row
        let file_size = data_offset + pixel_data_size;
        bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 4]);
        bmp.extend_from_slice(&(data_offset as u32).to_le_bytes());
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&4i32.to_le_bytes()); // width=4
        bmp.extend_from_slice(&1i32.to_le_bytes()); // height=1
        bmp.extend_from_slice(&1u16.to_le_bytes());
        bmp.extend_from_slice(&4u16.to_le_bytes()); // 4bpp
        bmp.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
        bmp.extend_from_slice(&(pixel_data_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 16]);
        // 16-color palette
        for i in 0..16u8 {
            bmp.extend_from_slice(&[i * 10, i * 15, i * 17, 0]); // BGRA
        }
        // Pixel data: indices [0xA, 0xB, 0xC, 0xD] → bytes [0xAB, 0xCD, 0x00, 0x00]
        bmp.extend_from_slice(&[0xAB, 0xCD, 0x00, 0x00]);

        let (header, decoded) = decode(&bmp).unwrap();
        assert_eq!(header.bits_per_pixel, 4);
        assert_eq!(decoded.len(), 4 * 4); // 4 pixels × RGBA
        // Pixel 0 = index 0xA: B=100, G=150, R=170 → output RGBA = [170, 150, 100, 255]
        assert_eq!(decoded[0], 170); // R (palette[0xA] R = 0xA*17)
    }

    #[test]
    fn decode_palette_1bit() {
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        let palette_size = 2 * 4;
        let data_offset = 14 + 40 + palette_size;
        let row_size = 4; // 8 pixels at 1bpp = 1 byte, padded to 4
        let file_size = data_offset + row_size;
        bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 4]);
        bmp.extend_from_slice(&(data_offset as u32).to_le_bytes());
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&8i32.to_le_bytes()); // width=8
        bmp.extend_from_slice(&1i32.to_le_bytes()); // height=1
        bmp.extend_from_slice(&1u16.to_le_bytes());
        bmp.extend_from_slice(&1u16.to_le_bytes()); // 1bpp
        bmp.extend_from_slice(&0u32.to_le_bytes());
        bmp.extend_from_slice(&(row_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 16]);
        // 2-color palette: black and white
        bmp.extend_from_slice(&[0, 0, 0, 0]); // index 0 = black
        bmp.extend_from_slice(&[255, 255, 255, 0]); // index 1 = white
        // Pixel data: 0b10101010 = alternating black/white
        bmp.extend_from_slice(&[0xAA, 0x00, 0x00, 0x00]);

        let (header, decoded) = decode(&bmp).unwrap();
        assert_eq!(header.bits_per_pixel, 1);
        assert_eq!(decoded.len(), 8 * 4); // 8 pixels × RGBA
        // Pixel 0 = index 1 (MSB) = white
        assert_eq!(&decoded[0..4], &[255, 255, 255, 255]);
        // Pixel 1 = index 0 = black
        assert_eq!(&decoded[4..8], &[0, 0, 0, 255]);
    }

    #[test]
    fn decode_rle8_solid() {
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        let palette_size = 256 * 4;
        let data_offset = 14 + 40 + palette_size;
        // RLE data: 4 pixels of index 5, then EOL, 4 pixels of index 5, then EOF
        let rle_data = [4, 5, 0, 0, 4, 5, 0, 1]; // run(4,5), EOL, run(4,5), EOF
        let file_size = data_offset + rle_data.len();
        bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 4]);
        bmp.extend_from_slice(&(data_offset as u32).to_le_bytes());
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&4i32.to_le_bytes()); // width=4
        bmp.extend_from_slice(&2i32.to_le_bytes()); // height=2
        bmp.extend_from_slice(&1u16.to_le_bytes());
        bmp.extend_from_slice(&8u16.to_le_bytes());
        bmp.extend_from_slice(&1u32.to_le_bytes()); // BI_RLE8
        bmp.extend_from_slice(&(rle_data.len() as u32).to_le_bytes());
        bmp.extend_from_slice(&[0u8; 16]);
        // Palette: index 5 = (0, 0, 200) blue-ish
        for i in 0..256u32 {
            if i == 5 {
                bmp.extend_from_slice(&[200, 0, 0, 0]); // BGR: B=200
            } else {
                bmp.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
        bmp.extend_from_slice(&rle_data);

        let (header, decoded) = decode(&bmp).unwrap();
        assert_eq!(header.compression, BI_RLE8);
        assert_eq!(decoded.len(), 4 * 2 * 4); // 4x2 × RGBA
        // All pixels should be index 5 = R=0, G=0, B=200
        for px in decoded.chunks_exact(4) {
            assert_eq!(px[0], 0); // R
            assert_eq!(px[1], 0); // G
            assert_eq!(px[2], 200); // B
        }
    }

    #[test]
    fn extract_channel_5bit() {
        // 5-5-5 format: 0x7C00 = R mask (bits 14:10)
        let val = 0b0_11111_00000_00000u32; // R=31, G=0, B=0
        assert_eq!(extract_channel(val, 0x7C00), 255); // 31 scaled to 255
        assert_eq!(extract_channel(val, 0x03E0), 0); // G=0
        assert_eq!(extract_channel(val, 0x001F), 0); // B=0
    }

    #[test]
    fn roundtrip_grayscale() {
        let pixels: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let encoded = encode_gray(&pixels, 16, 16).unwrap();
        assert_eq!(&encoded[..2], b"BM");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bits_per_pixel, 8);
        // Decoded is RGBA; gray N should map to (N, N, N, 255)
        for (i, chunk) in decoded.chunks_exact(4).enumerate() {
            let expected = pixels[i];
            assert_eq!(chunk[0], expected, "R mismatch at {i}");
            assert_eq!(chunk[1], expected, "G mismatch at {i}");
            assert_eq!(chunk[2], expected, "B mismatch at {i}");
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn invalid_magic_rejected() {
        assert!(
            decode(b"XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").is_err()
        );
    }
}
