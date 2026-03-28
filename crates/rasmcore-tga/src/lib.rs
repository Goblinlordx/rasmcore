//! Pure Rust TGA (Truevision TARGA) encoder/decoder.
//!
//! Supports per the TGA 2.0 specification:
//! - Uncompressed and RLE true-color (16/24/32-bit)
//! - Uncompressed and RLE color-mapped (8-bit index into palette)
//! - Uncompressed and RLE grayscale (8-bit)
//! - Top-down and bottom-up scanline order
//!
//! All decoded output is normalized to RGBA.

use std::fmt;

const TGA_HEADER_SIZE: usize = 18;

/// TGA image type byte values (per spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TgaImageType {
    RawColorMapped = 1,
    RawTrueColor = 2,
    RawGrayscale = 3,
    RleColorMapped = 9,
    RleTrueColor = 10,
    RleGrayscale = 11,
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

/// Decoded TGA header.
#[derive(Debug, Clone)]
pub struct TgaHeader {
    pub width: u16,
    pub height: u16,
    pub bits_per_pixel: u8,
    pub image_type: TgaImageType,
    pub top_down: bool,
    pub has_alpha: bool,
}

/// Encode RGB pixels to uncompressed true-color TGA.
pub fn encode_rgb(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 3;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_truecolor(pixels, width, height, 24)
}

/// Encode RGBA pixels to uncompressed true-color TGA.
pub fn encode_rgba(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 4;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_truecolor(pixels, width, height, 32)
}

/// Encode grayscale pixels to uncompressed grayscale TGA.
pub fn encode_gray(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    let mut out = Vec::with_capacity(TGA_HEADER_SIZE + expected);
    write_header(&mut out, width, height, 8, TgaImageType::RawGrayscale, 0);
    out.extend_from_slice(&pixels[..expected]);
    Ok(out)
}

/// Encode gray+alpha pixels to uncompressed grayscale+alpha TGA.
pub fn encode_gray_alpha(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 2;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    let mut out = Vec::with_capacity(TGA_HEADER_SIZE + expected);
    write_header(&mut out, width, height, 16, TgaImageType::RawGrayscale, 8);
    out.extend_from_slice(&pixels[..expected]);
    Ok(out)
}

/// Encode RGB pixels with RLE compression.
pub fn encode_rgb_rle(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 3;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_rle(pixels, width, height, 24, TgaImageType::RleTrueColor)
}

/// Encode RGBA pixels with RLE compression.
pub fn encode_rgba_rle(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize * 4;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_rle(pixels, width, height, 32, TgaImageType::RleTrueColor)
}

/// Encode grayscale pixels with RLE compression.
pub fn encode_gray_rle(pixels: &[u8], width: u16, height: u16) -> Result<Vec<u8>, TgaError> {
    let expected = width as usize * height as usize;
    if pixels.len() < expected {
        return Err(TgaError::BufferTooSmall);
    }
    encode_rle(pixels, width, height, 8, TgaImageType::RleGrayscale)
}

fn encode_truecolor(pixels: &[u8], width: u16, height: u16, bpp: u8) -> Result<Vec<u8>, TgaError> {
    let channels = (bpp / 8) as usize;
    let pixel_count = width as usize * height as usize;
    let alpha_bits = if bpp == 32 { 8 } else { 0 };
    let mut out = Vec::with_capacity(TGA_HEADER_SIZE + pixel_count * channels);
    write_header(
        &mut out,
        width,
        height,
        bpp,
        TgaImageType::RawTrueColor,
        alpha_bits,
    );

    // Top-down, RGB(A) → BGR(A)
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

fn encode_rle(
    pixels: &[u8],
    width: u16,
    height: u16,
    bpp: u8,
    image_type: TgaImageType,
) -> Result<Vec<u8>, TgaError> {
    let channels = (bpp / 8) as usize;
    let pixel_count = width as usize * height as usize;
    let alpha_bits = if bpp == 32 { 8 } else { 0 };
    let mut out = Vec::with_capacity(TGA_HEADER_SIZE + pixel_count * channels);
    write_header(&mut out, width, height, bpp, image_type, alpha_bits);

    // Convert to TGA pixel order (BGR/grayscale) then RLE encode
    let tga_pixels: Vec<u8> = if channels >= 3 {
        let mut buf = Vec::with_capacity(pixel_count * channels);
        for i in 0..pixel_count {
            let off = i * channels;
            buf.push(pixels[off + 2]); // B
            buf.push(pixels[off + 1]); // G
            buf.push(pixels[off]); // R
            if channels == 4 {
                buf.push(pixels[off + 3]); // A
            }
        }
        buf
    } else {
        pixels[..pixel_count * channels].to_vec()
    };

    // RLE encode: scan for runs of identical pixels and raw sequences
    let mut i = 0;
    while i < pixel_count {
        let px = &tga_pixels[i * channels..(i + 1) * channels];

        // Count consecutive identical pixels (run)
        let mut run_len = 1;
        while i + run_len < pixel_count
            && run_len < 128
            && tga_pixels[(i + run_len) * channels..(i + run_len + 1) * channels] == *px
        {
            run_len += 1;
        }

        if run_len > 1 {
            // RLE packet: 1xxxxxxx + pixel
            out.push(0x80 | (run_len as u8 - 1));
            out.extend_from_slice(px);
            i += run_len;
        } else {
            // Raw packet: count non-repeating pixels
            let mut raw_len = 1;
            while i + raw_len < pixel_count && raw_len < 128 {
                let next = &tga_pixels[(i + raw_len) * channels..(i + raw_len + 1) * channels];
                // Stop if next pixel starts a run of 2+
                if i + raw_len + 1 < pixel_count
                    && tga_pixels[(i + raw_len + 1) * channels..(i + raw_len + 2) * channels]
                        == *next
                {
                    break;
                }
                raw_len += 1;
            }
            // Raw packet: 0xxxxxxx + pixels
            out.push(raw_len as u8 - 1);
            out.extend_from_slice(&tga_pixels[i * channels..(i + raw_len) * channels]);
            i += raw_len;
        }
    }

    Ok(out)
}

fn write_header(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    bpp: u8,
    image_type: TgaImageType,
    alpha_bits: u8,
) {
    out.push(0); // id_length
    out.push(0); // color_map_type
    out.push(image_type as u8);
    out.extend_from_slice(&[0u8; 5]); // color map spec
    out.extend_from_slice(&0u16.to_le_bytes()); // x_origin
    out.extend_from_slice(&0u16.to_le_bytes()); // y_origin
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.push(bpp);
    out.push(0x20 | (alpha_bits & 0x0F)); // top-down + alpha bits
}

/// Decode a TGA file to RGBA pixels (top-down).
///
/// All formats (true-color, color-mapped, grayscale) are normalized to RGBA.
pub fn decode(data: &[u8]) -> Result<(TgaHeader, Vec<u8>), TgaError> {
    if data.len() < TGA_HEADER_SIZE {
        return Err(TgaError::InvalidHeader("file too small".into()));
    }

    let id_length = data[0] as usize;
    let color_map_type = data[1];
    let image_type_byte = data[2];

    // Color map spec
    let _cm_first = u16::from_le_bytes(data[3..5].try_into().unwrap()) as usize;
    let cm_length = u16::from_le_bytes(data[5..7].try_into().unwrap()) as usize;
    let cm_entry_bpp = data[7];

    let width = u16::from_le_bytes(data[12..14].try_into().unwrap());
    let height = u16::from_le_bytes(data[14..16].try_into().unwrap());
    let bpp = data[16];
    let descriptor = data[17];
    let top_down = (descriptor & 0x20) != 0;
    let alpha_bits = descriptor & 0x0F;
    let has_alpha = alpha_bits > 0 || bpp == 32;

    let image_type = match image_type_byte {
        1 => TgaImageType::RawColorMapped,
        2 => TgaImageType::RawTrueColor,
        3 => TgaImageType::RawGrayscale,
        9 => TgaImageType::RleColorMapped,
        10 => TgaImageType::RleTrueColor,
        11 => TgaImageType::RleGrayscale,
        _ => {
            return Err(TgaError::UnsupportedFormat(format!(
                "image type {image_type_byte}"
            )));
        }
    };

    let data_start = TGA_HEADER_SIZE + id_length;

    // Read color map if present
    let color_map = if color_map_type == 1 && cm_length > 0 {
        let cm_entry_bytes = (cm_entry_bpp as usize).div_ceil(8);
        let cm_start = data_start;
        let cm_size = cm_length * cm_entry_bytes;
        if cm_start + cm_size > data.len() {
            return Err(TgaError::InvalidData("color map truncated".into()));
        }
        let mut map = Vec::with_capacity(cm_length);
        for i in 0..cm_length {
            let off = cm_start + i * cm_entry_bytes;
            let rgba = match cm_entry_bpp {
                24 => [data[off + 2], data[off + 1], data[off], 255],
                32 => [data[off + 2], data[off + 1], data[off], data[off + 3]],
                16 => {
                    let val = u16::from_le_bytes(data[off..off + 2].try_into().unwrap());
                    tga16_to_rgba(val)
                }
                _ => [0, 0, 0, 255],
            };
            map.push(rgba);
        }
        Some((map, cm_start + cm_size)) // return map + offset past it
    } else {
        None
    };

    let pixel_data_start = color_map
        .as_ref()
        .map(|(_, end)| *end)
        .unwrap_or(data_start);

    let pixel_count = width as usize * height as usize;
    let is_rle = matches!(
        image_type,
        TgaImageType::RleColorMapped | TgaImageType::RleTrueColor | TgaImageType::RleGrayscale
    );

    let pixel_bpp = match image_type {
        TgaImageType::RawColorMapped | TgaImageType::RleColorMapped => 8, // index size
        _ => bpp,
    };
    let pixel_bytes = (pixel_bpp as usize).div_ceil(8);

    let raw_pixels = if is_rle {
        decode_rle(&data[pixel_data_start..], pixel_count, pixel_bytes)?
    } else {
        let needed = pixel_count * pixel_bytes;
        if pixel_data_start + needed > data.len() {
            return Err(TgaError::InvalidData("not enough pixel data".into()));
        }
        data[pixel_data_start..pixel_data_start + needed].to_vec()
    };

    // Convert to RGBA, handling scanline order
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for row in 0..height as usize {
        let src_row = if top_down {
            row
        } else {
            height as usize - 1 - row
        };
        for col in 0..width as usize {
            let idx = src_row * width as usize + col;
            let off = idx * pixel_bytes;

            let rgba = match image_type {
                TgaImageType::RawColorMapped | TgaImageType::RleColorMapped => {
                    let map = color_map
                        .as_ref()
                        .ok_or_else(|| TgaError::InvalidData("missing color map".into()))?;
                    let ci = raw_pixels[off] as usize;
                    map.0.get(ci).copied().unwrap_or([0, 0, 0, 255])
                }
                TgaImageType::RawGrayscale | TgaImageType::RleGrayscale => {
                    let g = raw_pixels[off];
                    if pixel_bytes == 2 {
                        [g, g, g, raw_pixels[off + 1]] // gray + alpha
                    } else {
                        [g, g, g, 255]
                    }
                }
                TgaImageType::RawTrueColor | TgaImageType::RleTrueColor => match bpp {
                    16 => {
                        let val = u16::from_le_bytes(raw_pixels[off..off + 2].try_into().unwrap());
                        tga16_to_rgba(val)
                    }
                    24 => [
                        raw_pixels[off + 2],
                        raw_pixels[off + 1],
                        raw_pixels[off],
                        255,
                    ],
                    32 => [
                        raw_pixels[off + 2],
                        raw_pixels[off + 1],
                        raw_pixels[off],
                        raw_pixels[off + 3],
                    ],
                    _ => [0, 0, 0, 255],
                },
            };
            pixels.extend_from_slice(&rgba);
        }
    }

    Ok((
        TgaHeader {
            width,
            height,
            bits_per_pixel: bpp,
            image_type,
            top_down,
            has_alpha,
        },
        pixels,
    ))
}

/// Convert 16-bit TGA pixel (A1R5G5B5) to RGBA.
fn tga16_to_rgba(val: u16) -> [u8; 4] {
    let b = ((val & 0x001F) as u8) << 3;
    let g = (((val >> 5) & 0x001F) as u8) << 3;
    let r = (((val >> 10) & 0x001F) as u8) << 3;
    let a = 255u8; // bit 15 is alpha in spec, but typically ignored by decoders
    [r | (r >> 5), g | (g >> 5), b | (b >> 5), a]
}

/// Decode RLE-compressed TGA pixel data.
fn decode_rle(
    data: &[u8],
    pixel_count: usize,
    bytes_per_pixel: usize,
) -> Result<Vec<u8>, TgaError> {
    let mut out = Vec::with_capacity(pixel_count * bytes_per_pixel);
    let mut pos = 0;

    while out.len() < pixel_count * bytes_per_pixel {
        if pos >= data.len() {
            return Err(TgaError::InvalidData("RLE data truncated".into()));
        }
        let header = data[pos];
        pos += 1;
        let count = (header & 0x7F) as usize + 1;

        if header & 0x80 != 0 {
            if pos + bytes_per_pixel > data.len() {
                return Err(TgaError::InvalidData("RLE pixel truncated".into()));
            }
            let pixel = &data[pos..pos + bytes_per_pixel];
            pos += bytes_per_pixel;
            for _ in 0..count {
                out.extend_from_slice(pixel);
            }
        } else {
            let needed = count * bytes_per_pixel;
            if pos + needed > data.len() {
                return Err(TgaError::InvalidData("raw packet truncated".into()));
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
    fn roundtrip_rgb24() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgb(&pixels, 8, 8).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 8);
        assert_eq!(header.height, 8);
        assert_eq!(header.bits_per_pixel, 24);
        // Decoded is RGBA now
        for (i, chunk) in decoded.chunks_exact(4).enumerate() {
            assert_eq!(chunk[0], pixels[i * 3]);
            assert_eq!(chunk[1], pixels[i * 3 + 1]);
            assert_eq!(chunk[2], pixels[i * 3 + 2]);
            assert_eq!(chunk[3], 255);
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
    fn roundtrip_grayscale() {
        let pixels: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let encoded = encode_gray(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.image_type, TgaImageType::RawGrayscale);
        assert_eq!(header.bits_per_pixel, 8);
        // Decoded is RGBA: gray → (g, g, g, 255)
        for (i, chunk) in decoded.chunks_exact(4).enumerate() {
            assert_eq!(chunk[0], pixels[i]);
            assert_eq!(chunk[1], pixels[i]);
            assert_eq!(chunk[2], pixels[i]);
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn decode_colormapped() {
        // Manually construct a color-mapped TGA: 2x2, 4 palette entries
        let mut tga = Vec::new();
        tga.push(0); // id_length
        tga.push(1); // color_map_type = present
        tga.push(1); // image_type = raw color-mapped
        // Color map spec: first=0, length=4, entry_bpp=24
        tga.extend_from_slice(&0u16.to_le_bytes());
        tga.extend_from_slice(&4u16.to_le_bytes());
        tga.push(24);
        // Image spec
        tga.extend_from_slice(&0u16.to_le_bytes()); // x_origin
        tga.extend_from_slice(&0u16.to_le_bytes()); // y_origin
        tga.extend_from_slice(&2u16.to_le_bytes()); // width
        tga.extend_from_slice(&2u16.to_le_bytes()); // height
        tga.push(8); // bpp (index size)
        tga.push(0x20); // top-down
        // Color map: 4 entries (BGR)
        tga.extend_from_slice(&[255, 0, 0]); // idx 0 = blue
        tga.extend_from_slice(&[0, 255, 0]); // idx 1 = green
        tga.extend_from_slice(&[0, 0, 255]); // idx 2 = red
        tga.extend_from_slice(&[255, 255, 255]); // idx 3 = white
        // Pixel data: indices [0, 1, 2, 3]
        tga.extend_from_slice(&[0, 1, 2, 3]);

        let (header, decoded) = decode(&tga).unwrap();
        assert_eq!(header.image_type, TgaImageType::RawColorMapped);
        // Pixel 0 = blue: RGBA = [0, 0, 255, 255]
        assert_eq!(&decoded[0..4], &[0, 0, 255, 255]);
        // Pixel 1 = green: RGBA = [0, 255, 0, 255]
        assert_eq!(&decoded[4..8], &[0, 255, 0, 255]);
        // Pixel 2 = red: RGBA = [255, 0, 0, 255]
        assert_eq!(&decoded[8..12], &[255, 0, 0, 255]);
        // Pixel 3 = white
        assert_eq!(&decoded[12..16], &[255, 255, 255, 255]);
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

    #[test]
    fn unsupported_type_rejected() {
        let mut tga = vec![0u8; TGA_HEADER_SIZE];
        tga[2] = 99; // invalid image type
        assert!(decode(&tga).is_err());
    }

    #[test]
    fn roundtrip_rgb_rle() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgb_rle(&pixels, 8, 8).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.image_type, TgaImageType::RleTrueColor);
        for (i, chunk) in decoded.chunks_exact(4).enumerate() {
            assert_eq!(chunk[0], pixels[i * 3], "R mismatch at {i}");
            assert_eq!(chunk[1], pixels[i * 3 + 1], "G mismatch at {i}");
            assert_eq!(chunk[2], pixels[i * 3 + 2], "B mismatch at {i}");
        }
    }

    #[test]
    fn roundtrip_rgba_rle() {
        let pixels: Vec<u8> = (0..4 * 4 * 4).map(|i| (i % 256) as u8).collect();
        let encoded = encode_rgba_rle(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.image_type, TgaImageType::RleTrueColor);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn rle_compresses_solid_color() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let raw = encode_rgb(&pixels, 64, 64).unwrap();
        let rle = encode_rgb_rle(&pixels, 64, 64).unwrap();
        assert!(
            rle.len() < raw.len() / 2,
            "RLE should compress solid: {} vs {}",
            rle.len(),
            raw.len()
        );
        // Verify roundtrip
        let (_, decoded) = decode(&rle).unwrap();
        for chunk in decoded.chunks_exact(4) {
            assert_eq!(chunk[0], 128);
            assert_eq!(chunk[1], 128);
            assert_eq!(chunk[2], 128);
        }
    }

    #[test]
    fn roundtrip_gray_rle() {
        let pixels: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let encoded = encode_gray_rle(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.image_type, TgaImageType::RleGrayscale);
        for (i, chunk) in decoded.chunks_exact(4).enumerate() {
            assert_eq!(chunk[0], pixels[i]);
            assert_eq!(chunk[1], pixels[i]);
            assert_eq!(chunk[2], pixels[i]);
        }
    }

    #[test]
    fn roundtrip_gray_alpha() {
        // 2x2 gray+alpha
        let pixels = vec![200, 255, 100, 128, 50, 64, 0, 0];
        let encoded = encode_gray_alpha(&pixels, 2, 2).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.bits_per_pixel, 16);
        // Pixel 0: gray=200, alpha=255
        assert_eq!(&decoded[0..4], &[200, 200, 200, 255]);
        // Pixel 1: gray=100, alpha=128
        assert_eq!(&decoded[4..8], &[100, 100, 100, 128]);
    }
}
