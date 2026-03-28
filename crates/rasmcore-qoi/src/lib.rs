//! Pure Rust QOI (Quite OK Image) encoder/decoder.
//!
//! Implements the QOI format per <https://qoiformat.org/qoi-specification.pdf>.
//! Zero external dependencies, WASM-ready.

#![allow(clippy::manual_range_contains)]
//!
//! # Format overview
//!
//! QOI is a lossless image format that uses:
//! - A running hash table of 64 recently seen pixels
//! - Run-length encoding for consecutive identical pixels
//! - Small difference encoding for pixels close to the previous one
//! - Full RGBA encoding as fallback

use std::fmt;

/// QOI magic bytes: "qoif"
const QOI_MAGIC: [u8; 4] = *b"qoif";
const QOI_HEADER_SIZE: usize = 14;
const QOI_END_MARKER: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];

// Chunk tag constants
const QOI_OP_RGB: u8 = 0xFE;
const QOI_OP_RGBA: u8 = 0xFF;
const QOI_OP_INDEX: u8 = 0x00; // 2-bit tag: 00
const QOI_OP_DIFF: u8 = 0x40; // 2-bit tag: 01
const QOI_OP_LUMA: u8 = 0x80; // 2-bit tag: 10
const QOI_OP_RUN: u8 = 0xC0; // 2-bit tag: 11
const QOI_MASK_2: u8 = 0xC0;

/// Color channels in a QOI image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Rgb = 3,
    Rgba = 4,
}

/// Color space of a QOI image (informational only, does not affect encoding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb = 0,
    Linear = 1,
}

/// QOI image header.
#[derive(Debug, Clone)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub channels: Channels,
    pub colorspace: ColorSpace,
}

/// QOI encoding/decoding error.
#[derive(Debug)]
pub enum QoiError {
    InvalidMagic,
    InvalidHeader,
    InvalidData(String),
    BufferTooSmall,
}

impl fmt::Display for QoiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid QOI magic bytes"),
            Self::InvalidHeader => write!(f, "invalid QOI header"),
            Self::InvalidData(msg) => write!(f, "invalid QOI data: {msg}"),
            Self::BufferTooSmall => write!(f, "pixel buffer too small"),
        }
    }
}

impl std::error::Error for QoiError {}

#[derive(Clone, Copy, PartialEq)]
struct Pixel {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Pixel {
    const ZERO: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };

    fn hash(&self) -> usize {
        (self.r as usize * 3 + self.g as usize * 5 + self.b as usize * 7 + self.a as usize * 11)
            % 64
    }
}

/// Encode raw pixels to QOI format.
///
/// `pixels` must be `width * height * channels` bytes.
/// Returns the complete QOI file as a byte vector.
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: Channels,
    colorspace: ColorSpace,
) -> Result<Vec<u8>, QoiError> {
    let ch = channels as usize;
    let pixel_count = width as usize * height as usize;
    if pixels.len() < pixel_count * ch {
        return Err(QoiError::BufferTooSmall);
    }

    let max_size = QOI_HEADER_SIZE + pixel_count * (ch + 1) + QOI_END_MARKER.len();
    let mut out = Vec::with_capacity(max_size);

    // Header
    out.extend_from_slice(&QOI_MAGIC);
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.push(ch as u8);
    out.push(colorspace as u8);

    let mut index = [Pixel::ZERO; 64];
    let mut prev = Pixel::ZERO;
    let mut run = 0u8;

    for i in 0..pixel_count {
        let off = i * ch;
        let px = Pixel {
            r: pixels[off],
            g: pixels[off + 1],
            b: pixels[off + 2],
            a: if ch == 4 { pixels[off + 3] } else { 255 },
        };

        if px == prev {
            run += 1;
            if run == 62 || i == pixel_count - 1 {
                out.push(QOI_OP_RUN | (run - 1));
                run = 0;
            }
            continue;
        }

        if run > 0 {
            out.push(QOI_OP_RUN | (run - 1));
            run = 0;
        }

        let hash = px.hash();
        if index[hash] == px {
            out.push(QOI_OP_INDEX | hash as u8);
        } else {
            index[hash] = px;

            if px.a == prev.a {
                let dr = px.r.wrapping_sub(prev.r) as i8;
                let dg = px.g.wrapping_sub(prev.g) as i8;
                let db = px.b.wrapping_sub(prev.b) as i8;

                let dr_dg = dr.wrapping_sub(dg);
                let db_dg = db.wrapping_sub(dg);

                if dr >= -2 && dr <= 1 && dg >= -2 && dg <= 1 && db >= -2 && db <= 1 {
                    out.push(
                        QOI_OP_DIFF
                            | ((dr + 2) as u8) << 4
                            | ((dg + 2) as u8) << 2
                            | (db + 2) as u8,
                    );
                } else if dr_dg >= -8
                    && dr_dg <= 7
                    && dg >= -32
                    && dg <= 31
                    && db_dg >= -8
                    && db_dg <= 7
                {
                    out.push(QOI_OP_LUMA | (dg + 32) as u8);
                    out.push(((dr_dg + 8) as u8) << 4 | (db_dg + 8) as u8);
                } else {
                    out.push(QOI_OP_RGB);
                    out.push(px.r);
                    out.push(px.g);
                    out.push(px.b);
                }
            } else {
                out.push(QOI_OP_RGBA);
                out.push(px.r);
                out.push(px.g);
                out.push(px.b);
                out.push(px.a);
            }
        }
        prev = px;
    }

    out.extend_from_slice(&QOI_END_MARKER);
    Ok(out)
}

/// Decode a QOI file to raw pixels.
///
/// Returns (header, pixels) where pixels is `width * height * channels` bytes.
pub fn decode(data: &[u8]) -> Result<(Header, Vec<u8>), QoiError> {
    if data.len() < QOI_HEADER_SIZE + QOI_END_MARKER.len() {
        return Err(QoiError::InvalidHeader);
    }
    if data[0..4] != QOI_MAGIC {
        return Err(QoiError::InvalidMagic);
    }

    let width = u32::from_be_bytes(data[4..8].try_into().unwrap());
    let height = u32::from_be_bytes(data[8..12].try_into().unwrap());
    let channels = match data[12] {
        3 => Channels::Rgb,
        4 => Channels::Rgba,
        _ => return Err(QoiError::InvalidHeader),
    };
    let colorspace = match data[13] {
        0 => ColorSpace::Srgb,
        1 => ColorSpace::Linear,
        _ => return Err(QoiError::InvalidHeader),
    };

    let ch = channels as usize;
    let pixel_count = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(pixel_count * ch);

    let mut index = [Pixel::ZERO; 64];
    let mut px = Pixel::ZERO;
    let mut pos = QOI_HEADER_SIZE;
    let mut run = 0u32;

    for _ in 0..pixel_count {
        if run > 0 {
            run -= 1;
        } else {
            if pos >= data.len().saturating_sub(QOI_END_MARKER.len()) {
                return Err(QoiError::InvalidData("unexpected end of data".into()));
            }

            let b1 = data[pos];
            pos += 1;

            if b1 == QOI_OP_RGB {
                px.r = data[pos];
                px.g = data[pos + 1];
                px.b = data[pos + 2];
                pos += 3;
            } else if b1 == QOI_OP_RGBA {
                px.r = data[pos];
                px.g = data[pos + 1];
                px.b = data[pos + 2];
                px.a = data[pos + 3];
                pos += 4;
            } else {
                match b1 & QOI_MASK_2 {
                    QOI_OP_INDEX => {
                        px = index[(b1 & 0x3F) as usize];
                    }
                    QOI_OP_DIFF => {
                        px.r = px.r.wrapping_add(((b1 >> 4) & 0x03).wrapping_sub(2));
                        px.g = px.g.wrapping_add(((b1 >> 2) & 0x03).wrapping_sub(2));
                        px.b = px.b.wrapping_add((b1 & 0x03).wrapping_sub(2));
                    }
                    QOI_OP_LUMA => {
                        let b2 = data[pos];
                        pos += 1;
                        let dg = (b1 & 0x3F).wrapping_sub(32);
                        px.r =
                            px.r.wrapping_add(dg.wrapping_add((b2 >> 4) & 0x0F).wrapping_sub(8));
                        px.g = px.g.wrapping_add(dg);
                        px.b =
                            px.b.wrapping_add(dg.wrapping_add(b2 & 0x0F).wrapping_sub(8));
                    }
                    QOI_OP_RUN => {
                        run = (b1 & 0x3F) as u32;
                        // This pixel is the first of the run; remaining `run` copies
                        // will be emitted on subsequent outer-loop iterations.
                    }
                    _ => unreachable!(),
                }
            }

            index[px.hash()] = px;
        }

        pixels.push(px.r);
        pixels.push(px.g);
        pixels.push(px.b);
        if ch == 4 {
            pixels.push(px.a);
        }
    }

    let header = Header {
        width,
        height,
        channels,
        colorspace,
    };
    Ok((header, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_rgb_solid() {
        let pixels = vec![128u8; 16 * 16 * 3];
        let encoded = encode(&pixels, 16, 16, Channels::Rgb, ColorSpace::Srgb).unwrap();
        assert_eq!(&encoded[..4], b"qoif");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 16);
        assert_eq!(header.height, 16);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_rgba_gradient() {
        let mut pixels = Vec::with_capacity(32 * 32 * 4);
        for y in 0..32u8 {
            for x in 0..32u8 {
                pixels.push(x * 8);
                pixels.push(y * 8);
                pixels.push(128);
                pixels.push(255);
            }
        }
        let encoded = encode(&pixels, 32, 32, Channels::Rgba, ColorSpace::Srgb).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.channels, Channels::Rgba);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_1x1() {
        let pixels = vec![255, 0, 0];
        let encoded = encode(&pixels, 1, 1, Channels::Rgb, ColorSpace::Srgb).unwrap();
        let (_, decoded) = decode(&encoded).unwrap();
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn run_length_encoding_compresses() {
        // Solid color → should compress heavily via RLE
        let pixels = vec![42u8; 256 * 256 * 3];
        let encoded = encode(&pixels, 256, 256, Channels::Rgb, ColorSpace::Srgb).unwrap();
        assert!(
            encoded.len() < pixels.len() / 10,
            "RLE should compress solid color heavily: {} vs {}",
            encoded.len(),
            pixels.len()
        );
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut data = vec![0u8; 30];
        data[..4].copy_from_slice(b"nope");
        assert!(decode(&data).is_err());
    }

    #[test]
    fn buffer_too_small_rejected() {
        assert!(encode(&[0; 2], 4, 4, Channels::Rgb, ColorSpace::Srgb).is_err());
    }
}
