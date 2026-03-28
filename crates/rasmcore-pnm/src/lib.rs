//! Pure Rust PNM (Portable Any Map) encoder/decoder.
//!
//! Full coverage matching ImageMagick and libvips:
//! - PBM (P1/P4): bitmap, ASCII and binary
//! - PGM (P2/P5): grayscale 8/16-bit, ASCII and binary
//! - PPM (P3/P6): RGB 8/16-bit, ASCII and binary
//! - PAM (P7): arbitrary depth/channels with TUPLTYPE header
//! - PFM: 32-bit floating point (RGB and grayscale)
//!
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
    /// P7 — Portable Arbitrary Map (PAM)
    Pam,
    /// PF / Pf — Portable Float Map
    Pfm,
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
    pub maxval: u32,
    /// Number of channels (PAM: 1-4, PFM: 1 or 3).
    pub depth: u32,
    /// PAM tuple type string (e.g., "GRAYSCALE", "RGB", "RGB_ALPHA").
    pub tupltype: Option<String>,
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

// ─── Encoders ──────────────────────────────────────────────────────────────

/// Encode bitmap to PBM (P4 binary).
/// `pixels`: one byte per pixel (0 = white, non-zero = black).
pub fn encode_pbm(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, 1)?;
    let header = format!("P4\n{width} {height}\n");
    let mut out = Vec::with_capacity(header.len() + width.div_ceil(8) as usize * height as usize);
    out.extend_from_slice(header.as_bytes());
    let mut bw = BitWriter::new(BitOrder::MsbFirst);
    for row in 0..height as usize {
        for col in 0..width as usize {
            bw.write_bit(pixels[row * width as usize + col] != 0);
        }
        bw.align_to_byte();
    }
    out.extend_from_slice(&bw.finish());
    Ok(out)
}

/// Encode bitmap to PBM ASCII (P1).
pub fn encode_pbm_ascii(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, 1)?;
    let mut out = format!("P1\n{width} {height}\n");
    for row in 0..height as usize {
        for col in 0..width as usize {
            if col > 0 {
                out.push(' ');
            }
            out.push(if pixels[row * width as usize + col] != 0 {
                '1'
            } else {
                '0'
            });
        }
        out.push('\n');
    }
    Ok(out.into_bytes())
}

/// Encode grayscale to PGM (P5 binary), 8-bit.
pub fn encode_pgm(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, 1)?;
    let n = (width * height) as usize;
    let header = format!("P5\n{width} {height}\n255\n");
    let mut out = Vec::with_capacity(header.len() + n);
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&pixels[..n]);
    Ok(out)
}

/// Encode grayscale to PGM ASCII (P2).
pub fn encode_pgm_ascii(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, 1)?;
    let mut out = format!("P2\n{width} {height}\n255\n");
    encode_ascii_values(&mut out, pixels, width, height, 1);
    Ok(out.into_bytes())
}

/// Encode grayscale to PGM (P5 binary), 16-bit big-endian.
pub fn encode_pgm_16(pixels: &[u16], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let n = (width * height) as usize;
    if pixels.len() < n {
        return Err(PnmError::BufferTooSmall);
    }
    let maxval = pixels.iter().copied().max().unwrap_or(255).max(256);
    let header = format!("P5\n{width} {height}\n{maxval}\n");
    let mut out = Vec::with_capacity(header.len() + n * 2);
    out.extend_from_slice(header.as_bytes());
    for &v in &pixels[..n] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    Ok(out)
}

/// Encode RGB to PPM (P6 binary), 8-bit.
pub fn encode_ppm(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, 3)?;
    let n = (width * height) as usize * 3;
    let header = format!("P6\n{width} {height}\n255\n");
    let mut out = Vec::with_capacity(header.len() + n);
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&pixels[..n]);
    Ok(out)
}

/// Encode RGB to PPM ASCII (P3).
pub fn encode_ppm_ascii(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, 3)?;
    let mut out = format!("P3\n{width} {height}\n255\n");
    encode_ascii_values(&mut out, pixels, width, height, 3);
    Ok(out.into_bytes())
}

/// Encode RGB to PPM (P6 binary), 16-bit big-endian.
pub fn encode_ppm_16(pixels: &[u16], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let n = (width * height) as usize * 3;
    if pixels.len() < n {
        return Err(PnmError::BufferTooSmall);
    }
    let maxval = pixels.iter().copied().max().unwrap_or(255).max(256);
    let header = format!("P6\n{width} {height}\n{maxval}\n");
    let mut out = Vec::with_capacity(header.len() + n * 2);
    out.extend_from_slice(header.as_bytes());
    for &v in &pixels[..n] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    Ok(out)
}

/// Encode PAM (P7). Supports arbitrary depth (1=gray, 2=gray+alpha, 3=RGB, 4=RGBA).
pub fn encode_pam(
    pixels: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    maxval: u32,
    tupltype: &str,
) -> Result<Vec<u8>, PnmError> {
    check_len(pixels, width, height, depth)?;
    let n = (width * height * depth) as usize;
    let bps = if maxval > 255 { 2 } else { 1 };
    let header = format!(
        "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {depth}\nMAXVAL {maxval}\nTUPLTYPE {tupltype}\nENDHDR\n"
    );
    let mut out = Vec::with_capacity(header.len() + n * bps);
    out.extend_from_slice(header.as_bytes());
    if bps == 1 {
        out.extend_from_slice(&pixels[..n]);
    } else {
        // 16-bit: interpret pixel pairs as u16 big-endian
        out.extend_from_slice(&pixels[..n * 2]);
    }
    Ok(out)
}

/// Encode PFM (Portable Float Map) — RGB, 32-bit float per channel.
/// `pixels`: width*height*3 f32 values. `scale`: negative = little-endian (standard).
pub fn encode_pfm_rgb(pixels: &[f32], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let n = (width * height) as usize * 3;
    if pixels.len() < n {
        return Err(PnmError::BufferTooSmall);
    }
    // PFM: negative scale = little-endian, bottom-up row order
    let header = format!("PF\n{width} {height}\n-1.0\n");
    let mut out = Vec::with_capacity(header.len() + n * 4);
    out.extend_from_slice(header.as_bytes());
    // Bottom-up row order
    for row in (0..height as usize).rev() {
        let start = row * width as usize * 3;
        for i in 0..width as usize * 3 {
            out.extend_from_slice(&pixels[start + i].to_le_bytes());
        }
    }
    Ok(out)
}

/// Encode PFM grayscale — 1 channel, 32-bit float.
pub fn encode_pfm_gray(pixels: &[f32], width: u32, height: u32) -> Result<Vec<u8>, PnmError> {
    let n = (width * height) as usize;
    if pixels.len() < n {
        return Err(PnmError::BufferTooSmall);
    }
    let header = format!("Pf\n{width} {height}\n-1.0\n");
    let mut out = Vec::with_capacity(header.len() + n * 4);
    out.extend_from_slice(header.as_bytes());
    for row in (0..height as usize).rev() {
        let start = row * width as usize;
        for i in 0..width as usize {
            out.extend_from_slice(&pixels[start + i].to_le_bytes());
        }
    }
    Ok(out)
}

// ─── Decoder ───────────────────────────────────────────────────────────────

/// Decode any PNM/PAM/PFM file to raw pixels.
///
/// Returns (header, pixels). For 8-bit formats, pixels is `Vec<u8>`.
/// For 16-bit (maxval > 255), each sample is 2 bytes big-endian in the output.
/// For PFM, samples are 4-byte little-endian f32.
pub fn decode(data: &[u8]) -> Result<(PnmHeader, Vec<u8>), PnmError> {
    if data.len() < 3 {
        return Err(PnmError::InvalidMagic);
    }

    // Check for PFM first (PF or Pf)
    if data.len() >= 2 && data[0] == b'P' && (data[1] == b'F' || data[1] == b'f') {
        return decode_pfm(data);
    }

    let (format, mode) = match &data[..2] {
        b"P1" => (PnmFormat::Pbm, PnmMode::Ascii),
        b"P2" => (PnmFormat::Pgm, PnmMode::Ascii),
        b"P3" => (PnmFormat::Ppm, PnmMode::Ascii),
        b"P4" => (PnmFormat::Pbm, PnmMode::Binary),
        b"P5" => (PnmFormat::Pgm, PnmMode::Binary),
        b"P6" => (PnmFormat::Ppm, PnmMode::Binary),
        b"P7" => return decode_pam(data),
        _ => return Err(PnmError::InvalidMagic),
    };

    let mut pos = 2;
    let width = parse_number(data, &mut pos)?;
    let height = parse_number(data, &mut pos)?;
    let maxval = if format == PnmFormat::Pbm {
        1
    } else {
        parse_number(data, &mut pos)?
    };

    // Skip single whitespace after header
    skip_single_whitespace(data, &mut pos);

    let depth = match format {
        PnmFormat::Pbm | PnmFormat::Pgm => 1,
        PnmFormat::Ppm => 3,
        _ => unreachable!(),
    };
    let bps = if maxval > 255 { 2 } else { 1 }; // bytes per sample

    let header = PnmHeader {
        format,
        mode,
        width,
        height,
        maxval,
        depth,
        tupltype: None,
    };

    let pixels = match (format, mode) {
        (PnmFormat::Pbm, PnmMode::Binary) => {
            let row_bytes = width.div_ceil(8) as usize;
            let total = row_bytes * height as usize;
            if pos + total > data.len() {
                return Err(PnmError::InvalidData("not enough pixel data".into()));
            }
            let mut pixels = Vec::with_capacity(width as usize * height as usize);
            let mut br = BitReader::new(&data[pos..pos + total], BitOrder::MsbFirst);
            for _row in 0..height as usize {
                for _col in 0..width as usize {
                    pixels.push(if br.read_bit().unwrap_or(false) {
                        255
                    } else {
                        0
                    });
                }
                br.align_to_byte();
            }
            pixels
        }
        (_, PnmMode::Binary) => {
            let len = width as usize * height as usize * depth as usize * bps;
            if pos + len > data.len() {
                return Err(PnmError::InvalidData("not enough pixel data".into()));
            }
            data[pos..pos + len].to_vec()
        }
        (_, PnmMode::Ascii) => {
            let count = width as usize * height as usize * depth as usize;
            let mut pixels = Vec::with_capacity(count * bps);
            for _ in 0..count {
                let val = parse_number(data, &mut pos)?;
                if bps == 1 {
                    pixels.push(val.min(maxval) as u8);
                } else {
                    pixels.extend_from_slice(&(val.min(maxval) as u16).to_be_bytes());
                }
            }
            pixels
        }
    };

    Ok((header, pixels))
}

fn decode_pam(data: &[u8]) -> Result<(PnmHeader, Vec<u8>), PnmError> {
    let mut pos = 2; // skip "P7"
    let mut width = 0u32;
    let mut height = 0u32;
    let mut depth = 0u32;
    let mut maxval = 255u32;
    let mut tupltype = None;

    // Parse PAM header lines until ENDHDR
    loop {
        skip_whitespace(data, &mut pos);
        if pos >= data.len() {
            return Err(PnmError::InvalidHeader("missing ENDHDR".into()));
        }
        let line_start = pos;
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        let line = std::str::from_utf8(&data[line_start..pos])
            .map_err(|_| PnmError::InvalidHeader("non-UTF8 header".into()))?
            .trim();
        if pos < data.len() {
            pos += 1; // skip \n
        }

        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if line == "ENDHDR" {
            break;
        }

        if let Some(val) = line.strip_prefix("WIDTH ") {
            width = val
                .trim()
                .parse()
                .map_err(|_| PnmError::InvalidHeader("bad WIDTH".into()))?;
        } else if let Some(val) = line.strip_prefix("HEIGHT ") {
            height = val
                .trim()
                .parse()
                .map_err(|_| PnmError::InvalidHeader("bad HEIGHT".into()))?;
        } else if let Some(val) = line.strip_prefix("DEPTH ") {
            depth = val
                .trim()
                .parse()
                .map_err(|_| PnmError::InvalidHeader("bad DEPTH".into()))?;
        } else if let Some(val) = line.strip_prefix("MAXVAL ") {
            maxval = val
                .trim()
                .parse()
                .map_err(|_| PnmError::InvalidHeader("bad MAXVAL".into()))?;
        } else if let Some(val) = line.strip_prefix("TUPLTYPE ") {
            tupltype = Some(val.trim().to_string());
        }
    }

    if width == 0 || height == 0 || depth == 0 {
        return Err(PnmError::InvalidHeader(
            "PAM missing required fields".into(),
        ));
    }

    let bps = if maxval > 255 { 2 } else { 1 };
    let len = width as usize * height as usize * depth as usize * bps;
    if pos + len > data.len() {
        return Err(PnmError::InvalidData("not enough PAM pixel data".into()));
    }

    let header = PnmHeader {
        format: PnmFormat::Pam,
        mode: PnmMode::Binary,
        width,
        height,
        maxval,
        depth,
        tupltype,
    };

    Ok((header, data[pos..pos + len].to_vec()))
}

fn decode_pfm(data: &[u8]) -> Result<(PnmHeader, Vec<u8>), PnmError> {
    let is_rgb = data[1] == b'F';
    let depth = if is_rgb { 3u32 } else { 1 };

    let mut pos = 2;
    let width = parse_number(data, &mut pos)?;
    let height = parse_number(data, &mut pos)?;

    // Parse scale (float) — skip whitespace, read until newline
    skip_whitespace(data, &mut pos);
    let scale_start = pos;
    while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
    }
    let scale_str = std::str::from_utf8(&data[scale_start..pos])
        .map_err(|_| PnmError::InvalidHeader("non-UTF8 scale".into()))?
        .trim();
    let scale: f32 = scale_str
        .parse()
        .map_err(|_| PnmError::InvalidHeader("bad PFM scale".into()))?;
    if pos < data.len() {
        pos += 1;
    }

    let little_endian = scale < 0.0;
    let n = width as usize * height as usize * depth as usize;
    let len = n * 4;
    if pos + len > data.len() {
        return Err(PnmError::InvalidData("not enough PFM pixel data".into()));
    }

    // PFM is stored bottom-up; convert to top-down
    let mut pixels = Vec::with_capacity(len);
    let row_len = width as usize * depth as usize * 4;
    for row in (0..height as usize).rev() {
        let row_start = pos + row * row_len;
        if little_endian {
            pixels.extend_from_slice(&data[row_start..row_start + row_len]);
        } else {
            // Big-endian: swap each 4-byte float
            for i in (0..row_len).step_by(4) {
                let off = row_start + i;
                pixels.push(data[off + 3]);
                pixels.push(data[off + 2]);
                pixels.push(data[off + 1]);
                pixels.push(data[off]);
            }
        }
    }

    let header = PnmHeader {
        format: PnmFormat::Pfm,
        mode: PnmMode::Binary,
        width,
        height,
        maxval: 0, // not applicable
        depth,
        tupltype: None,
    };

    Ok((header, pixels))
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn check_len(pixels: &[u8], width: u32, height: u32, channels: u32) -> Result<(), PnmError> {
    let expected = width as usize * height as usize * channels as usize;
    if pixels.len() < expected {
        Err(PnmError::BufferTooSmall)
    } else {
        Ok(())
    }
}

fn encode_ascii_values(out: &mut String, pixels: &[u8], width: u32, height: u32, channels: u32) {
    let w = width as usize * channels as usize;
    for row in 0..height as usize {
        for col in 0..w {
            if col > 0 {
                out.push(' ');
            }
            out.push_str(&pixels[row * w + col].to_string());
        }
        out.push('\n');
    }
}

fn parse_number(data: &[u8], pos: &mut usize) -> Result<u32, PnmError> {
    skip_whitespace_and_comments(data, pos);
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

fn skip_whitespace_and_comments(data: &[u8], pos: &mut usize) {
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
}

fn skip_whitespace(data: &[u8], pos: &mut usize) {
    while *pos < data.len() && matches!(data[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn skip_single_whitespace(data: &[u8], pos: &mut usize) {
    if *pos < data.len() && matches!(data[*pos], b' ' | b'\n' | b'\r' | b'\t') {
        *pos += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Binary roundtrips ──────────────────────────────────────────────

    #[test]
    fn roundtrip_ppm_binary() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_ppm(&pixels, 16, 16).unwrap();
        assert_eq!(&encoded[..2], b"P6");
        let (header, decoded) = decode(&encoded).unwrap();
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
        let pixels: Vec<u8> = (0..16).map(|i| if i % 2 == 0 { 255 } else { 0 }).collect();
        let encoded = encode_pbm(&pixels, 8, 2).unwrap();
        assert_eq!(&encoded[..2], b"P4");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pbm);
        assert_eq!(decoded, pixels);
    }

    // ─── ASCII roundtrips ───────────────────────────────────────────────

    #[test]
    fn roundtrip_ppm_ascii() {
        let pixels: Vec<u8> = (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_ppm_ascii(&pixels, 4, 4).unwrap();
        assert_eq!(&encoded[..2], b"P3");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Ppm);
        assert_eq!(header.mode, PnmMode::Ascii);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_pgm_ascii() {
        let pixels: Vec<u8> = (0..4 * 4).map(|i| (i * 16) as u8).collect();
        let encoded = encode_pgm_ascii(&pixels, 4, 4).unwrap();
        assert_eq!(&encoded[..2], b"P2");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pgm);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_pbm_ascii() {
        let pixels = vec![255, 0, 255, 0, 0, 255, 0, 255];
        let encoded = encode_pbm_ascii(&pixels, 4, 2).unwrap();
        assert_eq!(&encoded[..2], b"P1");
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pbm);
        assert_eq!(header.mode, PnmMode::Ascii);
        // ASCII PBM decodes as 0/1 values through parse_number → min(255)
        for (i, &v) in decoded.iter().enumerate() {
            let expected = if pixels[i] != 0 { 1 } else { 0 };
            assert_eq!(v, expected, "mismatch at {i}");
        }
    }

    // ─── 16-bit ─────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_pgm_16bit() {
        let pixels: Vec<u16> = (0..4 * 4).map(|i| i * 1000).collect();
        let encoded = encode_pgm_16(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert!(header.maxval > 255);
        // 16-bit data: each sample is 2 bytes big-endian
        for (i, &expected) in pixels.iter().enumerate() {
            let got = u16::from_be_bytes(decoded[i * 2..i * 2 + 2].try_into().unwrap());
            assert_eq!(got, expected, "sample {i} mismatch");
        }
    }

    #[test]
    fn roundtrip_ppm_16bit() {
        let pixels: Vec<u16> = (0..2 * 2 * 3).map(|i| (i * 500) as u16).collect();
        let encoded = encode_ppm_16(&pixels, 2, 2).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert!(header.maxval > 255);
        assert_eq!(decoded.len(), pixels.len() * 2);
    }

    // ─── PAM ────────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_pam_rgb() {
        let pixels: Vec<u8> = (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect();
        let encoded = encode_pam(&pixels, 4, 4, 3, 255, "RGB").unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pam);
        assert_eq!(header.depth, 3);
        assert_eq!(header.tupltype.as_deref(), Some("RGB"));
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_pam_rgba() {
        let pixels: Vec<u8> = (0..2 * 2 * 4).map(|i| (i % 256) as u8).collect();
        let encoded = encode_pam(&pixels, 2, 2, 4, 255, "RGB_ALPHA").unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.depth, 4);
        assert_eq!(header.tupltype.as_deref(), Some("RGB_ALPHA"));
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_pam_grayscale() {
        let pixels: Vec<u8> = (0..8 * 8).map(|i| (i % 256) as u8).collect();
        let encoded = encode_pam(&pixels, 8, 8, 1, 255, "GRAYSCALE").unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.depth, 1);
        assert_eq!(decoded, pixels);
    }

    // ─── PFM ────────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_pfm_rgb() {
        let pixels: Vec<f32> = (0..2 * 2 * 3).map(|i| i as f32 / 12.0).collect();
        let encoded = encode_pfm_rgb(&pixels, 2, 2).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pfm);
        assert_eq!(header.depth, 3);
        // Verify f32 roundtrip
        for i in 0..pixels.len() {
            let got = f32::from_le_bytes(decoded[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(
                (got - pixels[i]).abs() < 1e-6,
                "sample {i}: got {got}, expected {}",
                pixels[i]
            );
        }
    }

    #[test]
    fn roundtrip_pfm_gray() {
        let pixels: Vec<f32> = (0..4 * 4).map(|i| i as f32 * 0.1).collect();
        let encoded = encode_pfm_gray(&pixels, 4, 4).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.format, PnmFormat::Pfm);
        assert_eq!(header.depth, 1);
        for i in 0..pixels.len() {
            let got = f32::from_le_bytes(decoded[i * 4..i * 4 + 4].try_into().unwrap());
            assert!((got - pixels[i]).abs() < 1e-6);
        }
    }

    // ─── Decode compatibility ───────────────────────────────────────────

    #[test]
    fn decode_ppm_ascii_from_external() {
        let data = b"P3\n2 2\n255\n255 0 0 0 255 0 0 0 255 128 128 128\n";
        let (header, pixels) = decode(data).unwrap();
        assert_eq!(header.format, PnmFormat::Ppm);
        assert_eq!(pixels, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]);
    }

    #[test]
    fn decode_with_comments() {
        let data =
            b"P6\n# comment\n2 2\n# another\n255\n\x00\x00\x00\xFF\xFF\xFF\x80\x80\x80\x40\x40\x40";
        let (header, _) = decode(data).unwrap();
        assert_eq!(header.width, 2);
        assert_eq!(header.height, 2);
    }

    // ─── Error cases ────────────────────────────────────────────────────

    #[test]
    fn invalid_magic_rejected() {
        assert!(decode(b"XX\n1 1\n255\n\x00").is_err());
    }

    #[test]
    fn buffer_too_small_rejected() {
        assert!(encode_ppm(&[0; 2], 4, 4).is_err());
        assert!(encode_pgm(&[0; 2], 4, 4).is_err());
        assert!(encode_pbm(&[0; 2], 4, 4).is_err());
    }
}
