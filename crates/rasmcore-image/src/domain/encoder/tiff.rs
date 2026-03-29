//! TIFF encoder with configurable compression.
//!
//! Uses the `tiff` crate directly (not the `image` wrapper) to expose
//! compression options: LZW, Deflate, PackBits, and Uncompressed.

use std::io::Cursor;

use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// TIFF compression algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TiffCompression {
    /// No compression.
    None,
    /// LZW compression (default, good general-purpose lossless).
    #[default]
    Lzw,
    /// Deflate/zlib compression (better ratio, slower).
    Deflate,
    /// PackBits run-length encoding (fast, moderate compression).
    PackBits,
}

/// TIFF encode configuration.
#[derive(Debug, Clone)]
pub struct TiffEncodeConfig {
    /// Compression algorithm (default: LZW).
    pub compression: TiffCompression,
}

impl Default for TiffEncodeConfig {
    fn default() -> Self {
        Self {
            compression: TiffCompression::Lzw,
        }
    }
}

/// Map our compression enum to the tiff crate's Compression type.
fn map_compression(c: TiffCompression) -> tiff::encoder::Compression {
    match c {
        TiffCompression::None => tiff::encoder::Compression::Uncompressed,
        TiffCompression::Lzw => tiff::encoder::Compression::Lzw,
        TiffCompression::Deflate => {
            tiff::encoder::Compression::Deflate(tiff::encoder::DeflateLevel::Balanced)
        }
        TiffCompression::PackBits => tiff::encoder::Compression::Packbits,
    }
}

/// Encode pixel data to TIFF with the given configuration.
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    config: &TiffEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    let mut buf = Vec::new();
    let cursor = Cursor::new(&mut buf);

    let compression = map_compression(config.compression);
    let mut encoder = tiff::encoder::TiffEncoder::new(cursor)
        .map_err(|e| ImageError::ProcessingFailed(format!("TIFF encoder init: {e}")))?
        .with_compression(compression);

    let err_map = |e: tiff::TiffError| ImageError::ProcessingFailed(format!("TIFF encode: {e}"));

    match info.format {
        PixelFormat::Rgb8 => {
            let img = encoder
                .new_image::<tiff::encoder::colortype::RGB8>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(pixels).map_err(err_map)?;
        }
        PixelFormat::Rgba8 => {
            let img = encoder
                .new_image::<tiff::encoder::colortype::RGBA8>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(pixels).map_err(err_map)?;
        }
        PixelFormat::Gray8 => {
            let img = encoder
                .new_image::<tiff::encoder::colortype::Gray8>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(pixels).map_err(err_map)?;
        }
        PixelFormat::Gray16 => {
            let u16_data: Vec<u16> = pixels
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let img = encoder
                .new_image::<tiff::encoder::colortype::Gray16>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(&u16_data).map_err(err_map)?;
        }
        PixelFormat::Rgb16 => {
            let u16_data: Vec<u16> = pixels
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let img = encoder
                .new_image::<tiff::encoder::colortype::RGB16>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(&u16_data).map_err(err_map)?;
        }
        PixelFormat::Rgba16 => {
            let u16_data: Vec<u16> = pixels
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let img = encoder
                .new_image::<tiff::encoder::colortype::RGBA16>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(&u16_data).map_err(err_map)?;
        }
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "TIFF encoding from {other:?} not supported"
            )));
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_rgb8(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_rgba8(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_gray8(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn encode_rgb8_default_produces_valid_tiff() {
        let (pixels, info) = make_rgb8(16, 16);
        let result = encode(&pixels, &info, &TiffEncodeConfig::default()).unwrap();
        // TIFF magic: "II" (little-endian) + 42 or "MM" (big-endian) + 42
        assert!(
            (&result[..2] == b"II" || &result[..2] == b"MM"),
            "not a valid TIFF header"
        );
    }

    #[test]
    fn encode_rgba8_produces_valid_tiff() {
        let (pixels, info) = make_rgba8(16, 16);
        let result = encode(&pixels, &info, &TiffEncodeConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn encode_gray8_produces_valid_tiff() {
        let (pixels, info) = make_gray8(16, 16);
        let result = encode(&pixels, &info, &TiffEncodeConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn all_compressions_produce_valid_output() {
        let (pixels, info) = make_rgb8(32, 32);
        for comp in [
            TiffCompression::None,
            TiffCompression::Lzw,
            TiffCompression::Deflate,
            TiffCompression::PackBits,
        ] {
            let config = TiffEncodeConfig { compression: comp };
            let result = encode(&pixels, &info, &config);
            assert!(result.is_ok(), "compression {comp:?} failed");
        }
    }

    #[test]
    fn roundtrip_rgb8_pixel_exact() {
        let (pixels, info) = make_rgb8(16, 16);
        let encoded = encode(&pixels, &info, &TiffEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 16);
        assert_eq!(decoded.info.height, 16);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn roundtrip_gray8_pixel_exact() {
        let (pixels, info) = make_gray8(16, 16);
        let encoded = encode(&pixels, &info, &TiffEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.width, 16);
        assert_eq!(decoded.info.height, 16);
    }

    #[test]
    fn lzw_smaller_than_uncompressed() {
        let (pixels, info) = make_rgb8(64, 64);
        let uncompressed = encode(
            &pixels,
            &info,
            &TiffEncodeConfig {
                compression: TiffCompression::None,
            },
        )
        .unwrap();
        let lzw = encode(&pixels, &info, &TiffEncodeConfig::default()).unwrap();
        assert!(
            lzw.len() < uncompressed.len(),
            "LZW ({}) should be smaller than uncompressed ({})",
            lzw.len(),
            uncompressed.len()
        );
    }

    #[test]
    fn determinism() {
        let (pixels, info) = make_rgb8(32, 32);
        let config = TiffEncodeConfig::default();
        let r1 = encode(&pixels, &info, &config).unwrap();
        let r2 = encode(&pixels, &info, &config).unwrap();
        assert_eq!(r1, r2, "encoding must be deterministic");
    }

    #[test]
    fn default_compression_is_lzw() {
        assert_eq!(
            TiffEncodeConfig::default().compression,
            TiffCompression::Lzw
        );
    }
}
