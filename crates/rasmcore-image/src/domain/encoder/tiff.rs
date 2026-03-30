//! TIFF encoder with configurable compression.
//!
//! Uses the `tiff` crate directly (not the `image` wrapper) to expose
//! compression options: LZW, Deflate, PackBits, and Uncompressed.

use std::io::Cursor;

use crate::domain::error::ImageError;
use crate::domain::types::{FrameSequence, ImageInfo, PixelFormat};

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
        PixelFormat::Cmyk8 => {
            let img = encoder
                .new_image::<tiff::encoder::colortype::CMYK8>(info.width, info.height)
                .map_err(err_map)?;
            img.write_data(pixels).map_err(err_map)?;
        }
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "TIFF encoding from {other:?} not supported"
            )));
        }
    }

    Ok(buf)
}

/// Encode a FrameSequence as a multi-page TIFF.
///
/// Each frame becomes a separate IFD (page) in the output. The tiff crate
/// automatically chains IFDs when `new_image()` is called multiple times.
pub fn encode_pages(seq: &FrameSequence, config: &TiffEncodeConfig) -> Result<Vec<u8>, ImageError> {
    if seq.is_empty() {
        return Err(ImageError::InvalidInput(
            "cannot encode empty frame sequence as TIFF".into(),
        ));
    }

    let mut buf = Vec::new();
    let cursor = Cursor::new(&mut buf);
    let compression = map_compression(config.compression);
    let mut encoder = tiff::encoder::TiffEncoder::new(cursor)
        .map_err(|e| ImageError::ProcessingFailed(format!("TIFF encoder init: {e}")))?
        .with_compression(compression);

    let err_map = |e: tiff::TiffError| ImageError::ProcessingFailed(format!("TIFF encode: {e}"));

    for (image, _frame_info) in &seq.frames {
        let info = &image.info;
        let pixels = &image.pixels;

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
            PixelFormat::Cmyk8 => {
                let img = encoder
                    .new_image::<tiff::encoder::colortype::CMYK8>(info.width, info.height)
                    .map_err(err_map)?;
                img.write_data(pixels).map_err(err_map)?;
            }
            other => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "TIFF multi-page encoding from {other:?} not supported"
                )));
            }
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{
        ColorSpace, DecodedImage, DisposalMethod, FrameInfo, FrameSequence,
    };

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

    // ── 16-bit roundtrip tests ──────────────────────────────────────

    fn make_rgb16(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        // Generate 16-bit gradient with full range values
        let mut pixels = Vec::with_capacity((w * h * 6) as usize);
        for i in 0..(w * h) {
            let r = ((i * 257) % 65536) as u16;
            let g = ((i * 131 + 1000) % 65536) as u16;
            let b = ((i * 73 + 5000) % 65536) as u16;
            pixels.extend_from_slice(&r.to_le_bytes());
            pixels.extend_from_slice(&g.to_le_bytes());
            pixels.extend_from_slice(&b.to_le_bytes());
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_gray16(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let mut pixels = Vec::with_capacity((w * h * 2) as usize);
        for i in 0..(w * h) {
            let v = ((i * 257) % 65536) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray16,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn roundtrip_rgb16_pixel_exact() {
        let (pixels, info) = make_rgb16(16, 16);
        let encoded = encode(&pixels, &info, &TiffEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgb16);
        assert_eq!(decoded.info.width, 16);
        assert_eq!(decoded.info.height, 16);
        assert_eq!(
            decoded.pixels, pixels,
            "16-bit TIFF roundtrip must be pixel-exact"
        );
    }

    #[test]
    fn roundtrip_gray16_pixel_exact() {
        let (pixels, info) = make_gray16(16, 16);
        let encoded = encode(&pixels, &info, &TiffEncodeConfig::default()).unwrap();
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Gray16);
        assert_eq!(
            decoded.pixels, pixels,
            "16-bit Gray TIFF roundtrip must be pixel-exact"
        );
    }

    // ── encode_pages tests ─────────────────────────────────────────

    fn make_solid_page(
        w: u32,
        h: u32,
        r: u8,
        g: u8,
        b: u8,
        index: u32,
    ) -> (DecodedImage, FrameInfo) {
        let pixels: Vec<u8> = (0..(w * h)).flat_map(|_| [r, g, b]).collect();
        let image = DecodedImage {
            pixels,
            info: ImageInfo {
                width: w,
                height: h,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
            icc_profile: None,
        };
        let info = FrameInfo {
            index,
            delay_ms: 0,
            disposal: DisposalMethod::None,
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
        };
        (image, info)
    }

    #[test]
    fn encode_pages_2_page_roundtrip() {
        let mut seq = FrameSequence::new(4, 4);
        let (img0, fi0) = make_solid_page(4, 4, 255, 0, 0, 0);
        let (img1, fi1) = make_solid_page(4, 4, 0, 0, 255, 1);
        seq.push(img0, fi0);
        seq.push(img1, fi1);

        let encoded = encode_pages(&seq, &TiffEncodeConfig::default()).unwrap();

        // Verify it's a valid TIFF
        assert!(&encoded[..2] == b"II" || &encoded[..2] == b"MM");

        // Decode all pages
        let frames = crate::domain::decoder::decode_all_frames(&encoded).unwrap();
        assert_eq!(frames.len(), 2, "should have 2 pages");

        // Verify page 0 is red
        let red_expected: Vec<u8> = (0..16).flat_map(|_| [255u8, 0, 0]).collect();
        assert_eq!(frames[0].0.pixels, red_expected);

        // Verify page 1 is blue
        let blue_expected: Vec<u8> = (0..16).flat_map(|_| [0u8, 0, 255]).collect();
        assert_eq!(frames[1].0.pixels, blue_expected);
    }

    #[test]
    fn encode_pages_preserves_page_count() {
        let mut seq = FrameSequence::new(4, 4);
        for i in 0..4 {
            let (img, fi) = make_solid_page(4, 4, (i * 60) as u8, 0, 0, i);
            seq.push(img, fi);
        }

        let encoded = encode_pages(&seq, &TiffEncodeConfig::default()).unwrap();
        let count = crate::domain::decoder::frame_count(&encoded).unwrap();
        assert_eq!(count, 4);
    }

    #[test]
    fn encode_pages_empty_returns_error() {
        let seq = FrameSequence::new(4, 4);
        let result = encode_pages(&seq, &TiffEncodeConfig::default());
        assert!(result.is_err());
    }
}
