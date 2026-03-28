#![allow(dead_code)] // Modules used incrementally as encoder features are added
//! Pure Rust JPEG encoder/decoder — ITU-T T.81.
//!
//! Implements baseline sequential JPEG encoding with Huffman coding.
//! Uses shared infrastructure: rasmcore-bitio, rasmcore-deflate::huffman, rasmcore-color.
//!
//! # Pipeline
//!
//! ```text
//! pixels → color convert → subsample → DCT → quantize → Huffman → JFIF markers → output
//! ```

mod color;
mod dct;
pub mod entropy;
mod error;
mod markers;
mod quantize;
mod types;

pub use error::EncodeError;
pub use types::*;

/// Encode raw pixels to baseline JPEG.
///
/// Produces a valid JFIF file decodable by any JPEG decoder.
///
/// # Example
///
/// ```
/// use rasmcore_jpeg::{encode, EncodeConfig, PixelFormat};
///
/// let pixels = vec![128u8; 16 * 16 * 3];
/// let config = EncodeConfig::default();
/// let jpeg = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
/// assert_eq!(&jpeg[..2], &[0xFF, 0xD8]); // SOI marker
/// ```
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 || width > 65535 || height > 65535 {
        return Err(EncodeError::InvalidInput(format!(
            "invalid dimensions: {width}x{height}"
        )));
    }
    let bpp = match format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
    };
    let expected = width as usize * height as usize * bpp;
    if pixels.len() < expected {
        return Err(EncodeError::InvalidInput(format!(
            "pixel data too small: need {expected}, got {}",
            pixels.len()
        )));
    }

    let quality = config.quality.clamp(1, 100);
    let is_gray = format == PixelFormat::Gray8;
    let twelve_bit = config.sample_precision == SamplePrecision::Twelve;

    // 1. Color conversion + chroma subsampling
    let rgb_pixels = match format {
        PixelFormat::Rgb8 => &pixels[..expected],
        PixelFormat::Rgba8 => {
            // Strip alpha → RGB
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for chunk in pixels.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            return encode(&rgb, width, height, PixelFormat::Rgb8, config);
        }
        PixelFormat::Gray8 => &pixels[..expected],
    };

    let ycbcr = if is_gray {
        color::gray_to_y(rgb_pixels, width, height)
    } else {
        color::rgb_to_ycbcr(rgb_pixels, width, height, config.subsampling)
    };

    // 2. Build quantization tables
    let luma_qt = if let Some(ref custom) = config.custom_quant_tables {
        custom.luminance.values
    } else {
        quantize::luma_quant_table(quality, config.quant_preset, twelve_bit)
    };
    let chroma_qt = if let Some(ref custom) = config.custom_quant_tables {
        custom.chrominance.values
    } else {
        quantize::chroma_quant_table(quality, config.quant_preset, twelve_bit)
    };

    // Zigzag-ordered tables for DQT marker
    let luma_qt_zz = to_zigzag(&luma_qt);
    let chroma_qt_zz = if !is_gray {
        to_zigzag(&chroma_qt)
    } else {
        [0; 64]
    };

    // 3. Build output with JFIF markers
    let mut out = Vec::with_capacity(width as usize * height as usize);
    markers::write_soi(&mut out);
    markers::write_app0(&mut out);
    markers::write_dqt(&mut out, 0, &luma_qt_zz);
    if !is_gray {
        markers::write_dqt(&mut out, 1, &chroma_qt_zz);
    }

    // SOF0
    let (h_max, v_max) = if is_gray {
        (1, 1)
    } else {
        color::subsampling_factors(config.subsampling)
    };

    if is_gray {
        markers::write_sof0(&mut out, width as u16, height as u16, 8, &[(1, 1, 1, 0)]);
    } else {
        markers::write_sof0(
            &mut out,
            width as u16,
            height as u16,
            8,
            &[
                (1, h_max as u8, v_max as u8, 0), // Y
                (2, 1, 1, 1),                     // Cb
                (3, 1, 1, 1),                     // Cr
            ],
        );
    }

    // DHT — standard Huffman tables
    if is_gray {
        markers::write_dht(&mut out, 0, 0, &entropy::DC_LUMA_CODE_LENGTHS);
        markers::write_dht(&mut out, 1, 0, &entropy::AC_LUMA_CODE_LENGTHS);
    } else {
        markers::write_standard_huffman_tables(&mut out);
    }

    // DRI (restart interval)
    if let Some(interval) = config.restart_interval {
        markers::write_dri(&mut out, interval);
    }

    // SOS
    if is_gray {
        markers::write_sos(&mut out, &[(1, 0, 0)]);
    } else {
        markers::write_sos(&mut out, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
    }

    // 4. Encode MCUs
    let mcu_data = encode_mcus(
        &ycbcr,
        is_gray,
        config.subsampling,
        &luma_qt,
        &chroma_qt,
        config.restart_interval,
    );

    // Byte-stuff and append entropy data
    let stuffed = markers::byte_stuff(&mcu_data);
    out.extend_from_slice(&stuffed);

    markers::write_eoi(&mut out);
    Ok(out)
}

/// Encode all MCUs and return the raw entropy-coded data.
fn encode_mcus(
    ycbcr: &color::YcbcrImage,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    luma_qt: &[u16; 64],
    chroma_qt: &[u16; 64],
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let (mcu_w, mcu_h) = color::mcu_dimensions(subsampling);
    let mcu_cols = ycbcr.width.div_ceil(mcu_w) as usize;
    let mcu_rows = ycbcr.height.div_ceil(mcu_h) as usize;

    let mut y_enc = entropy::HuffmanEntropyEncoder::new_luma();
    let mut cb_enc = if !is_gray {
        Some(entropy::HuffmanEntropyEncoder::new_chroma())
    } else {
        None
    };
    let mut cr_enc = if !is_gray {
        Some(entropy::HuffmanEntropyEncoder::new_chroma())
    } else {
        None
    };

    let (h_blocks, v_blocks) = if is_gray {
        (1usize, 1usize)
    } else {
        color::subsampling_factors(subsampling)
    };

    let mut mcu_count = 0u32;
    let mut all_data = Vec::new();

    for mcu_row in 0..mcu_rows {
        for mcu_col in 0..mcu_cols {
            // Check restart
            #[allow(clippy::collapsible_if)]
            if let Some(interval) = restart_interval {
                if mcu_count > 0 && (mcu_count as u16).is_multiple_of(interval) {
                    // Flush current data, insert RST marker
                    all_data.extend_from_slice(&y_enc.finish());
                    y_enc = entropy::HuffmanEntropyEncoder::new_luma();
                    cb_enc = if !is_gray {
                        Some(entropy::HuffmanEntropyEncoder::new_chroma())
                    } else {
                        None
                    };
                    cr_enc = if !is_gray {
                        Some(entropy::HuffmanEntropyEncoder::new_chroma())
                    } else {
                        None
                    };
                    // Insert RST marker (not byte-stuffed)
                    let stuffed = markers::byte_stuff(&all_data);
                    all_data = stuffed;
                    markers::write_rst(&mut all_data, mcu_count / interval as u32 - 1);
                }
            }

            // Encode Y blocks (h_blocks × v_blocks per MCU)
            for vb in 0..v_blocks {
                for hb in 0..h_blocks {
                    let block = extract_block(
                        &ycbcr.y,
                        ycbcr.width as usize,
                        ycbcr.height as usize,
                        mcu_col * mcu_w as usize + hb * 8,
                        mcu_row * mcu_h as usize + vb * 8,
                    );
                    let mut dct_out = [0i32; 64];
                    dct::forward_dct(&block, &mut dct_out);
                    let mut quantized = [0i16; 64];
                    quantize::quantize(&dct_out, luma_qt, &mut quantized);
                    let zz = zigzag_reorder(&quantized);
                    y_enc.encode_block(&zz);
                }
            }

            // Encode Cb and Cr blocks (1 each per MCU)
            if !is_gray {
                for (plane, enc) in [
                    (&ycbcr.cb, cb_enc.as_mut().unwrap()),
                    (&ycbcr.cr, cr_enc.as_mut().unwrap()),
                ] {
                    let block = extract_block(
                        plane,
                        ycbcr.chroma_width as usize,
                        ycbcr.chroma_height as usize,
                        mcu_col * 8,
                        mcu_row * 8,
                    );
                    let mut dct_out = [0i32; 64];
                    dct::forward_dct(&block, &mut dct_out);
                    let mut quantized = [0i16; 64];
                    quantize::quantize(&dct_out, chroma_qt, &mut quantized);
                    let zz = zigzag_reorder(&quantized);
                    enc.encode_block(&zz);
                }
            }

            mcu_count += 1;
        }
    }

    // Collect all entropy data
    all_data.extend_from_slice(&y_enc.finish());
    if let Some(enc) = cb_enc {
        all_data.extend_from_slice(&enc.finish());
    }
    if let Some(enc) = cr_enc {
        all_data.extend_from_slice(&enc.finish());
    }

    all_data
}

/// Extract an 8x8 block from a plane, level-shifted (subtract 128 for JPEG).
fn extract_block(plane: &[u8], stride: usize, height: usize, x: usize, y: usize) -> [i16; 64] {
    let mut block = [0i16; 64];
    for row in 0..8 {
        for col in 0..8 {
            let py = (y + row).min(height.saturating_sub(1));
            let px = (x + col).min(stride.saturating_sub(1));
            block[row * 8 + col] = plane[py * stride + px] as i16 - 128;
        }
    }
    block
}

/// Reorder coefficients from natural 8x8 order to zigzag scan order.
fn zigzag_reorder(coeffs: &[i16; 64]) -> [i16; 64] {
    let mut zz = [0i16; 64];
    for (i, &zi) in quantize::ZIGZAG.iter().enumerate() {
        zz[i] = coeffs[zi];
    }
    zz
}

/// Convert quantization table from natural order to zigzag order for DQT.
fn to_zigzag(table: &[u16; 64]) -> [u16; 64] {
    let mut zz = [0u16; 64];
    for (i, &zi) in quantize::ZIGZAG.iter().enumerate() {
        zz[i] = table[zi];
    }
    zz
}

/// Decode JPEG data to raw pixels.
pub fn decode(_data: &[u8]) -> Result<DecodedOutput, EncodeError> {
    Err(EncodeError::NotYetImplemented)
}

/// Decoded JPEG output.
pub struct DecodedOutput {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_solid_gray_8x8() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let config = EncodeConfig::default();
        let jpeg = encode(&pixels, 8, 8, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]); // EOI
    }

    #[test]
    fn encode_grayscale() {
        let pixels = vec![200u8; 16 * 16];
        let config = EncodeConfig::default();
        let jpeg = encode(&pixels, 16, 16, PixelFormat::Gray8, &config).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encode_16x16_420() {
        let pixels = vec![100u8; 16 * 16 * 3];
        let config = EncodeConfig {
            subsampling: ChromaSubsampling::Quarter420,
            ..Default::default()
        };
        let jpeg = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        assert!(jpeg.len() > 100); // should have markers + data
    }

    #[test]
    fn encode_444() {
        let pixels = vec![150u8; 8 * 8 * 3];
        let config = EncodeConfig {
            subsampling: ChromaSubsampling::None444,
            ..Default::default()
        };
        let jpeg = encode(&pixels, 8, 8, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encode_deterministic() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
        let config = EncodeConfig::default();
        let a = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        let b = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(a, b, "encode should be deterministic");
    }

    #[test]
    fn encode_rgba_strips_alpha() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let config = EncodeConfig::default();
        let jpeg = encode(&pixels, 8, 8, PixelFormat::Rgba8, &config).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encode_invalid_dimensions_rejected() {
        assert!(encode(&[], 0, 10, PixelFormat::Rgb8, &EncodeConfig::default()).is_err());
    }

    #[test]
    fn decode_returns_not_yet_implemented() {
        assert!(matches!(
            decode(&[0xFF, 0xD8, 0xFF, 0xD9]),
            Err(EncodeError::NotYetImplemented)
        ));
    }

    #[test]
    fn default_config_values() {
        let config = EncodeConfig::default();
        assert_eq!(config.quality, 85);
        assert!(!config.progressive);
        assert_eq!(config.subsampling, ChromaSubsampling::Quarter420);
        assert!(!config.arithmetic_coding);
        assert!(config.restart_interval.is_none());
        assert!(!config.optimize_huffman);
        assert!(!config.trellis);
        assert_eq!(config.sample_precision, SamplePrecision::Eight);
        assert_eq!(config.quant_preset, crate::quantize::QuantPreset::Robidoux);
        assert!(config.custom_quant_tables.is_none());
    }

    #[test]
    fn encode_decodable_by_image_crate() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32u8 {
            for x in 0..32u8 {
                pixels.push(x * 8);
                pixels.push(y * 8);
                pixels.push(128);
            }
        }
        let config = EncodeConfig::default();
        let jpeg = encode(&pixels, 32, 32, PixelFormat::Rgb8, &config).unwrap();

        let result = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg);
        assert!(
            result.is_ok(),
            "JPEG should be decodable: {:?}",
            result.err()
        );
        let img = result.unwrap().to_rgb8();
        assert_eq!(img.dimensions(), (32, 32));
    }
}
