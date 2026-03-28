#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::manual_div_ceil,
    clippy::empty_docs
)]
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
pub mod decode;
pub mod entropy;
mod error;
mod markers;
mod quantize;
pub mod trellis;
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

    // Select SOF marker based on mode
    let precision = if twelve_bit { 12u8 } else { 8u8 };
    let (h_max, v_max) = if is_gray {
        (1, 1)
    } else {
        color::subsampling_factors(config.subsampling)
    };

    let components: Vec<(u8, u8, u8, u8)> = if is_gray {
        vec![(1, 1, 1, 0)]
    } else {
        vec![
            (1, h_max as u8, v_max as u8, 0), // Y
            (2, 1, 1, 1),                     // Cb
            (3, 1, 1, 1),                     // Cr
        ]
    };

    // Write the appropriate SOF marker
    let write_sof = match (config.progressive, config.arithmetic_coding, twelve_bit) {
        (true, true, _) => markers::write_sof10, // Progressive + arithmetic
        (true, false, _) => markers::write_sof2, // Progressive + Huffman
        (false, true, _) => markers::write_sof9, // Sequential + arithmetic
        (false, false, true) => markers::write_sof1, // Extended sequential (12-bit)
        (false, false, false) => markers::write_sof0, // Baseline
    };
    write_sof(
        &mut out,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // Entropy coding tables
    if config.arithmetic_coding {
        // DAC: define arithmetic conditioning (default values)
        for comp_id in 0..components.len() {
            markers::write_dac(&mut out, 0, comp_id as u8, 0); // DC conditioning
            markers::write_dac(&mut out, 1, comp_id as u8, 1); // AC conditioning
        }
    } else {
        // DHT — standard Huffman tables
        if is_gray {
            markers::write_dht(&mut out, 0, 0, &entropy::DC_LUMA_CODE_LENGTHS);
            markers::write_dht(&mut out, 1, 0, &entropy::AC_LUMA_CODE_LENGTHS);
        } else {
            markers::write_standard_huffman_tables(&mut out);
        }
    }

    // DRI (restart interval)
    if let Some(interval) = config.restart_interval {
        markers::write_dri(&mut out, interval);
    }

    if config.progressive {
        // Progressive mode: multiple scans with spectral selection
        encode_progressive(
            &mut out,
            &ycbcr,
            is_gray,
            config.subsampling,
            &luma_qt,
            &chroma_qt,
            config.restart_interval,
        );
    } else {
        // Sequential mode: single SOS with full spectral range
        if is_gray {
            markers::write_sos(&mut out, &[(1, 0, 0)]);
        } else {
            markers::write_sos(&mut out, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
        }

        let mcu_data = encode_mcus(
            &ycbcr,
            is_gray,
            config.subsampling,
            &luma_qt,
            &chroma_qt,
            config.restart_interval,
        );
        let stuffed = markers::byte_stuff(&mcu_data);
        out.extend_from_slice(&stuffed);
    }

    markers::write_eoi(&mut out);
    Ok(out)
}

/// Encode in progressive mode with multiple scans.
///
/// Uses a standard progressive scan order matching libjpeg:
/// 1. DC coefficients for all components (Ss=0, Se=0)
/// 2. Y AC coefficients 1-5 (low frequency)
/// 3. Cb AC coefficients 1-63
/// 4. Cr AC coefficients 1-63
/// 5. Y AC coefficients 6-63 (high frequency)
fn encode_progressive(
    out: &mut Vec<u8>,
    ycbcr: &color::YcbcrImage,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    luma_qt: &[u16; 64],
    chroma_qt: &[u16; 64],
    restart_interval: Option<u16>,
) {
    // First, compute all DCT coefficients for all blocks
    let all_coeffs = compute_all_dct_coefficients(ycbcr, is_gray, subsampling, luma_qt, chroma_qt);

    // Progressive scan definitions: (component_ids, ss, se, ah, al)
    let scans: Vec<(Vec<u8>, u8, u8, u8, u8)> = if is_gray {
        vec![
            (vec![1], 0, 0, 0, 0),  // DC
            (vec![1], 1, 5, 0, 0),  // AC low
            (vec![1], 6, 63, 0, 0), // AC high
        ]
    } else {
        vec![
            (vec![1, 2, 3], 0, 0, 0, 0), // DC all components
            (vec![1], 1, 5, 0, 0),       // Y AC 1-5
            (vec![2], 1, 63, 0, 0),      // Cb AC 1-63
            (vec![3], 1, 63, 0, 0),      // Cr AC 1-63
            (vec![1], 6, 63, 0, 0),      // Y AC 6-63
        ]
    };

    for (comp_ids, ss, se, ah, al) in &scans {
        // Write SOS for this scan
        let sos_components: Vec<(u8, u8, u8)> = comp_ids
            .iter()
            .map(|&id| {
                let (dc_tab, ac_tab) = if id == 1 { (0, 0) } else { (1, 1) };
                (id, dc_tab, ac_tab)
            })
            .collect();
        markers::write_sos_progressive(out, &sos_components, *ss, *se, *ah, *al);

        // Encode coefficients for this scan's spectral band
        let scan_data = encode_progressive_scan(
            &all_coeffs,
            comp_ids,
            *ss,
            *se,
            is_gray,
            subsampling,
            restart_interval,
        );
        let stuffed = markers::byte_stuff(&scan_data);
        out.extend_from_slice(&stuffed);
    }
}

/// Pre-compute all DCT coefficients for all MCU blocks.
/// Returns a map: component_id → Vec of [i16; 64] blocks in raster order.
fn compute_all_dct_coefficients(
    ycbcr: &color::YcbcrImage,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    luma_qt: &[u16; 64],
    chroma_qt: &[u16; 64],
) -> std::collections::HashMap<u8, Vec<[i16; 64]>> {
    let mut result = std::collections::HashMap::new();

    let (mcu_w, mcu_h) = color::mcu_dimensions(subsampling);
    let mcu_cols = ycbcr.width.div_ceil(mcu_w) as usize;
    let mcu_rows = ycbcr.height.div_ceil(mcu_h) as usize;

    let (h_blocks, v_blocks) = if is_gray {
        (1usize, 1usize)
    } else {
        color::subsampling_factors(subsampling)
    };

    let mut y_blocks = Vec::new();
    let mut cb_blocks = Vec::new();
    let mut cr_blocks = Vec::new();

    for mcu_row in 0..mcu_rows {
        for mcu_col in 0..mcu_cols {
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
                    y_blocks.push(zigzag_reorder(&quantized));
                }
            }

            if !is_gray {
                for (plane, qt, blocks) in [
                    (&ycbcr.cb, chroma_qt, &mut cb_blocks),
                    (&ycbcr.cr, chroma_qt, &mut cr_blocks),
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
                    quantize::quantize(&dct_out, qt, &mut quantized);
                    blocks.push(zigzag_reorder(&quantized));
                }
            }
        }
    }

    result.insert(1, y_blocks);
    if !is_gray {
        result.insert(2, cb_blocks);
        result.insert(3, cr_blocks);
    }

    result
}

/// Encode a single progressive scan (subset of spectral coefficients).
fn encode_progressive_scan(
    all_coeffs: &std::collections::HashMap<u8, Vec<[i16; 64]>>,
    comp_ids: &[u8],
    ss: u8,
    se: u8,
    _is_gray: bool,
    _subsampling: ChromaSubsampling,
    _restart_interval: Option<u16>,
) -> Vec<u8> {
    use rasmcore_bitio::{BitOrder, BitWriter};

    let mut writer = BitWriter::new(BitOrder::MsbFirst);

    for &comp_id in comp_ids {
        let blocks = match all_coeffs.get(&comp_id) {
            Some(b) => b,
            None => continue,
        };

        let is_luma = comp_id == 1;
        let mut prev_dc: i16 = 0;

        for block in blocks {
            if ss == 0 {
                // DC coefficient
                let dc = block[0];
                let diff = dc - prev_dc;
                prev_dc = dc;

                let cat = entropy::magnitude_category(diff as i32);
                // Encode DC category using default Huffman table
                let dc_lengths = entropy::standard_dc_lengths(is_luma);
                encode_huffman_symbol(&mut writer, dc_lengths, cat);
                if cat > 0 {
                    encode_coefficient_bits(&mut writer, diff as i32, cat);
                }
            }

            if se > 0 {
                // AC coefficients in spectral range [max(ss,1)..=se]
                let start = if ss == 0 { 1 } else { ss as usize };
                let end = se as usize;
                let mut zero_run = 0u8;

                for coeff_ref in &block[start..=end] {
                    let coeff = *coeff_ref;
                    if coeff == 0 {
                        zero_run += 1;
                        if zero_run == 16 {
                            // ZRL (15, 0)
                            let ac_lengths = entropy::standard_ac_lengths(is_luma);
                            encode_huffman_symbol(&mut writer, ac_lengths, 0xF0);
                            zero_run = 0;
                        }
                    } else {
                        let cat = entropy::magnitude_category(coeff as i32);
                        let symbol = (zero_run << 4) | cat;
                        let ac_lengths = entropy::standard_ac_lengths(is_luma);
                        encode_huffman_symbol(&mut writer, ac_lengths, symbol);
                        encode_coefficient_bits(&mut writer, coeff as i32, cat);
                        zero_run = 0;
                    }
                }

                // EOB if we have remaining zeros
                if zero_run > 0 {
                    let ac_lengths = entropy::standard_ac_lengths(is_luma);
                    encode_huffman_symbol(&mut writer, ac_lengths, 0x00); // EOB
                }
            }
        }
    }

    writer.finish()
}

/// Encode a single Huffman symbol using the standard code length table.
fn encode_huffman_symbol(writer: &mut rasmcore_bitio::BitWriter, lengths: &[u8], symbol: u8) {
    // Build code from lengths (canonical Huffman)
    let mut code: u32 = 0;
    let mut idx = 0;
    for bit_len in 1..=16u8 {
        let count = lengths[bit_len as usize - 1] as usize;
        for _ in 0..count {
            if idx == symbol as usize {
                writer.write_bits(bit_len, code);
                return;
            }
            code += 1;
            idx += 1;
        }
        code <<= 1;
    }
    // Symbol not found — shouldn't happen with valid tables
}

/// Encode the raw bits for a coefficient value.
fn encode_coefficient_bits(writer: &mut rasmcore_bitio::BitWriter, value: i32, cat: u8) {
    if cat == 0 {
        return;
    }
    let bits = if value >= 0 {
        value as u32
    } else {
        (value + (1 << cat) - 1) as u32
    };
    writer.write_bits(cat, bits);
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

    // Single interleaved encoder — all components share one BitWriter.
    // This is the correct JPEG encoding: all MCU data in one bitstream.
    let mut enc = if is_gray {
        entropy::InterleavedMcuEncoder::new_gray()
    } else {
        entropy::InterleavedMcuEncoder::new_ycbcr()
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
                    // Flush current bitstream, insert RST marker
                    all_data.extend_from_slice(&enc.finish());
                    enc = if is_gray {
                        entropy::InterleavedMcuEncoder::new_gray()
                    } else {
                        entropy::InterleavedMcuEncoder::new_ycbcr()
                    };
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
                    enc.encode_block(0, &zz); // component 0 = Y
                }
            }

            // Encode Cb and Cr blocks (1 each per MCU)
            if !is_gray {
                for (comp_idx, plane) in [(1usize, &ycbcr.cb), (2, &ycbcr.cr)] {
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
                    enc.encode_block(comp_idx, &zz);
                }
            }

            mcu_count += 1;
        }
    }

    // Finalize single bitstream
    all_data.extend_from_slice(&enc.finish());
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
///
/// Supports baseline sequential JPEG (SOF0) with Huffman coding.
/// Returns RGB8 for color images, Gray8 for grayscale.
pub fn decode(data: &[u8]) -> Result<DecodedOutput, EncodeError> {
    let (pixels, width, height, is_gray) = decode::jpeg_decode(data)?;
    Ok(DecodedOutput {
        pixels,
        width,
        height,
        format: if is_gray {
            PixelFormat::Gray8
        } else {
            PixelFormat::Rgb8
        },
    })
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
    fn decode_rejects_empty_jpeg() {
        // SOI + EOI with no image data should error (not panic)
        let result = decode(&[0xFF, 0xD8, 0xFF, 0xD9]);
        assert!(result.is_err());
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

#[cfg(test)]
mod advanced_tests {
    use super::*;

    fn test_pixels(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h * 3)).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn progressive_produces_valid_jpeg() {
        let pixels = test_pixels(32, 32);
        let config = EncodeConfig {
            progressive: true,
            ..Default::default()
        };
        let result = encode(&pixels, 32, 32, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(&result[..2], &[0xFF, 0xD8]); // SOI
        // Should contain SOF2 marker (0xFFC2)
        let has_sof2 = result.windows(2).any(|w| w == [0xFF, 0xC2]);
        assert!(has_sof2, "progressive JPEG should contain SOF2 marker");
    }

    #[test]
    fn progressive_decodable_by_image_crate() {
        let pixels = test_pixels(16, 16);
        let config = EncodeConfig {
            progressive: true,
            ..Default::default()
        };
        let jpeg = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        let result = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg);
        assert!(
            result.is_ok(),
            "progressive JPEG should be decodable: {:?}",
            result.err()
        );
    }

    #[test]
    fn progressive_grayscale() {
        let pixels: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        let config = EncodeConfig {
            progressive: true,
            ..Default::default()
        };
        let result = encode(&pixels, 32, 32, PixelFormat::Gray8, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn twelve_bit_produces_sof1() {
        let pixels = test_pixels(16, 16);
        let config = EncodeConfig {
            sample_precision: SamplePrecision::Twelve,
            ..Default::default()
        };
        let result = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        // Should contain SOF1 marker (0xFFC1)
        let has_sof1 = result.windows(2).any(|w| w == [0xFF, 0xC1]);
        assert!(has_sof1, "12-bit JPEG should contain SOF1 marker");
    }

    #[test]
    fn arithmetic_produces_sof9() {
        let pixels = test_pixels(16, 16);
        let config = EncodeConfig {
            arithmetic_coding: true,
            ..Default::default()
        };
        let result = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        // Should contain SOF9 marker (0xFFC9)
        let has_sof9 = result.windows(2).any(|w| w == [0xFF, 0xC9]);
        assert!(has_sof9, "arithmetic JPEG should contain SOF9 marker");
    }

    #[test]
    fn progressive_arithmetic_produces_sof10() {
        let pixels = test_pixels(16, 16);
        let config = EncodeConfig {
            progressive: true,
            arithmetic_coding: true,
            ..Default::default()
        };
        let result = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        let has_sof10 = result.windows(2).any(|w| w == [0xFF, 0xCA]);
        assert!(
            has_sof10,
            "progressive+arithmetic should contain SOF10 marker"
        );
    }

    #[test]
    fn progressive_deterministic() {
        let pixels = test_pixels(16, 16);
        let config = EncodeConfig {
            progressive: true,
            ..Default::default()
        };
        let r1 = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        let r2 = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(r1, r2, "progressive encoding must be deterministic");
    }

    #[test]
    fn all_modes_deterministic() {
        let pixels = test_pixels(16, 16);
        for (prog, arith, prec) in [
            (false, false, SamplePrecision::Eight),
            (true, false, SamplePrecision::Eight),
            (false, true, SamplePrecision::Eight),
            (true, true, SamplePrecision::Eight),
            (false, false, SamplePrecision::Twelve),
        ] {
            let config = EncodeConfig {
                progressive: prog,
                arithmetic_coding: arith,
                sample_precision: prec,
                ..Default::default()
            };
            let r1 = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
            let r2 = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
            assert_eq!(
                r1, r2,
                "mode prog={prog} arith={arith} prec={prec:?} must be deterministic"
            );
        }
    }
}
