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
mod decode_arith;
pub mod entropy;
mod error;
mod markers;
pub mod qm_coder;
mod quantize;
pub mod trellis;
mod types;

pub use error::EncodeError;
pub use quantize::QuantPreset;
pub use types::*;
// QuantPreset is re-exported for external consumers to set quant_preset on EncodeConfig.

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

    // Apply turbo overrides (turbo=true disables trellis, optimize_huffman, eob_optimize)
    let config = &config.effective();

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

    // DRI (restart interval)
    if let Some(interval) = config.restart_interval {
        markers::write_dri(&mut out, interval);
    }

    if config.arithmetic_coding && !config.progressive {
        // Arithmetic coding: DAC tables + single SOS
        for comp_id in 0..components.len() {
            markers::write_dac(&mut out, 0, comp_id as u8, 0);
            markers::write_dac(&mut out, 1, comp_id as u8, 1);
        }
        if is_gray {
            markers::write_sos(&mut out, &[(1, 0, 0)]);
        } else {
            markers::write_sos(&mut out, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
        }

        let mcu_data =
            encode_mcus_arithmetic(&ycbcr, is_gray, config.subsampling, &luma_qt, &chroma_qt);
        out.extend_from_slice(&mcu_data);
    } else if config.progressive {
        // Progressive mode: DHT + multiple scans
        if is_gray {
            markers::write_dht(&mut out, 0, 0, &entropy::DC_LUMA_CODE_LENGTHS);
            markers::write_dht(&mut out, 1, 0, &entropy::AC_LUMA_CODE_LENGTHS);
        } else {
            markers::write_standard_huffman_tables(&mut out);
        }
        encode_progressive(
            &mut out,
            &ycbcr,
            is_gray,
            config.subsampling,
            &luma_qt,
            &chroma_qt,
            config.restart_interval,
            config.trellis,
        );
    } else if config.optimize_huffman {
        // Multi-pass optimized Huffman:
        // Pass 1: trellis with standard Huffman tables → collect frequencies
        // Pass 2: build optimal tables → re-trellis with accurate rates → re-count
        // Pass 3: encode with final optimal tables
        let (luma_freq, chroma_freq, mut blocks, raw_dct) = collect_frequencies(
            &ycbcr,
            is_gray,
            config.subsampling,
            &luma_qt,
            &chroma_qt,
            config.trellis,
        );

        let (dc_luma_len, ac_luma_len) = luma_freq.build_optimal_tables();
        let (dc_chroma_len, ac_chroma_len) = if is_gray {
            (vec![0u8; 12], vec![0u8; 256])
        } else {
            chroma_freq.build_optimal_tables()
        };

        // Pass 2: re-trellis with accurate rate estimates from optimal tables
        if config.trellis {
            let ac_luma_arr: [u8; 256] = ac_luma_len.clone().try_into().unwrap_or([0u8; 256]);
            let ac_chroma_arr: [u8; 256] = ac_chroma_len.clone().try_into().unwrap_or([0u8; 256]);

            let mut luma_freq2 = entropy::FrequencyCounter::new();
            let mut cb_freq2 = entropy::FrequencyCounter::new();
            let mut cr_freq2 = entropy::FrequencyCounter::new();

            for (idx, (comp, dct_coeffs)) in raw_dct.iter().enumerate() {
                let (qt, is_luma, ac_codes) = if *comp == 0 {
                    (&luma_qt, true, &ac_luma_arr)
                } else {
                    (&chroma_qt, false, &ac_chroma_arr)
                };
                let lambda = trellis::default_lambda(qt);
                let zz = trellis::trellis_quantize_with_codes(
                    dct_coeffs, qt, lambda, is_luma, Some(ac_codes),
                );

                match comp {
                    0 => luma_freq2.count_block(&zz),
                    1 => cb_freq2.count_block(&zz),
                    _ => cr_freq2.count_block(&zz),
                }
                blocks[idx] = (*comp, zz);
            }

            // Rebuild optimal tables from re-trellis'd coefficients
            let (dc_luma_len2, ac_luma_len2) = luma_freq2.build_optimal_tables();
            let (dc_chroma_len2, ac_chroma_len2) = if is_gray {
                (vec![0u8; 12], vec![0u8; 256])
            } else {
                let mut chroma_freq2 = entropy::FrequencyCounter::new();
                for i in 0..12 {
                    chroma_freq2.dc_freq[i] = cb_freq2.dc_freq[i] + cr_freq2.dc_freq[i];
                }
                for i in 0..256 {
                    chroma_freq2.ac_freq[i] = cb_freq2.ac_freq[i] + cr_freq2.ac_freq[i];
                }
                chroma_freq2.build_optimal_tables()
            };

            // EOB block-level optimization (mozjpeg trellis_eob_opt)
            // Note: for sequential JPEG this is a no-op — per-coefficient trellis
            // already handles AC zeroing. Meaningful savings only in progressive mode.
            if config.eob_optimize {
                trellis::eob_optimize_blocks(
                    &mut blocks,
                    &luma_qt,
                    &chroma_qt,
                    &ac_luma_arr,
                    &ac_chroma_arr,
                    1.0,
                );
            }

            // Use the refined tables
            // (shadow the outer bindings for the encode step below)
            let dc_luma_len = dc_luma_len2;
            let ac_luma_len = ac_luma_len2;
            let dc_chroma_len = dc_chroma_len2;
            let ac_chroma_len = ac_chroma_len2;

            // Write optimized DHT markers
            markers::write_dht(&mut out, 0, 0, &dc_luma_len);
            markers::write_dht(&mut out, 1, 0, &ac_luma_len);
            if !is_gray {
                markers::write_dht(&mut out, 0, 1, &dc_chroma_len);
                markers::write_dht(&mut out, 1, 1, &ac_chroma_len);
            }

            // SOS
            if is_gray {
                markers::write_sos(&mut out, &[(1, 0, 0)]);
            } else {
                markers::write_sos(&mut out, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
            }

            // Encode with final optimal tables
            let (h_blocks, v_blocks) = if is_gray {
                (1usize, 1usize)
            } else {
                color::subsampling_factors(config.subsampling)
            };
            let blocks_per_mcu = if is_gray {
                h_blocks * v_blocks
            } else {
                h_blocks * v_blocks + 2
            };

            let mcu_data = encode_blocks_with_tables(
                &blocks,
                config.restart_interval,
                blocks_per_mcu,
                || {
                    if is_gray {
                        entropy::InterleavedMcuEncoder::new_gray_custom(&dc_luma_len, &ac_luma_len)
                    } else {
                        entropy::InterleavedMcuEncoder::new_ycbcr_custom(
                            &dc_luma_len, &ac_luma_len,
                            &dc_chroma_len, &ac_chroma_len,
                        )
                    }
                },
            );
            let stuffed = markers::byte_stuff(&mcu_data);
            out.extend_from_slice(&stuffed);

            markers::write_eoi(&mut out);
            return Ok(out);
        }

        // Non-trellis path: just use pass-1 tables (no re-trellis needed)
        let dc_luma_len = dc_luma_len;
        let ac_luma_len = ac_luma_len;

        // Write optimized DHT markers
        markers::write_dht(&mut out, 0, 0, &dc_luma_len);
        markers::write_dht(&mut out, 1, 0, &ac_luma_len);
        if !is_gray {
            markers::write_dht(&mut out, 0, 1, &dc_chroma_len);
            markers::write_dht(&mut out, 1, 1, &ac_chroma_len);
        }

        // SOS
        if is_gray {
            markers::write_sos(&mut out, &[(1, 0, 0)]);
        } else {
            markers::write_sos(&mut out, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
        }

        // Second pass: encode pre-quantized blocks with optimal tables
        let (h_blocks, v_blocks) = if is_gray {
            (1usize, 1usize)
        } else {
            color::subsampling_factors(config.subsampling)
        };
        let blocks_per_mcu = if is_gray {
            h_blocks * v_blocks
        } else {
            h_blocks * v_blocks + 2
        };

        let mcu_data =
            encode_blocks_with_tables(&blocks, config.restart_interval, blocks_per_mcu, || {
                if is_gray {
                    entropy::InterleavedMcuEncoder::new_gray_custom(&dc_luma_len, &ac_luma_len)
                } else {
                    entropy::InterleavedMcuEncoder::new_ycbcr_custom(
                        &dc_luma_len,
                        &ac_luma_len,
                        &dc_chroma_len,
                        &ac_chroma_len,
                    )
                }
            });
        let stuffed = markers::byte_stuff(&mcu_data);
        out.extend_from_slice(&stuffed);
    } else {
        // Standard Huffman: single-pass with Annex K tables
        if is_gray {
            markers::write_dht(&mut out, 0, 0, &entropy::DC_LUMA_CODE_LENGTHS);
            markers::write_dht(&mut out, 1, 0, &entropy::AC_LUMA_CODE_LENGTHS);
        } else {
            markers::write_standard_huffman_tables(&mut out);
        }
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
            config.trellis,
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
    use_trellis: bool,
) {
    // First, compute all DCT coefficients for all blocks
    let all_coeffs =
        compute_all_dct_coefficients(ycbcr, is_gray, subsampling, luma_qt, chroma_qt, use_trellis);

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
        let (mcu_w, mcu_h) = if is_gray {
            (8u32, 8u32)
        } else {
            color::mcu_dimensions(subsampling)
        };
        let mcu_cols = ycbcr.width.div_ceil(mcu_w) as usize;
        let mcu_rows = ycbcr.height.div_ceil(mcu_h) as usize;

        let scan_data = encode_progressive_scan(
            &all_coeffs,
            comp_ids,
            *ss,
            *se,
            is_gray,
            subsampling,
            restart_interval,
            mcu_cols,
            mcu_rows,
            ycbcr.width,
            ycbcr.height,
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
    use_trellis: bool,
) -> std::collections::HashMap<u8, Vec<[i16; 64]>> {
    let mut result = std::collections::HashMap::new();

    let (mcu_w, mcu_h) = if is_gray {
        (8u32, 8u32)
    } else {
        color::mcu_dimensions(subsampling)
    };
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
                    let zz = if use_trellis {
                        let lambda = trellis::default_lambda(luma_qt);
                        trellis::trellis_quantize(&dct_out, luma_qt, lambda, true)
                    } else {
                        let mut quantized = [0i16; 64];
                        quantize::quantize(&dct_out, luma_qt, &mut quantized);
                        zigzag_reorder(&quantized)
                    };
                    y_blocks.push(zz);
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
                    let zz = if use_trellis {
                        let lambda = trellis::default_lambda(qt);
                        trellis::trellis_quantize(&dct_out, qt, lambda, false)
                    } else {
                        let mut quantized = [0i16; 64];
                        quantize::quantize(&dct_out, qt, &mut quantized);
                        zigzag_reorder(&quantized)
                    };
                    blocks.push(zz);
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
///
/// For interleaved scans (multiple components), encodes in MCU-interleaved
/// order per ITU-T T.81. For non-interleaved scans (single component),
/// encodes in component raster order per ITU-T T.81 A.2.3.
fn encode_progressive_scan(
    all_coeffs: &std::collections::HashMap<u8, Vec<[i16; 64]>>,
    comp_ids: &[u8],
    ss: u8,
    se: u8,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    _restart_interval: Option<u16>,
    mcu_cols: usize,
    mcu_rows: usize,
    img_width: u32,
    img_height: u32,
) -> Vec<u8> {
    use rasmcore_bitio::{BitOrder, BitWriter};

    let mut writer = BitWriter::new(BitOrder::MsbFirst);
    let mut prev_dc: std::collections::HashMap<u8, i16> =
        comp_ids.iter().map(|&id| (id, 0i16)).collect();
    let encoders = ProgressiveHuffmanEncoders::new();

    if comp_ids.len() == 1 {
        // Non-interleaved: component raster order (left-to-right, top-to-bottom
        // in the component's own block grid). Per ITU-T T.81 A.2.3.
        let comp_id = comp_ids[0];
        let blocks = match all_coeffs.get(&comp_id) {
            Some(b) => b,
            None => return writer.finish(),
        };
        let is_luma = comp_id == 1;

        let (h_samp, v_samp) = if comp_id == 1 && !is_gray {
            color::subsampling_factors(subsampling)
        } else {
            (1, 1)
        };

        let (h_max, v_max) = if is_gray {
            (1usize, 1usize)
        } else {
            color::subsampling_factors(subsampling)
        };

        // Xi, Yi per ITU-T T.81 A.1.1 — actual block count, not padded
        let xi = (img_width as usize * h_samp + h_max * 8 - 1) / (h_max * 8);
        let yi = (img_height as usize * v_samp + v_max * 8 - 1) / (v_max * 8);
        let blocks_per_mcu = h_samp * v_samp;

        for block_row in 0..yi {
            for block_col in 0..xi {
                // Map raster position to MCU-order buffer index
                let mc = block_col / h_samp;
                let mr = block_row / v_samp;
                let hb = block_col % h_samp;
                let vb = block_row % v_samp;
                let idx = (mr * mcu_cols + mc) * blocks_per_mcu + vb * h_samp + hb;

                encode_progressive_block(
                    &mut writer,
                    &blocks[idx],
                    ss,
                    se,
                    is_luma,
                    prev_dc.get_mut(&comp_id).unwrap(),
                    &encoders,
                );
            }
        }
    } else {
        // Interleaved: iterate per MCU (ITU-T T.81 compliant ordering)
        let blocks_per_mcu: Vec<(u8, usize)> = comp_ids
            .iter()
            .map(|&id| {
                if id == 1 && !is_gray {
                    let (h, v) = color::subsampling_factors(subsampling);
                    (id, h * v)
                } else {
                    (id, 1)
                }
            })
            .collect();

        let num_mcus = if is_gray {
            all_coeffs.get(&1).map_or(0, |b| b.len())
        } else {
            all_coeffs.get(&2).map_or(0, |b| b.len())
        };

        let mut block_offsets: std::collections::HashMap<u8, usize> =
            comp_ids.iter().map(|&id| (id, 0usize)).collect();

        for _mcu in 0..num_mcus {
            for &(comp_id, bpm) in &blocks_per_mcu {
                let blocks = match all_coeffs.get(&comp_id) {
                    Some(b) => b,
                    None => continue,
                };
                let is_luma = comp_id == 1;
                let offset = block_offsets.get_mut(&comp_id).unwrap();

                for bi in 0..bpm {
                    let block = &blocks[*offset + bi];
                    encode_progressive_block(
                        &mut writer,
                        block,
                        ss,
                        se,
                        is_luma,
                        prev_dc.get_mut(&comp_id).unwrap(),
                        &encoders,
                    );
                }
                *offset += bpm;
            }
        }
    }

    writer.finish()
}

/// Huffman encoders for progressive scans, built from the standard code lengths.
struct ProgressiveHuffmanEncoders {
    dc_luma: rasmcore_deflate::huffman::HuffmanEncoder,
    dc_chroma: rasmcore_deflate::huffman::HuffmanEncoder,
    ac_luma: rasmcore_deflate::huffman::HuffmanEncoder,
    ac_chroma: rasmcore_deflate::huffman::HuffmanEncoder,
}

impl ProgressiveHuffmanEncoders {
    fn new() -> Self {
        Self {
            dc_luma: rasmcore_deflate::huffman::HuffmanEncoder::from_code_lengths(
                &entropy::DC_LUMA_CODE_LENGTHS,
            ),
            dc_chroma: rasmcore_deflate::huffman::HuffmanEncoder::from_code_lengths(
                &entropy::DC_CHROMA_CODE_LENGTHS,
            ),
            ac_luma: rasmcore_deflate::huffman::HuffmanEncoder::from_code_lengths(
                &entropy::AC_LUMA_CODE_LENGTHS,
            ),
            ac_chroma: rasmcore_deflate::huffman::HuffmanEncoder::from_code_lengths(
                &entropy::AC_CHROMA_CODE_LENGTHS,
            ),
        }
    }
}

/// Encode one block's contribution to a progressive scan.
fn encode_progressive_block(
    writer: &mut rasmcore_bitio::BitWriter,
    block: &[i16; 64],
    ss: u8,
    se: u8,
    is_luma: bool,
    prev_dc: &mut i16,
    encoders: &ProgressiveHuffmanEncoders,
) {
    if ss == 0 {
        let dc = block[0];
        let diff = dc - *prev_dc;
        *prev_dc = dc;

        let cat = entropy::magnitude_category(diff as i32);
        let dc_enc = if is_luma {
            &encoders.dc_luma
        } else {
            &encoders.dc_chroma
        };
        dc_enc.write_symbol(writer, cat as u16);
        if cat > 0 {
            encode_coefficient_bits(writer, diff as i32, cat);
        }
    }

    if se > 0 {
        let start = if ss == 0 { 1 } else { ss as usize };
        let end = se as usize;
        let mut zero_run = 0u8;
        let ac_enc = if is_luma {
            &encoders.ac_luma
        } else {
            &encoders.ac_chroma
        };

        for coeff_ref in &block[start..=end] {
            let coeff = *coeff_ref;
            if coeff == 0 {
                zero_run += 1;
                if zero_run == 16 {
                    ac_enc.write_symbol(writer, 0xF0);
                    zero_run = 0;
                }
            } else {
                let cat = entropy::magnitude_category(coeff as i32);
                let symbol = (zero_run << 4) | cat;
                ac_enc.write_symbol(writer, symbol as u16);
                encode_coefficient_bits(writer, coeff as i32, cat);
                zero_run = 0;
            }
        }

        if zero_run > 0 {
            ac_enc.write_symbol(writer, 0x00);
        }
    }
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
/// First pass: quantize all blocks and collect Huffman symbol frequencies.
/// Returns (luma_counter, chroma_counter, all_quantized_blocks).
/// Each quantized block is stored as (component_index, [i16; 64]) in MCU order.
fn collect_frequencies(
    ycbcr: &color::YcbcrImage,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    luma_qt: &[u16; 64],
    chroma_qt: &[u16; 64],
    use_trellis: bool,
) -> (
    entropy::FrequencyCounter,
    entropy::FrequencyCounter,
    Vec<(usize, [i16; 64])>,
    Vec<(usize, [i32; 64])>, // raw DCT coefficients for re-trellis
) {
    let (mcu_w, mcu_h) = if is_gray {
        (8u32, 8u32)
    } else {
        color::mcu_dimensions(subsampling)
    };
    let mcu_cols = ycbcr.width.div_ceil(mcu_w) as usize;
    let mcu_rows = ycbcr.height.div_ceil(mcu_h) as usize;

    let (h_blocks, v_blocks) = if is_gray {
        (1usize, 1usize)
    } else {
        color::subsampling_factors(subsampling)
    };

    let mut luma_freq = entropy::FrequencyCounter::new();
    let mut cb_freq = entropy::FrequencyCounter::new();
    let mut cr_freq = entropy::FrequencyCounter::new();
    let cap = mcu_rows * mcu_cols * (h_blocks * v_blocks + if is_gray { 0 } else { 2 });
    let mut blocks = Vec::with_capacity(cap);
    let mut raw_dct = Vec::with_capacity(cap);

    for mcu_row in 0..mcu_rows {
        for mcu_col in 0..mcu_cols {
            // Y blocks
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
                    let zz = if use_trellis {
                        let lambda = trellis::default_lambda(luma_qt);
                        trellis::trellis_quantize(&dct_out, luma_qt, lambda, true)
                    } else {
                        let mut quantized = [0i16; 64];
                        quantize::quantize(&dct_out, luma_qt, &mut quantized);
                        zigzag_reorder(&quantized)
                    };
                    luma_freq.count_block(&zz);
                    blocks.push((0, zz));
                    raw_dct.push((0, dct_out));
                }
            }

            // Cb/Cr blocks — use per-component frequency counters
            // (DC prediction is per-component in the actual encoder)
            if !is_gray {
                for (comp_idx, plane, freq) in [
                    (1usize, &ycbcr.cb, &mut cb_freq),
                    (2, &ycbcr.cr, &mut cr_freq),
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
                    let zz = if use_trellis {
                        let lambda = trellis::default_lambda(chroma_qt);
                        trellis::trellis_quantize(&dct_out, chroma_qt, lambda, false)
                    } else {
                        let mut quantized = [0i16; 64];
                        quantize::quantize(&dct_out, chroma_qt, &mut quantized);
                        zigzag_reorder(&quantized)
                    };
                    freq.count_block(&zz);
                    blocks.push((comp_idx, zz));
                    raw_dct.push((comp_idx, dct_out));
                }
            }
        }
    }

    // Merge Cb and Cr frequencies into one chroma table
    // (both components share the same Huffman tables in JPEG)
    let mut chroma_freq = entropy::FrequencyCounter::new();
    for i in 0..12 {
        chroma_freq.dc_freq[i] = cb_freq.dc_freq[i] + cr_freq.dc_freq[i];
    }
    for i in 0..256 {
        chroma_freq.ac_freq[i] = cb_freq.ac_freq[i] + cr_freq.ac_freq[i];
    }

    (luma_freq, chroma_freq, blocks, raw_dct)
}

/// Second pass: encode pre-quantized blocks with the given encoder factory.
fn encode_blocks_with_tables(
    blocks: &[(usize, [i16; 64])],
    restart_interval: Option<u16>,
    blocks_per_mcu: usize,
    make_encoder: impl Fn() -> entropy::InterleavedMcuEncoder,
) -> Vec<u8> {
    let mut enc = make_encoder();
    let mut all_data = Vec::new();
    let mut mcu_count = 0u32;
    let mut block_idx = 0;

    while block_idx < blocks.len() {
        // Check restart
        #[allow(clippy::collapsible_if)]
        if let Some(interval) = restart_interval {
            if mcu_count > 0 && (mcu_count as u16).is_multiple_of(interval) {
                all_data.extend_from_slice(&enc.finish());
                enc = make_encoder();
                let stuffed = markers::byte_stuff(&all_data);
                all_data = stuffed;
                markers::write_rst(&mut all_data, mcu_count / interval as u32 - 1);
            }
        }

        // Encode one MCU worth of blocks
        for _ in 0..blocks_per_mcu {
            if block_idx >= blocks.len() {
                break;
            }
            let (comp, ref coeffs) = blocks[block_idx];
            enc.encode_block(comp, coeffs);
            block_idx += 1;
        }

        mcu_count += 1;
    }

    all_data.extend_from_slice(&enc.finish());
    all_data
}

fn encode_mcus(
    ycbcr: &color::YcbcrImage,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    luma_qt: &[u16; 64],
    chroma_qt: &[u16; 64],
    restart_interval: Option<u16>,
    use_trellis: bool,
) -> Vec<u8> {
    // For grayscale, MCU is always 8x8 (single component, 1x1 sampling).
    // For color, MCU depends on chroma subsampling.
    let (mcu_w, mcu_h) = if is_gray {
        (8u32, 8u32)
    } else {
        color::mcu_dimensions(subsampling)
    };
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
                    let zz = if use_trellis {
                        let lambda = trellis::adaptive_lambda(&dct_out, 1.0);
                        trellis::trellis_quantize(&dct_out, luma_qt, lambda, true)
                    } else {
                        let mut quantized = [0i16; 64];
                        quantize::quantize(&dct_out, luma_qt, &mut quantized);
                        zigzag_reorder(&quantized)
                    };
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
                    let zz = if use_trellis {
                        let lambda = trellis::adaptive_lambda(&dct_out, 1.0);
                        trellis::trellis_quantize(&dct_out, chroma_qt, lambda, false)
                    } else {
                        let mut quantized = [0i16; 64];
                        quantize::quantize(&dct_out, chroma_qt, &mut quantized);
                        zigzag_reorder(&quantized)
                    };
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

/// Encode all MCUs using arithmetic coding and return the raw entropy data.
fn encode_mcus_arithmetic(
    ycbcr: &color::YcbcrImage,
    is_gray: bool,
    subsampling: ChromaSubsampling,
    luma_qt: &[u16; 64],
    chroma_qt: &[u16; 64],
) -> Vec<u8> {
    let (mcu_w, mcu_h) = if is_gray {
        (8u32, 8u32)
    } else {
        color::mcu_dimensions(subsampling)
    };
    let mcu_cols = ycbcr.width.div_ceil(mcu_w) as usize;
    let mcu_rows = ycbcr.height.div_ceil(mcu_h) as usize;

    let (h_blocks, v_blocks) = if is_gray {
        (1usize, 1usize)
    } else {
        color::subsampling_factors(subsampling)
    };

    let mut enc = qm_coder::JpegArithEncoder::new();

    for mcu_row in 0..mcu_rows {
        for mcu_col in 0..mcu_cols {
            // Y blocks (DC table 0, AC table 0)
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
                    enc.encode_block(&zz, 0, 0, 0); // ci=0, dc_tbl=0, ac_tbl=0
                }
            }

            // Cb and Cr blocks (DC table 1, AC table 1)
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
                    enc.encode_block(&zz, comp_idx, 1, 1); // dc_tbl=1, ac_tbl=1
                }
            }
        }
    }

    enc.finish()
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
        assert!(!config.turbo);
        assert!(!config.progressive);
        assert_eq!(config.subsampling, ChromaSubsampling::Quarter420);
        assert!(!config.arithmetic_coding);
        assert!(config.restart_interval.is_none());
        // Default enables all optimizations (mozjpeg-quality output)
        assert!(config.optimize_huffman);
        assert!(config.trellis);
        assert!(config.eob_optimize);
        assert_eq!(config.sample_precision, SamplePrecision::Eight);
        assert_eq!(config.quant_preset, crate::quantize::QuantPreset::Robidoux);
        assert!(config.custom_quant_tables.is_none());
    }

    #[test]
    fn turbo_config_disables_optimizations() {
        let config = EncodeConfig::turbo(75);
        assert!(config.turbo);
        let eff = config.effective();
        assert!(!eff.trellis);
        assert!(!eff.optimize_huffman);
        assert!(!eff.eob_optimize);
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

#[cfg(test)]
mod trellis_integration_tests {
    use super::*;

    #[test]
    fn trellis_reduces_file_size() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32u8 {
            for x in 0..32u8 {
                pixels.push(x.wrapping_mul(8));
                pixels.push(y.wrapping_mul(8));
                pixels.push(128);
            }
        }

        let baseline = encode(
            &pixels,
            32,
            32,
            PixelFormat::Rgb8,
            &EncodeConfig {
                quality: 75,
                trellis: false,
                ..Default::default()
            },
        )
        .unwrap();

        let with_trellis = encode(
            &pixels,
            32,
            32,
            PixelFormat::Rgb8,
            &EncodeConfig {
                quality: 75,
                trellis: true,
                ..Default::default()
            },
        )
        .unwrap();

        eprintln!(
            "Trellis: baseline={} trellis={} bytes (savings={:.1}%)",
            baseline.len(),
            with_trellis.len(),
            (1.0 - with_trellis.len() as f64 / baseline.len() as f64) * 100.0
        );

        // Trellis should produce equal or smaller files
        assert!(
            with_trellis.len() <= baseline.len(),
            "trellis ({}) should be <= baseline ({})",
            with_trellis.len(),
            baseline.len()
        );

        // Both should be decodable
        let ref_baseline = image::load_from_memory_with_format(&baseline, image::ImageFormat::Jpeg);
        assert!(ref_baseline.is_ok(), "baseline decode failed");

        let ref_trellis =
            image::load_from_memory_with_format(&with_trellis, image::ImageFormat::Jpeg);
        assert!(ref_trellis.is_ok(), "trellis decode failed");
    }

    #[test]
    fn trellis_deterministic() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let config = EncodeConfig {
            quality: 85,
            trellis: true,
            ..Default::default()
        };
        let a = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        let b = encode(&pixels, 16, 16, PixelFormat::Rgb8, &config).unwrap();
        assert_eq!(a, b, "trellis encoding should be deterministic");
    }
}

#[test]
fn trellis_savings_complex_content() {
    // Create complex content: checkerboard + noise
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u16 {
        for x in 0..64u16 {
            let checker = if ((x / 8) + (y / 8)) % 2 == 0 {
                200u8
            } else {
                50u8
            };
            let noise = ((x.wrapping_mul(17).wrapping_add(y.wrapping_mul(31))) % 20) as u8;
            pixels.push(checker.wrapping_add(noise));
            pixels.push(checker.wrapping_sub(noise.min(checker)));
            pixels.push(128);
        }
    }

    for &quality in &[50u8, 75, 85] {
        let baseline = encode(
            &pixels,
            64,
            64,
            PixelFormat::Rgb8,
            &EncodeConfig {
                quality,
                trellis: false,
                ..Default::default()
            },
        )
        .unwrap();
        let with_trellis = encode(
            &pixels,
            64,
            64,
            PixelFormat::Rgb8,
            &EncodeConfig {
                quality,
                trellis: true,
                ..Default::default()
            },
        )
        .unwrap();
        let savings = (1.0 - with_trellis.len() as f64 / baseline.len() as f64) * 100.0;
        eprintln!(
            "Q{quality}: baseline={} trellis={} savings={savings:.1}%",
            baseline.len(),
            with_trellis.len()
        );

        // Both decodable
        assert!(
            image::load_from_memory_with_format(&with_trellis, image::ImageFormat::Jpeg).is_ok()
        );
    }
}

#[test]
fn trellis_savings_256x256() {
    let mut pixels = Vec::with_capacity(256 * 256 * 3);
    for y in 0..256u16 {
        for x in 0..256u16 {
            let checker = if ((x / 8) + (y / 8)) % 2 == 0 {
                200u8
            } else {
                50u8
            };
            let noise = ((x.wrapping_mul(17).wrapping_add(y.wrapping_mul(31))) % 30) as u8;
            pixels.push(checker.wrapping_add(noise));
            pixels.push(checker.wrapping_sub(noise.min(checker)));
            pixels.push(128);
        }
    }

    for &quality in &[50u8, 75, 85] {
        let baseline = encode(
            &pixels,
            256,
            256,
            PixelFormat::Rgb8,
            &EncodeConfig {
                quality,
                trellis: false,
                quant_preset: quantize::QuantPreset::AnnexK,
                ..Default::default()
            },
        )
        .unwrap();
        let with_trellis = encode(
            &pixels,
            256,
            256,
            PixelFormat::Rgb8,
            &EncodeConfig {
                quality,
                trellis: true,
                quant_preset: quantize::QuantPreset::AnnexK,
                ..Default::default()
            },
        )
        .unwrap();
        let savings = (1.0 - with_trellis.len() as f64 / baseline.len() as f64) * 100.0;
        eprintln!(
            "256x256 Q{quality}: baseline={} trellis={} savings={savings:.1}%",
            baseline.len(),
            with_trellis.len()
        );
        assert!(
            image::load_from_memory_with_format(&with_trellis, image::ImageFormat::Jpeg).is_ok(),
            "Q{quality}: trellis output not decodable"
        );
    }
}

#[test]
fn eob_optimize_sequential_no_regression() {
    // EOB block-level optimization follows mozjpeg's trellis_eob_opt (disabled by
    // default in mozjpeg). For sequential baseline JPEG, the per-coefficient trellis
    // already optimally handles AC zeroing. Block-level EOB runs only benefit
    // progressive mode (EOBn symbols). Verify: no regression and output is valid.
    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u16 {
        for x in 0..128u16 {
            let checker = if ((x / 16) + (y / 16)) % 2 == 0 { 180u8 } else { 80u8 };
            let noise = ((x.wrapping_mul(17).wrapping_add(y.wrapping_mul(31))) % 25) as u8;
            pixels.push(checker.wrapping_add(noise));
            pixels.push(checker.wrapping_sub(noise.min(checker)));
            pixels.push(128);
        }
    }

    for &quality in &[50u8, 75] {
        let without_eob = encode(
            &pixels, 128, 128, PixelFormat::Rgb8,
            &EncodeConfig {
                quality,
                optimize_huffman: true,
                trellis: true,
                eob_optimize: false,
                ..Default::default()
            },
        ).unwrap();

        let with_eob = encode(
            &pixels, 128, 128, PixelFormat::Rgb8,
            &EncodeConfig {
                quality,
                optimize_huffman: true,
                trellis: true,
                eob_optimize: true,
                ..Default::default()
            },
        ).unwrap();

        // For sequential mode: should produce identical output (no block-run benefit)
        assert_eq!(
            without_eob.len(), with_eob.len(),
            "Q{quality}: sequential EOB opt should produce same size"
        );

        assert!(
            image::load_from_memory_with_format(&with_eob, image::ImageFormat::Jpeg).is_ok(),
            "Q{quality}: EOB optimized output not decodable"
        );
    }
}

#[test]
fn turbo_vs_default_speed_and_validity() {
    // Verify turbo mode is significantly faster and produces valid output
    let mut pixels = Vec::with_capacity(256 * 256 * 3);
    for y in 0..256u16 {
        for x in 0..256u16 {
            let v = ((x * 255) / 255) as u8;
            let noise = ((x.wrapping_mul(17).wrapping_add(y.wrapping_mul(31))) % 30) as u8;
            pixels.push(v.wrapping_add(noise));
            pixels.push(v.wrapping_sub(noise.min(v)));
            pixels.push(128);
        }
    }

    // Turbo mode
    let turbo_start = std::time::Instant::now();
    let turbo_jpeg = encode(
        &pixels, 256, 256, PixelFormat::Rgb8,
        &EncodeConfig::turbo(75),
    ).unwrap();
    let turbo_time = turbo_start.elapsed();

    // Default mode (trellis + optimize_huffman)
    let default_start = std::time::Instant::now();
    let default_jpeg = encode(
        &pixels, 256, 256, PixelFormat::Rgb8,
        &EncodeConfig { quality: 75, ..Default::default() },
    ).unwrap();
    let default_time = default_start.elapsed();

    let speedup = default_time.as_secs_f64() / turbo_time.as_secs_f64().max(0.0001);
    eprintln!(
        "turbo={:.1}ms default={:.1}ms speedup={speedup:.1}x | turbo_size={} default_size={}",
        turbo_time.as_secs_f64() * 1000.0,
        default_time.as_secs_f64() * 1000.0,
        turbo_jpeg.len(),
        default_jpeg.len(),
    );

    // Both must be valid
    assert!(image::load_from_memory_with_format(&turbo_jpeg, image::ImageFormat::Jpeg).is_ok());
    assert!(image::load_from_memory_with_format(&default_jpeg, image::ImageFormat::Jpeg).is_ok());

    // Turbo should be faster (at least 2x in debug mode, typically 3-10x in release)
    assert!(
        speedup > 1.5,
        "turbo should be at least 1.5x faster, got {speedup:.1}x"
    );

    // Default should produce smaller files (trellis is more efficient)
    assert!(
        default_jpeg.len() < turbo_jpeg.len(),
        "default (optimized) should be smaller: {} vs {}",
        default_jpeg.len(),
        turbo_jpeg.len()
    );
}
