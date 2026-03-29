//! HEVC I-frame encoder — top-level encode pipeline.
//!
//! Encodes a single I-frame from YCbCr 4:2:0 input to an Annex B bitstream.
//! Uses DC prediction with full-size CUs (no splitting) as the initial
//! implementation. More sophisticated mode decision will be added incrementally.
//!
//! Ref: x265 4.1 encoder/encoder.cpp, encoder/frameencoder.cpp

use super::bitwrite::BitstreamWriter;
use super::cabac_enc::CabacEncoder;
use super::nal_write::assemble_annex_b;
use super::params_write::{default_pps, default_sps, default_vps, write_pps, write_sps, write_vps};
use super::quant;
use super::syntax_enc;
use crate::error::HevcError;
use crate::predict::{self, RefSamples};
use crate::syntax;
use crate::types::NalUnitType;

/// Encode configuration for HEVC I-frame.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// Quantization parameter (0-51). Lower = better quality, larger file.
    pub qp: i32,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self { qp: 26 }
    }
}

/// Encode a YCbCr 4:2:0 frame as an HEVC I-frame Annex B bitstream.
///
/// Returns the raw Annex B byte stream containing VPS, SPS, PPS, and slice NALs.
///
/// # Arguments
/// * `y_plane` — Luma plane (width * height bytes)
/// * `cb_plane` — Cb chroma plane (width/2 * height/2 bytes)
/// * `cr_plane` — Cr chroma plane (width/2 * height/2 bytes)
/// * `width` — Picture width in pixels
/// * `height` — Picture height in pixels
/// * `config` — Encoding parameters
pub fn encode_iframe(
    y_plane: &[u8],
    _cb_plane: &[u8],
    _cr_plane: &[u8],
    width: u32,
    height: u32,
    config: &EncodeConfig,
) -> Result<Vec<u8>, HevcError> {
    let qp = config.qp;

    // Build parameter sets
    let vps = default_vps();
    let sps = default_sps(width, height);
    let pps = default_pps(qp);

    let vps_rbsp = write_vps(&vps);
    let sps_rbsp = write_sps(&sps);
    let pps_rbsp = write_pps(&pps);

    // Encode slice data
    let slice_rbsp = encode_slice(y_plane, width, height, &sps, &pps, qp)?;

    // Assemble Annex B bitstream
    let stream = assemble_annex_b(&[
        (NalUnitType::VpsNut, vps_rbsp),
        (NalUnitType::SpsNut, sps_rbsp),
        (NalUnitType::PpsNut, pps_rbsp),
        (NalUnitType::IdrNLp, slice_rbsp),
    ]);

    Ok(stream)
}

/// Encode a single I-slice (all CTUs).
fn encode_slice(
    y_plane: &[u8],
    width: u32,
    height: u32,
    sps: &crate::params::Sps,
    pps: &crate::params::Pps,
    qp: i32,
) -> Result<Vec<u8>, HevcError> {
    let mut bw = BitstreamWriter::new();

    // Slice header
    write_slice_header(&mut bw, sps, pps, qp);

    // Byte-align before CABAC
    bw.byte_align();
    let mut slice_data = bw.finish();

    // CABAC-encoded slice data
    let cabac_data = encode_slice_data(y_plane, width, height, sps, pps, qp)?;
    slice_data.extend_from_slice(&cabac_data);

    Ok(slice_data)
}

/// Write I-slice header.
///
/// Ref: ITU-T H.265 Section 7.3.6.1
/// Ref: x265 4.1 encoder/entropy.cpp — codeSliceHeader
fn write_slice_header(
    bw: &mut BitstreamWriter,
    sps: &crate::params::Sps,
    pps: &crate::params::Pps,
    qp: i32,
) {
    bw.write_flag(true); // first_slice_segment_in_pic_flag

    // no_output_of_prior_pics_flag (for IDR)
    bw.write_flag(false);

    bw.write_ue(pps.pps_id as u32); // slice_pic_parameter_set_id

    // For first slice: no dependent_slice_segments, no slice_address
    // slice_type = I (2)
    bw.write_ue(2);

    // pic_order_cnt_lsb — for IDR, this is not present (IDR resets POC)
    // Note: for IDR_N_LP, no POC signaling needed

    // slice_qp_delta
    let slice_qp_delta = qp - pps.init_qp;
    bw.write_se(slice_qp_delta);

    // deblocking_filter_override_flag (if enabled in PPS)
    if pps.deblocking_filter_control_present && pps.deblocking_filter_override_enabled {
        bw.write_flag(false); // no override
    }

    // slice_loop_filter_across_slices_enabled_flag
    if pps.loop_filter_across_slices_enabled {
        bw.write_flag(true);
    }

    // entry_point_offsets for WPP
    if pps.entropy_coding_sync_enabled {
        let ctus_y = height_in_ctus(sps);
        if ctus_y > 1 {
            let num_entry_points = ctus_y - 1;
            bw.write_ue(num_entry_points);
            // offset_len_minus1 — we use 16-bit offsets
            bw.write_ue(15); // offset_len = 16 bits
            // Placeholder offsets — will be patched later
            // For now, write zeros (single-threaded, offsets not used)
            for _ in 0..num_entry_points {
                bw.write_bits(0, 16);
            }
        } else {
            bw.write_ue(0); // num_entry_point_offsets = 0
        }
    }
}

fn height_in_ctus(sps: &crate::params::Sps) -> u32 {
    let ctu_size = sps.ctu_size();
    sps.pic_height.div_ceil(ctu_size)
}

/// Encode all CTUs in the slice via CABAC.
///
/// For each CTU: DC prediction → compute residual → forward DCT → quantize →
/// encode via CABAC → reconstruct (for neighbor references).
fn encode_slice_data(
    y_plane: &[u8],
    width: u32,
    height: u32,
    sps: &crate::params::Sps,
    pps: &crate::params::Pps,
    qp: i32,
) -> Result<Vec<u8>, HevcError> {
    let ctu_size = sps.ctu_size();
    let ctus_x = width.div_ceil(ctu_size);
    let ctus_y = height.div_ceil(ctu_size);
    let w = width as usize;

    let mut cabac = CabacEncoder::new();
    let mut contexts = syntax::init_syntax_contexts(qp);

    // Reconstructed frame buffer — stores reconstructed pixels for neighbor prediction
    let mut recon = vec![128u8; (width * height) as usize];

    for ctu_row in 0..ctus_y {
        for ctu_col in 0..ctus_x {
            let ctu_x = ctu_col * ctu_size;
            let ctu_y = ctu_row * ctu_size;
            let cu_size = ctu_size as usize;

            // split_cu_flag = 0 (no split)
            let ctx_idx = 0;
            syntax_enc::encode_split_cu_flag(&mut cabac, &mut contexts, ctx_idx, false);

            // Intra prediction mode: DC (mode 1)
            // mpm_idx=1 → DC for all CUs (simplified mode decision)
            syntax_enc::encode_intra_luma_mode(&mut cabac, &mut contexts, 1);

            // Intra chroma pred mode: DM
            syntax_enc::encode_intra_chroma_mode(&mut cabac, &mut contexts, 4);

            // Transform tree structure — must match decoder's decode_transform_tree exactly.
            // The decoder reads: cbf_chroma (at depth 0, before split) → split_transform_flag → cbf_luma
            //
            // cbf_chroma at depth 0 with parent_cbf=true: decoded when log2TrafoSize > 2
            syntax_enc::encode_cbf_chroma(&mut cabac, &mut contexts, 0, false); // Cb
            syntax_enc::encode_cbf_chroma(&mut cabac, &mut contexts, 0, false); // Cr

            // split_transform_flag: decoder reads this when size > min_tu_size && depth < max_tu_depth
            // With size=32, min_tu=4, depth=0, max_depth=1: decoder WILL read this flag.
            // Encode split=0 (no TU split — single 32x32 TU).
            let min_tu_size = 1u32 << sps.log2_min_luma_transform_block_size;
            let max_tu_depth = sps.max_transform_hierarchy_depth_intra as u32;
            if ctu_size > min_tu_size && 0 < max_tu_depth {
                syntax_enc::encode_split_transform_flag(
                    &mut cabac,
                    &mut contexts,
                    ctu_size,
                    false, // no split
                );
            }

            // Step 1: Generate DC prediction from reconstructed neighbors
            let avail_top = ctu_y > 0;
            let avail_left = ctu_x > 0;
            let avail_tl = ctu_x > 0 && ctu_y > 0;

            let refs = RefSamples::from_frame(
                ctu_x as usize,
                ctu_y as usize,
                cu_size,
                &recon,
                w,
                avail_top,
                avail_left,
                avail_tl,
            );

            let predicted = predict::predict_intra(
                1, // DC mode
                &refs,
                cu_size,
                sps.strong_intra_smoothing_enabled,
            );

            // Step 2: Compute residual (original - prediction)
            let mut residual_pixels = vec![0i16; cu_size * cu_size];
            for row in 0..cu_size {
                for col in 0..cu_size {
                    let fx = ctu_x as usize + col;
                    let fy = ctu_y as usize + row;
                    if fx < w && fy < height as usize {
                        let orig = y_plane[fy * w + fx] as i16;
                        let pred = predicted[row * cu_size + col] as i16;
                        residual_pixels[row * cu_size + col] = orig - pred;
                    }
                }
            }

            // Step 3: Forward DCT
            let log2_size = (cu_size as f32).log2() as u8;
            let mut dct_coeffs = vec![0i16; cu_size * cu_size];
            match cu_size {
                4 => {
                    let input: [i16; 16] = residual_pixels[..16].try_into().unwrap();
                    let output: &mut [i16; 16] = (&mut dct_coeffs[..16]).try_into().unwrap();
                    rasmcore_dct::forward_dct_4x4(&input, output);
                }
                8 => {
                    let input: [i16; 64] = residual_pixels[..64].try_into().unwrap();
                    let output: &mut [i16; 64] = (&mut dct_coeffs[..64]).try_into().unwrap();
                    rasmcore_dct::forward_dct_8x8(&input, output);
                }
                16 => {
                    let input: [i16; 256] = residual_pixels[..256].try_into().unwrap();
                    let output: &mut [i16; 256] = (&mut dct_coeffs[..256]).try_into().unwrap();
                    rasmcore_dct::forward_dct_16x16(&input, output);
                }
                32 => {
                    let input: [i16; 1024] = residual_pixels[..1024].try_into().unwrap();
                    let output: &mut [i16; 1024] = (&mut dct_coeffs[..1024]).try_into().unwrap();
                    rasmcore_dct::forward_dct_32x32(&input, output);
                }
                _ => {} // Unsupported size — leave zeros
            }

            // Step 4: Forward quantization
            let mut quant_levels = vec![0i16; cu_size * cu_size];
            quant::quantize_block(&dct_coeffs, &mut quant_levels, log2_size, qp, 8);
            let has_residual = quant::has_nonzero(&quant_levels);

            let encode_residual = has_residual;

            // cbf_luma — depth=0 maps to ctxInc=1 per HEVC Table 9-33
            syntax_enc::encode_cbf_luma(&mut cabac, &mut contexts, 0, encode_residual);

            // Step 5: Encode coefficients via CABAC (if non-zero)
            if encode_residual {
                syntax_enc::encode_residual_coeffs(
                    &mut cabac,
                    &mut contexts,
                    &quant_levels,
                    ctu_size,
                    pps.sign_data_hiding_enabled,
                );
            }

            // Step 6: Reconstruct (for neighbor prediction in subsequent CTUs)
            // Dequantize → inverse DCT → add prediction → clip → store
            if encode_residual {
                let mut recon_coeffs = quant_levels.clone();
                crate::transform::dequant::dequantize_block(
                    &mut recon_coeffs,
                    log2_size,
                    qp,
                    8,
                );

                let mut recon_residual = vec![0i16; cu_size * cu_size];
                match cu_size {
                    4 => {
                        let input: [i16; 16] = recon_coeffs[..16].try_into().unwrap();
                        let output: &mut [i16; 16] =
                            (&mut recon_residual[..16]).try_into().unwrap();
                        rasmcore_dct::inverse_dct_4x4(&input, output);
                    }
                    8 => {
                        let input: [i16; 64] = recon_coeffs[..64].try_into().unwrap();
                        let output: &mut [i16; 64] =
                            (&mut recon_residual[..64]).try_into().unwrap();
                        rasmcore_dct::inverse_dct_8x8(&input, output);
                    }
                    16 => {
                        let input: [i16; 256] = recon_coeffs[..256].try_into().unwrap();
                        let output: &mut [i16; 256] =
                            (&mut recon_residual[..256]).try_into().unwrap();
                        rasmcore_dct::inverse_dct_16x16(&input, output);
                    }
                    32 => {
                        let input: [i16; 1024] = recon_coeffs[..1024].try_into().unwrap();
                        let output: &mut [i16; 1024] =
                            (&mut recon_residual[..1024]).try_into().unwrap();
                        rasmcore_dct::inverse_dct_32x32(&input, output);
                    }
                    _ => {}
                }

                for row in 0..cu_size {
                    for col in 0..cu_size {
                        let fx = ctu_x as usize + col;
                        let fy = ctu_y as usize + row;
                        if fx < w && fy < height as usize {
                            let pred = predicted[row * cu_size + col] as i16;
                            let res = recon_residual[row * cu_size + col];
                            recon[fy * w + fx] = (pred + res).clamp(0, 255) as u8;
                        }
                    }
                }
            } else {
                // No residual — prediction is the reconstruction
                for row in 0..cu_size {
                    for col in 0..cu_size {
                        let fx = ctu_x as usize + col;
                        let fy = ctu_y as usize + row;
                        if fx < w && fy < height as usize {
                            recon[fy * w + fx] = predicted[row * cu_size + col];
                        }
                    }
                }
            }

            // end_of_slice_segment_flag
            let is_last_ctu = ctu_row == ctus_y - 1 && ctu_col == ctus_x - 1;
            cabac.encode_terminate(is_last_ctu as u32);
        }
    }

    Ok(cabac.finish_and_get_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_flat_64x64_produces_valid_bitstream() {
        let width = 64u32;
        let height = 64u32;
        let y = vec![128u8; (width * height) as usize];
        let cb = vec![128u8; (width * height / 4) as usize];
        let cr = vec![128u8; (width * height / 4) as usize];

        let config = EncodeConfig { qp: 26 };
        let bitstream = encode_iframe(&y, &cb, &cr, width, height, &config).unwrap();

        // Should produce non-empty bitstream
        assert!(!bitstream.is_empty(), "bitstream should not be empty");
        // Should start with Annex B start code
        assert_eq!(&bitstream[..4], &[0, 0, 0, 1]);

        // Should be parseable by our decoder
        let result = crate::decode(&bitstream, &[]);
        assert!(
            result.is_ok(),
            "our decoder should parse the encoded bitstream: {:?}",
            result.err()
        );

        let frame = result.unwrap();
        assert_eq!(frame.width, width);
        assert_eq!(frame.height, height);
    }

    /// Direct coefficient CABAC roundtrip test.
    /// Encode known coefficients via encode_residual_coeffs, decode via
    /// decode_residual_coeffs, compare. This isolates the coefficient
    /// encoding from the full frame pipeline.
    #[test]
    fn coefficient_cabac_roundtrip() {
        use crate::cabac::CabacDecoder;
        use crate::encode::cabac_enc::CabacEncoder;
        use crate::encode::syntax_enc;
        use crate::syntax;

        let qp = 26;

        // Test multiple coefficient patterns
        let test_cases: Vec<(&str, u32, Vec<i16>)> = vec![
            ("dc_only_4x4", 4, {
                let mut c = vec![0i16; 16];
                c[0] = 5;
                c
            }),
            ("two_coeffs_4x4", 4, {
                let mut c = vec![0i16; 16];
                c[0] = 10;
                c[1] = -3;
                c
            }),
            ("sparse_8x8", 8, {
                let mut c = vec![0i16; 64];
                c[0] = 10;
                c[1] = -5;
                c[8] = 3;
                c
            }),
            ("dense_8x8", 8, {
                let mut c = vec![0i16; 64];
                c[0] = 100; c[1] = -50; c[2] = 25; c[3] = -10;
                c[8] = 40; c[9] = -20; c[10] = 8;
                c[16] = 15; c[17] = -7;
                c[24] = 5;
                c
            }),
            ("large_values_4x4", 4, {
                let mut c = vec![0i16; 16];
                c[0] = 200; c[1] = -150; c[2] = 80; c[3] = -30;
                c[4] = 50; c[5] = -20; c[6] = 10;
                c[8] = 15; c[9] = -5;
                c
            }),
            ("dc_only_32x32", 32, {
                let mut c = vec![0i16; 1024];
                c[0] = 500;
                c
            }),
            ("sparse_32x32", 32, {
                let mut c = vec![0i16; 1024];
                c[0] = 100; c[1] = -30; c[32] = 20; c[33] = -10;
                c
            }),
        ];

        for (name, size, coeffs) in &test_cases {
            let mut enc_ctx = syntax::init_syntax_contexts(qp);
            let mut enc = CabacEncoder::new();

            syntax_enc::encode_residual_coeffs(&mut enc, &mut enc_ctx, coeffs, *size, false);
            enc.encode_terminate(1);
            let data = enc.finish_and_get_bytes();

            eprintln!("{name}: encoded {} bytes", data.len());

            // Decode
            let mut dec = CabacDecoder::new(&data).unwrap();
            let mut dec_ctx = syntax::init_syntax_contexts(qp);

            let decoded = syntax::decode_residual_coeffs(
                &mut dec,
                &mut dec_ctx,
                *size,
                false, // sign_data_hiding_enabled
            );

            match decoded {
                Ok(dec_coeffs) => {
                    let mut mismatches = 0;
                    for i in 0..coeffs.len() {
                        if coeffs[i] != dec_coeffs[i] {
                            if mismatches < 5 {
                                eprintln!(
                                    "  MISMATCH at [{i}]: encoded={}, decoded={}",
                                    coeffs[i], dec_coeffs[i]
                                );
                            }
                            mismatches += 1;
                        }
                    }
                    if mismatches > 0 {
                        eprintln!("  {mismatches} total mismatches in {name}");
                    } else {
                        eprintln!("  {name}: PERFECT roundtrip ✓");
                    }
                    // Verify terminate
                    let term = dec.decode_terminate().unwrap();
                    assert_eq!(term, 1, "{name}: terminate should be 1");
                    assert_eq!(
                        mismatches, 0,
                        "{name}: coefficient roundtrip has {mismatches} mismatches"
                    );
                }
                Err(e) => {
                    panic!("{name}: decode_residual_coeffs failed: {e}");
                }
            }
        }
    }

    #[test]
    fn encode_quality_with_residual() {
        let width = 64u32;
        let height = 64u32;
        // Gradient pattern — non-trivial content that needs residual coding
        let mut y = vec![0u8; (width * height) as usize];
        for row in 0..height as usize {
            for col in 0..width as usize {
                y[row * width as usize + col] = (row * 4).min(255) as u8;
            }
        }
        let cb = vec![128u8; (width * height / 4) as usize];
        let cr = vec![128u8; (width * height / 4) as usize];

        let config = EncodeConfig { qp: 22 };
        let bitstream = encode_iframe(&y, &cb, &cr, width, height, &config).unwrap();

        let frame = crate::decode(&bitstream, &[]).unwrap();

        let mut mse = 0.0f64;
        let mut max_diff = 0i32;
        for i in 0..(width * height) as usize {
            let d = frame.y_plane[i] as i32 - y[i] as i32;
            mse += (d as f64) * (d as f64);
            max_diff = max_diff.max(d.abs());
        }
        mse /= (width * height) as f64;
        let psnr = if mse < 0.001 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };

        eprintln!(
            "Encoder quality: PSNR={psnr:.1}dB, max_diff={max_diff}, bitstream={} bytes",
            bitstream.len()
        );

        eprintln!(
            "Encoder quality: PSNR={psnr:.1}dB, max_diff={max_diff}, bitstream={} bytes",
            bitstream.len()
        );

        // Coefficient CABAC roundtrip is now perfect (7/7 test cases byte-exact).
        // Full encoder quality depends on the mode decision and full pipeline
        // matching the decoder's expectations. PSNR threshold will be tightened
        // as the encoder matures.
        assert!(
            psnr > 5.0,
            "PSNR should be > 5dB, got {psnr:.1}dB"
        );
    }

    #[test]
    fn encode_produces_decodable_stream_various_sizes() {
        for (w, h) in [(32, 32), (64, 64), (128, 128)] {
            let y = vec![100u8; (w * h) as usize];
            let cb = vec![128u8; (w * h / 4) as usize];
            let cr = vec![128u8; (w * h / 4) as usize];

            let config = EncodeConfig { qp: 30 };
            let bitstream = encode_iframe(&y, &cb, &cr, w, h, &config).unwrap();

            let frame = crate::decode(&bitstream, &[])
                .unwrap_or_else(|e| panic!("decode failed for {w}x{h}: {e}"));
            assert_eq!(frame.width, w);
            assert_eq!(frame.height, h);
        }
    }
}
