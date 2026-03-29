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
use super::syntax_enc;
use crate::cabac::ContextModel;
use crate::error::HevcError;
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
fn encode_slice_data(
    y_plane: &[u8],
    width: u32,
    height: u32,
    sps: &crate::params::Sps,
    _pps: &crate::params::Pps,
    qp: i32,
) -> Result<Vec<u8>, HevcError> {
    let ctu_size = sps.ctu_size();
    let ctus_x = width.div_ceil(ctu_size);
    let ctus_y = height.div_ceil(ctu_size);

    let mut cabac = CabacEncoder::new();
    let mut contexts = syntax::init_syntax_contexts(qp);

    for ctu_row in 0..ctus_y {
        for ctu_col in 0..ctus_x {
            let ctu_x = ctu_col * ctu_size;
            let ctu_y = ctu_row * ctu_size;
            let cu_size = ctu_size.min(width - ctu_x).min(height - ctu_y);

            // Encode CU: no split, single CU = CTU size
            // split_cu_flag = 0 (no split, at max depth)
            // For a single-CU CTU, split_cu_flag is not coded when
            // CU size == CTU size and depth == 0 and max_depth allows no split.
            // With our SPS (log2_diff=1), the CTU can split once.
            // We encode split_cu_flag = 0 to signal no split.
            let ctx_idx = 0; // split_cu_flag context for depth 0
            syntax_enc::encode_split_cu_flag(&mut cabac, &mut contexts, ctx_idx, false);

            // Intra prediction mode: DC (mode 1)
            // prev_intra_luma_pred_flag = 1 (use MPM)
            // For the first CU with no neighbors: MPM candidates = [Planar(0), DC(1), VER(26)]
            // mpm_idx = 1 → DC
            syntax_enc::encode_intra_luma_mode(&mut cabac, &mut contexts, 1); // mpm_idx=1=DC

            // Intra chroma pred mode: DM (derived from luma)
            syntax_enc::encode_intra_chroma_mode(&mut cabac, &mut contexts, 4);

            // Transform tree: single TU = CU size, no split
            // cbf_chroma (before split decision for depth 0)
            syntax_enc::encode_cbf_chroma(&mut cabac, &mut contexts, 0, false); // Cb
            syntax_enc::encode_cbf_chroma(&mut cabac, &mut contexts, 0, false); // Cr

            // cbf_luma
            let has_residual = has_nonzero_residual(y_plane, width, ctu_x, ctu_y, cu_size, qp);
            syntax_enc::encode_cbf_luma(&mut cabac, &mut contexts, 0, has_residual);

            // For now: skip residual coding (cbf_luma = false in most cases)
            // Full coefficient encoding will be added in the residual phase

            // end_of_slice_segment_flag (terminate bin)
            let is_last_ctu = ctu_row == ctus_y - 1 && ctu_col == ctus_x - 1;
            cabac.encode_terminate(is_last_ctu as u32);
        }
    }

    Ok(cabac.finish_and_get_bytes())
}

/// Quick check if a CU has non-zero residual after DC prediction + quantization.
/// For the initial encoder, we always signal cbf_luma=false to produce a
/// prediction-only bitstream (simplest valid output).
fn has_nonzero_residual(
    _y_plane: &[u8],
    _width: u32,
    _ctu_x: u32,
    _ctu_y: u32,
    _cu_size: u32,
    _qp: i32,
) -> bool {
    false // No residual for now — prediction-only output
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
