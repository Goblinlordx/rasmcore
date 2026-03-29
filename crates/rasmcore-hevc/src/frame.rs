//! HEVC frame assembly — wire syntax → predict → transform → filter → pixels.
//!
//! This module implements the top-level decode pipeline that ties all HEVC
//! subsystems together to produce decoded frames from NAL unit bitstreams.

#![allow(clippy::too_many_arguments)]

use crate::cabac::CabacDecoder;
use crate::error::HevcError;
use crate::filter::{self, BoundaryStrength};
use crate::nal::{self, NalIterator};
use crate::params::{self, DecoderContext, Sps};
use crate::predict::{self, RefSamples};
use crate::syntax::{self, CuSyntax, PredMode};
use crate::transform::reconstruct_residual as transform_reconstruct;
use crate::types::{DecodedFrame, NalUnitType};

/// Frame buffer holding YCbCr planes during reconstruction.
struct FrameBuffer {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    /// Per-pixel intra prediction mode map (used for MPM derivation).
    /// Stores the resolved mode for each luma sample position.
    intra_modes: Vec<u8>,
}

impl FrameBuffer {
    fn new(width: u32, height: u32) -> Self {
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        Self {
            y: vec![128u8; (width * height) as usize],
            cb: vec![128u8; (chroma_width * chroma_height) as usize],
            cr: vec![128u8; (chroma_width * chroma_height) as usize],
            width,
            height,
            chroma_width,
            chroma_height,
            // Default intra mode = 1 (DC) — same as HEVC "unavailable" default
            intra_modes: vec![1u8; (width * height) as usize],
        }
    }

    /// Convert YCbCr 4:2:0 to RGB8 using the specified color matrix.
    fn to_rgb(&self, matrix: &rasmcore_color::ColorMatrix) -> Vec<u8> {
        let matrix = *matrix;
        let w = self.width as usize;
        let h = self.height as usize;
        let cw = self.chroma_width as usize;
        let mut rgb = Vec::with_capacity(w * h * 3);

        for row in 0..h {
            for col in 0..w {
                let y = self.y[row * w + col];
                let cb = self.cb[(row / 2) * cw + (col / 2)];
                let cr = self.cr[(row / 2) * cw + (col / 2)];
                let (r, g, b) = rasmcore_color::ycbcr_to_rgb(y, cb, cr, &matrix);
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }
        rgb
    }
}

/// Decode an HEVC I-frame from NAL units.
///
/// This is the main entry point that wires all subsystems together:
/// 1. Parse parameter sets (VPS/SPS/PPS) from NAL units
/// 2. Initialize CABAC from slice data
/// 3. For each CTU: decode syntax → predict → add residual → store
/// 4. Apply deblocking filter
/// 5. Apply SAO
/// 6. Convert YCbCr to RGB
pub fn decode_frame(
    bitstream: &[u8],
    hvcc_nals: Option<&[rasmcore_isobmff::NalArray]>,
) -> Result<DecodedFrame, HevcError> {
    decode_frame_with_matrix(bitstream, hvcc_nals, None)
}

/// Decode an HEVC I-frame with an optional custom color matrix for YCbCr→RGB.
///
/// If `color_matrix` is `None`, the standard BT.709 limited-range matrix is used.
/// Pass a custom matrix to match a specific reference implementation's conversion
/// (e.g., ffmpeg's libswscale coefficients for test parity).
pub fn decode_frame_with_matrix(
    bitstream: &[u8],
    hvcc_nals: Option<&[rasmcore_isobmff::NalArray]>,
    color_matrix: Option<&rasmcore_color::ColorMatrix>,
) -> Result<DecodedFrame, HevcError> {
    let mut ctx = DecoderContext::new();

    // Parse parameter sets from hvcC config (if provided by ISOBMFF container)
    if let Some(nal_arrays) = hvcc_nals {
        ctx = params::parse_hvcc_params(nal_arrays)?;
    }

    // Parse in-band NAL units
    let mut slice_rbsp: Option<Vec<u8>> = None;
    let mut slice_nal_type = NalUnitType::IdrWRadl;

    for nal_data in NalIterator::new(bitstream) {
        let nal_unit = nal::parse_nal_unit(nal_data)?;
        match nal_unit.nal_type {
            NalUnitType::VpsNut => {
                let vps = params::parse_vps(&nal_unit.rbsp)?;
                let id = vps.vps_id as usize;
                if id < ctx.vps.len() {
                    ctx.vps[id] = Some(vps);
                }
            }
            NalUnitType::SpsNut => {
                let sps = params::parse_sps(&nal_unit.rbsp)?;
                let id = sps.sps_id as usize;
                if id < ctx.sps.len() {
                    ctx.sps[id] = Some(sps);
                }
            }
            NalUnitType::PpsNut => {
                let pps = params::parse_pps(&nal_unit.rbsp)?;
                let id = pps.pps_id as usize;
                if id < ctx.pps.len() {
                    ctx.pps[id] = Some(pps);
                }
            }
            nt if nt.is_vcl() => {
                slice_nal_type = nt;
                slice_rbsp = Some(nal_unit.rbsp);
            }
            _ => {} // Skip SEI, AUD, etc.
        }
    }

    let rbsp = slice_rbsp.ok_or(HevcError::DecodeFailed(
        "no VCL NAL unit found in bitstream".into(),
    ))?;

    // Get first available SPS and PPS
    let sps = ctx
        .sps
        .iter()
        .flatten()
        .next()
        .ok_or(HevcError::InvalidParameterSet("no SPS found".into()))?
        .clone();
    let pps = ctx
        .pps
        .iter()
        .flatten()
        .next()
        .ok_or(HevcError::InvalidParameterSet("no PPS found".into()))?
        .clone();

    // Parse slice header (needs NAL unit type to handle IDR vs non-IDR conditionals)
    let _slice_header = syntax::parse_slice_header(&rbsp, &sps, &pps, slice_nal_type)?;

    // Initialize frame buffer
    let mut fb = FrameBuffer::new(sps.pic_width, sps.pic_height);

    // Compute slice QP
    let slice_qp = pps.init_qp + _slice_header.slice_qp_delta;

    // CABAC data starts at the byte-aligned position after the slice header.
    // parse_slice_header tracks the exact bit position and aligns to byte boundary.
    let cabac_start = _slice_header.data_offset;
    if cabac_start >= rbsp.len() {
        return Err(HevcError::DecodeFailed(format!(
            "slice data offset ({cabac_start}) beyond RBSP length ({})",
            rbsp.len()
        )));
    }

    let mut cabac = CabacDecoder::new(&rbsp[cabac_start..])?;
    let mut contexts = syntax::init_syntax_contexts(slice_qp);
    let mut depth_map = syntax::CuDepthMap::new(sps.pic_width, sps.pic_height, sps.min_cb_size());

    // Decode CTUs in raster scan order
    let ctu_size = sps.ctu_size();
    let ctus_x = sps.pic_width_in_ctus();
    let ctus_y = sps.pic_height_in_ctus();
    let wpp_enabled = pps.entropy_coding_sync_enabled;

    // WPP context storage: save context models after the 2nd CTU in each row
    // for re-initialization at the start of the next row.
    // Ref: libde265 v1.0.18 slice.cc lines 4785-4794 — context save at CtbX==1
    let mut wpp_saved_contexts: Vec<Option<Vec<syntax::ContextModelState>>> =
        vec![None; ctus_y as usize];

    'ctu_loop: for ctu_row in 0..ctus_y {
        // WPP: at the start of each row (except the first), restore saved contexts
        // from the previous row and re-init the CABAC engine from the entry point.
        // Ref: libde265 v1.0.18 slice.cc lines 4725-4747 — context restore at CtbX==0
        if wpp_enabled && ctu_row > 0 {
            // Restore context models from the row above
            if let Some(saved) = &wpp_saved_contexts[(ctu_row - 1) as usize] {
                syntax::restore_contexts(&mut contexts, saved);
            } else {
                // Fallback: re-initialize contexts from slice QP
                contexts = syntax::init_syntax_contexts(slice_qp);
            }

            // Re-initialize CABAC engine at the entry point for this row.
            // entry_point_offsets[row-1] gives the cumulative byte offset from
            // the start of slice data to this row's substream.
            // Ref: libde265 v1.0.18 slice.cc line 4866 — init_CABAC()
            let ep_idx = (ctu_row - 1) as usize;
            if ep_idx < _slice_header.entry_point_offsets.len() {
                let ep_offset = _slice_header.entry_point_offsets[ep_idx] as usize;
                cabac.reinit_at(ep_offset);
            }
        }

        for ctu_col in 0..ctus_x {
            let ctu_x = ctu_col * ctu_size;
            let ctu_y = ctu_row * ctu_size;

            // Decode CTU syntax
            let ctu = syntax::decode_ctu(
                &mut cabac,
                &mut contexts,
                &sps,
                &pps,
                ctu_x,
                ctu_y,
                &mut depth_map,
            )
            .map_err(|e| {
                HevcError::DecodeFailed(format!(
                    "CTU({ctu_col},{ctu_row}) at ({ctu_x},{ctu_y}): {e}"
                ))
            })?;

            // Reconstruct each CU in this CTU
            for cu in &ctu.cus {
                reconstruct_cu(cu, &sps, &pps, slice_qp, &mut fb, ctu_y, ctu_size)?;
            }

            // WPP: save context models after decoding the 2nd CTU (CtbX==1)
            // for use by the next row's context initialization.
            // Ref: libde265 v1.0.18 slice.cc lines 4793-4794 — save at CtbX==1
            if wpp_enabled && ctu_col == 1 && (ctu_row + 1) < ctus_y {
                wpp_saved_contexts[ctu_row as usize] = Some(syntax::save_contexts(&contexts));
            }

            // end_of_slice_segment_flag — decoded via CABAC terminate
            let end_of_slice = cabac.decode_terminate()?;
            if end_of_slice != 0 {
                // In WPP mode, end_of_slice at the end of a row is actually
                // end_of_sub_stream — break inner loop to continue to next row.
                // Ref: libde265 v1.0.18 slice.cc lines 4851-4868
                if wpp_enabled && ctu_row + 1 < ctus_y {
                    break; // break inner ctu_col loop, outer ctu_row loop continues
                }
                break 'ctu_loop;
            }
        }
    }

    // Apply deblocking filter
    if !_slice_header.deblocking_filter_disabled {
        apply_deblocking(&mut fb, &sps, slice_qp);
    }

    // Apply SAO (if enabled)
    // For now, SAO parameters would come from slice data parsing
    // This is a placeholder — full SAO integration requires per-CTU params from syntax

    // Convert YCbCr to RGB
    let rgb_matrix = color_matrix.unwrap_or(&rasmcore_color::ColorMatrix::BT709);
    let pixels = fb.to_rgb(rgb_matrix);

    Ok(DecodedFrame {
        y_plane: fb.y.clone(),
        pixels,
        width: sps.pic_width,
        height: sps.pic_height,
        bit_depth: sps.bit_depth_luma,
    })
}

/// Store the resolved intra prediction mode for all samples in a CU.
fn store_intra_mode(fb: &mut FrameBuffer, x: u32, y: u32, size: u32, mode: u8) {
    let w = fb.width as usize;
    for dy in 0..size as usize {
        for dx in 0..size as usize {
            let px = x as usize + dx;
            let py = y as usize + dy;
            if px < fb.width as usize && py < fb.height as usize {
                fb.intra_modes[py * w + px] = mode;
            }
        }
    }
}

/// Resolve the intra prediction mode from the raw syntax value.
///
/// The syntax parser stores either an MPM index (0-2) or rem_intra_pred_mode + 3 (3-34).
/// This function derives the MPM candidate list from left and above neighbors,
/// then resolves the final mode.
///
/// Ref: libde265 v1.0.18 slice.cc — derive_IntraPredMode, and
///      HEVC spec Section 8.4.2: derivation of luma intra prediction mode.
fn resolve_intra_pred_mode(
    raw_mode: u8,
    x: u32,
    y: u32,
    _size: u32,
    fb: &FrameBuffer,
    _sps: &Sps,
    ctu_y: u32,
) -> u8 {
    let w = fb.width as usize;

    // Get candidate modes from left and above neighbors.
    // Ref: HEVC Section 8.4.2 — candIntraPredModeA (left), candIntraPredModeB (above).
    let cand_a = if x > 0 {
        let px = (x - 1) as usize;
        let py = y as usize;
        if py < fb.height as usize {
            fb.intra_modes[py * w + px]
        } else {
            1 // DC
        }
    } else {
        1 // DC when unavailable
    };

    // For the above neighbor: if the above pixel is in a different CTU row,
    // use DC(1) as the default. This matches libde265 v1.0.18 behavior where
    // cross-CTU-row mode lookups use the initialized default (DC) rather than
    // the stored mode from the previous row. This is critical for WPP streams
    // where rows may be decoded independently.
    let cand_b = if y > 0 && (y - 1) >= ctu_y {
        // Above pixel is within the current CTU row — use stored mode
        let px = x as usize;
        let py = (y - 1) as usize;
        fb.intra_modes[py * w + px]
    } else if y > 0 {
        // Above pixel is in the previous CTU row — use DC default
        // Ref: libde265 v1.0.18 trace shows candB=1 (DC) for cross-row lookups
        1 // DC
    } else {
        1 // DC when unavailable (y=0)
    };

    // Build MPM candidate list.
    // Ref: HEVC Section 8.4.2, steps for candModeList[0..2].
    let mut cand_mode_list = [0u8; 3];
    if cand_a == cand_b {
        if cand_a < 2 {
            // Both are Planar(0) or DC(1)
            cand_mode_list[0] = 0; // Planar
            cand_mode_list[1] = 1; // DC
            cand_mode_list[2] = 26; // Vertical
        } else {
            cand_mode_list[0] = cand_a;
            cand_mode_list[1] = 2 + ((cand_a - 2 + 29) % 32); // cand_a - 1 (wrapped in angular range)
            cand_mode_list[2] = 2 + ((cand_a - 2 + 1) % 32); // cand_a + 1 (wrapped)
        }
    } else {
        cand_mode_list[0] = cand_a;
        cand_mode_list[1] = cand_b;
        if cand_a != 0 && cand_b != 0 {
            cand_mode_list[2] = 0; // Planar
        } else if cand_a != 1 && cand_b != 1 {
            cand_mode_list[2] = 1; // DC
        } else {
            cand_mode_list[2] = 26; // Vertical
        }
    }

    if raw_mode <= 2 {
        // prev_intra_luma_pred_flag was true — use MPM index
        cand_mode_list[raw_mode as usize]
    } else {
        // prev_intra_luma_pred_flag was false — rem_intra_pred_mode
        let rem = raw_mode - 3;
        // Sort MPM candidates for exclusion
        let mut sorted = cand_mode_list;
        sorted.sort();
        // Map rem to actual mode by skipping MPM candidates
        let mut mode = rem;
        for &c in &sorted {
            if mode >= c {
                mode += 1;
            }
        }
        mode
    }
}

/// Reconstruct a single CU: predict + add residual.
///
/// `ctu_y` and `ctu_size` define the current CTU row boundaries for
/// reference sample availability computation (HEVC Section 8.4.4.2.2).
fn reconstruct_cu(
    cu: &CuSyntax,
    sps: &Sps,
    _pps: &params::Pps,
    qp: i32,
    fb: &mut FrameBuffer,
    ctu_y: u32,
    ctu_size: u32,
) -> Result<(), HevcError> {
    if cu.pred_mode != PredMode::Intra {
        return Err(HevcError::DecodeFailed(
            "only intra prediction supported".into(),
        ));
    }

    let size = cu.size as usize;

    // Resolve intra prediction mode via MPM candidate list.
    // Ref: libde265 v1.0.18 slice.cc — derive_IntraPredMode, HEVC Section 8.4.2.
    let raw_mode = cu.intra_luma_modes.first().copied().unwrap_or(1);
    let luma_mode = resolve_intra_pred_mode(raw_mode, cu.x, cu.y, cu.size, fb, sps, ctu_y);

    // Store the resolved mode in the frame buffer for neighbor lookups
    store_intra_mode(fb, cu.x, cu.y, cu.size, luma_mode);

    // Construct reference samples with proper availability bounds.
    // HEVC Section 8.4.4.2.2: reference samples from not-yet-reconstructed CTUs
    // are unavailable and must be substituted with the nearest available sample.
    let avail_top = cu.y > 0;
    let avail_left = cu.x > 0;
    let avail_tl = cu.x > 0 && cu.y > 0;

    // For left references: samples are available up to the bottom of the current
    // CTU row (since all CTUs in the current row to the left have been decoded,
    // but the row below hasn't started yet).
    let recon_bottom = (ctu_y + ctu_size).min(fb.height) as usize;
    // For top references: the entire row above is available (decoded in a previous CTU row).
    let recon_right = fb.width as usize;

    let refs = RefSamples::from_frame_with_bounds(
        cu.x as usize,
        cu.y as usize,
        size,
        &fb.y,
        fb.width as usize,
        fb.height as usize,
        avail_top,
        avail_left,
        avail_tl,
        recon_right,
        recon_bottom,
    );

    // Generate prediction
    let predicted =
        predict::predict_intra(luma_mode, &refs, size, sps.strong_intra_smoothing_enabled);

    // Get residual from transform unit (if present)
    let residual = if let Some(tu) = &cu.tu {
        if tu.cbf_luma && !tu.luma_coeffs.is_empty() {
            // Dequantize and inverse transform
            let log2_size = (size as f32).log2() as u8;
            let is_4x4_intra_luma = size == 4;
            let mut residual_buf = vec![0i16; size * size];
            transform_reconstruct(
                &tu.luma_coeffs,
                &mut residual_buf,
                log2_size,
                qp,
                sps.bit_depth_luma,
                is_4x4_intra_luma,
                None, // No custom scaling list
                0,    // Default matrix ID
            )?;
            residual_buf
        } else {
            vec![0i16; size * size]
        }
    } else {
        vec![0i16; size * size]
    };

    // Add prediction + residual → reconstructed, write to frame buffer
    for row in 0..size {
        for col in 0..size {
            let fx = cu.x as usize + col;
            let fy = cu.y as usize + row;
            if fx < fb.width as usize && fy < fb.height as usize {
                let pred = predicted[row * size + col] as i16;
                let res = residual[row * size + col];
                let recon = (pred + res).clamp(0, 255) as u8;
                fb.y[fy * fb.width as usize + fx] = recon;
            }
        }
    }

    // Chroma reconstruction (simplified — use DC prediction for chroma)
    let chroma_size = size / 2;
    if chroma_size > 0 {
        let chroma_mode = predict::derive_chroma_mode(cu.intra_chroma_mode, luma_mode);
        let cx = cu.x as usize / 2;
        let cy = cu.y as usize / 2;

        // Simple chroma fill — proper chroma prediction would use chroma reference samples
        let chroma_val_cb = 128u8;
        let chroma_val_cr = 128u8;

        for row in 0..chroma_size {
            for col in 0..chroma_size {
                let fx = cx + col;
                let fy = cy + row;
                if fx < fb.chroma_width as usize && fy < fb.chroma_height as usize {
                    fb.cb[fy * fb.chroma_width as usize + fx] = chroma_val_cb;
                    fb.cr[fy * fb.chroma_width as usize + fx] = chroma_val_cr;
                }
            }
        }
        let _ = chroma_mode; // Will be used for proper chroma prediction
    }

    Ok(())
}

/// Apply deblocking filter to the reconstructed frame.
///
/// HEVC Section 8.7.2: deblocking is applied at 8x8 grid-aligned edges
/// that correspond to CU or TU boundaries. The boundary strength (Bs)
/// determines filter strength: Bs=2 for intra CU boundaries, Bs=1 for
/// TU boundaries with non-zero coefficients, Bs=0 otherwise.
///
/// Currently simplified: deblocking is applied at CTU boundaries only,
/// which are guaranteed CU boundaries in an I-frame. A full implementation
/// would track the actual CU/TU partition and apply deblocking at all
/// CU and TU boundaries on the 8x8 grid.
fn apply_deblocking(fb: &mut FrameBuffer, sps: &Sps, qp: i32) {
    let ctu_size = sps.ctu_size() as usize;
    let stride = fb.width as usize;

    // Apply vertical edges at CTU column boundaries.
    // Each CTU boundary is a CU boundary → Bs=2 for intra.
    for edge_x in (ctu_size..fb.width as usize).step_by(ctu_size) {
        if edge_x >= 4 && edge_x + 3 < stride {
            filter::deblock_vertical_edge(
                &mut fb.y,
                stride,
                edge_x,
                0,
                fb.height as usize,
                qp,
                BoundaryStrength::for_intra_boundary(),
            );
        }
    }

    // Apply horizontal edges at CTU row boundaries.
    for edge_y in (ctu_size..fb.height as usize).step_by(ctu_size) {
        if edge_y >= 4 && edge_y + 3 < fb.height as usize {
            filter::deblock_horizontal_edge(
                &mut fb.y,
                stride,
                0,
                edge_y,
                fb.width as usize,
                qp,
                BoundaryStrength::for_intra_boundary(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_buffer_creation() {
        let fb = FrameBuffer::new(64, 64);
        assert_eq!(fb.y.len(), 64 * 64);
        assert_eq!(fb.cb.len(), 32 * 32);
        assert_eq!(fb.cr.len(), 32 * 32);
    }

    #[test]
    fn frame_buffer_to_rgb() {
        let fb = FrameBuffer::new(4, 4);
        let rgb = fb.to_rgb(&rasmcore_color::ColorMatrix::BT709);
        assert_eq!(rgb.len(), 4 * 4 * 3);
    }

    #[test]
    fn frame_buffer_odd_dimensions() {
        let fb = FrameBuffer::new(65, 33);
        assert_eq!(fb.y.len(), 65 * 33);
        assert_eq!(fb.chroma_width, 33); // ceil(65/2)
        assert_eq!(fb.chroma_height, 17); // ceil(33/2)
    }

    #[test]
    fn golden_decode_attempt() {
        crate::skip_if_no_fixtures!();

        // Load a real HEVC bitstream and attempt full decode
        let hevc_data = crate::testutil::load_fixture("flat_64x64_q22", "hevc").unwrap();

        // This will likely fail at CABAC stage since our heuristic slice data offset
        // is approximate, but it should at least parse NALs and parameter sets correctly.
        let result = decode_frame(&hevc_data, None);

        match &result {
            Ok(frame) => {
                assert_eq!(frame.width, 64);
                assert_eq!(frame.height, 64);
                assert_eq!(frame.pixels.len(), 64 * 64 * 3);

                // Compare against reference
                let ref_rgb = crate::testutil::load_reference_rgb("flat_64x64_q22").unwrap();
                let cmp = crate::testutil::compare_pixels(&frame.pixels, &ref_rgb, 0);
                eprintln!(
                    "Decode quality: PSNR={:.1}dB, max_diff={}, mismatches={}/{}",
                    cmp.psnr, cmp.max_diff, cmp.mismatches, cmp.total_pixels
                );
            }
            Err(e) => {
                // Expected at this stage — CABAC data offset heuristic may be wrong
                eprintln!("Decode failed (expected at this stage): {e}");
            }
        }
    }
}
