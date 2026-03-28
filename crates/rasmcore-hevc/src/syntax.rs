//! HEVC syntax parsing — slice header, CTU/CU/PU/TU hierarchy.
//!
//! ITU-T H.265 Sections 7.3.6-7.3.8.
//! Decodes syntax elements from the CABAC bitstream and stores them
//! in per-CTU data structures for the reconstruction pipeline.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::bitread::HevcBitReader;
use crate::cabac::{CabacDecoder, ContextModel, SliceType};
use crate::error::HevcError;
use crate::params::{Pps, Sps};

/// Parsed slice header.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_slice_segment_in_pic: bool,
    pub slice_type: SliceType,
    pub slice_pic_parameter_set_id: u8,
    pub slice_qp_delta: i32,
    pub slice_cb_qp_offset: i32,
    pub slice_cr_qp_offset: i32,
    pub deblocking_filter_disabled: bool,
    pub slice_beta_offset: i32,
    pub slice_tc_offset: i32,
    pub slice_sao_luma_flag: bool,
    pub slice_sao_chroma_flag: bool,
    /// Byte offset where slice data (CABAC) begins in the RBSP.
    pub data_offset: usize,
}

/// Decoded syntax elements for a single Coding Unit.
#[derive(Debug, Clone)]
pub struct CuSyntax {
    pub x: u32,
    pub y: u32,
    pub size: u32,
    pub pred_mode: PredMode,
    pub part_mode: PartMode,
    pub intra_luma_modes: Vec<u8>,
    pub intra_chroma_mode: u8,
    pub tu: Option<TuSyntax>,
}

/// Prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredMode {
    Intra,
    Inter, // Not used for HEIC I-frames
}

/// Partition mode for intra CUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartMode {
    /// Single 2Nx2N partition.
    Part2Nx2N,
    /// Four NxN partitions (only for smallest CU size).
    PartNxN,
}

/// Decoded transform unit syntax.
#[derive(Debug, Clone)]
pub struct TuSyntax {
    pub x: u32,
    pub y: u32,
    pub size: u32,
    pub cbf_luma: bool,
    pub cbf_cb: bool,
    pub cbf_cr: bool,
    /// Decoded residual coefficients for luma (if cbf_luma).
    pub luma_coeffs: Vec<i16>,
    /// Sub-TUs (for recursive TU split).
    pub children: Vec<TuSyntax>,
}

/// All decoded syntax for a CTU.
#[derive(Debug, Clone)]
pub struct CtuSyntax {
    pub x: u32,
    pub y: u32,
    pub size: u32,
    pub cus: Vec<CuSyntax>,
}

/// Parse slice header from RBSP data (for I-slices).
///
/// ITU-T H.265 Section 7.3.6.1.
pub fn parse_slice_header(
    rbsp: &[u8],
    sps: &Sps,
    pps: &Pps,
    nal_unit_type: crate::types::NalUnitType,
) -> Result<SliceHeader, HevcError> {
    let mut r = HevcBitReader::new(rbsp);

    let first_slice_segment_in_pic = r.read_flag()?;

    // no_output_of_prior_pics_flag — present for IRAP (BLA, IDR, CRA)
    if nal_unit_type.is_irap() {
        let _no_output_of_prior_pics_flag = r.read_flag()?;
    }

    let slice_pic_parameter_set_id = r.read_ue()? as u8;

    // dependent_slice_segment_flag (if enabled in PPS)
    if !first_slice_segment_in_pic && pps.dependent_slice_segments_enabled {
        let _dependent = r.read_flag()?;
    }

    // slice_segment_address (if not first slice)
    if !first_slice_segment_in_pic {
        let ctb_count = sps.pic_width_in_ctus() * sps.pic_height_in_ctus();
        let bits = (ctb_count as f32).log2().ceil() as u8;
        if bits > 0 {
            let _addr = r.read_u(bits)?;
        }
    }

    // Skip extra slice header bits from PPS
    for _ in 0..pps.num_extra_slice_header_bits {
        let _ = r.read_flag()?;
    }

    let slice_type_val = r.read_ue()?;
    let slice_type = match slice_type_val {
        0 => SliceType::B,
        1 => SliceType::P,
        2 => SliceType::I,
        _ => SliceType::I,
    };

    // output_flag_present
    if pps.output_flag_present {
        let _pic_output_flag = r.read_flag()?;
    }

    // For I-slices: pic_order_cnt_lsb and short_term_ref_pic_set are not present
    // (they're only in non-IDR slices)

    // slice_qp_delta
    let slice_qp_delta = r.read_se()?;

    // Chroma QP offsets
    let mut slice_cb_qp_offset = 0i32;
    let mut slice_cr_qp_offset = 0i32;
    if pps.slice_chroma_qp_offsets_present {
        slice_cb_qp_offset = r.read_se()?;
        slice_cr_qp_offset = r.read_se()?;
    }

    // Deblocking filter
    let mut deblocking_filter_disabled = pps.deblocking_filter_disabled;
    let mut slice_beta_offset = pps.beta_offset;
    let mut slice_tc_offset = pps.tc_offset;

    if pps.deblocking_filter_control_present && pps.deblocking_filter_override_enabled {
        let deblocking_filter_override = r.read_flag()?;
        if deblocking_filter_override {
            deblocking_filter_disabled = r.read_flag()?;
            if !deblocking_filter_disabled {
                slice_beta_offset = r.read_se()? * 2;
                slice_tc_offset = r.read_se()? * 2;
            }
        }
    }

    // SAO flags
    let slice_sao_luma_flag = if sps.sample_adaptive_offset_enabled {
        r.read_flag()?
    } else {
        false
    };
    let slice_sao_chroma_flag = if sps.sample_adaptive_offset_enabled && sps.chroma_format_idc > 0 {
        r.read_flag()?
    } else {
        false
    };

    // byte_alignment() — HEVC spec Section 7.3.2.11
    // Read alignment_bit_equal_to_one (1) then alignment_bit_equal_to_zero (0..7)
    // This aligns to the next byte boundary where CABAC data starts.
    let _alignment_bit = r.read_flag()?; // should be 1
    r.align_to_byte();

    // Record the exact byte position where CABAC slice data begins
    let data_offset = r.byte_position();

    Ok(SliceHeader {
        first_slice_segment_in_pic,
        slice_type,
        slice_pic_parameter_set_id,
        slice_qp_delta,
        slice_cb_qp_offset,
        slice_cr_qp_offset,
        deblocking_filter_disabled,
        slice_beta_offset,
        slice_tc_offset,
        slice_sao_luma_flag,
        slice_sao_chroma_flag,
        data_offset,
    })
}

/// Decode a CTU's syntax elements from the CABAC bitstream.
///
/// This is the entry point for decoding a single CTU. It calls coding_quadtree()
/// recursively to split the CTU into CUs, then decodes each CU's prediction
/// and residual syntax.
pub fn decode_ctu(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
    sps: &Sps,
    _pps: &Pps,
    ctu_x: u32,
    ctu_y: u32,
) -> Result<CtuSyntax, HevcError> {
    let ctu_size = sps.ctu_size();
    let mut cus = Vec::new();

    coding_quadtree(cabac, contexts, sps, ctu_x, ctu_y, ctu_size, 0, &mut cus)?;

    Ok(CtuSyntax {
        x: ctu_x,
        y: ctu_y,
        size: ctu_size,
        cus,
    })
}

/// Recursive quad-tree split (ITU-T H.265 Section 7.3.8.4).
fn coding_quadtree(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
    sps: &Sps,
    x: u32,
    y: u32,
    size: u32,
    depth: u32,
    cus: &mut Vec<CuSyntax>,
) -> Result<(), HevcError> {
    // Check if we should split
    let min_cb_size = sps.min_cb_size();
    let max_depth = sps.log2_diff_max_min_luma_coding_block_size as u32;

    let split = if size > min_cb_size && depth < max_depth {
        // Decode split_cu_flag
        let ctx_idx = get_split_cu_ctx(depth);
        cabac.decode_bin(&mut contexts[ctx_idx])? != 0
    } else {
        false // At max depth or min size, can't split
    };

    if split {
        let half = size / 2;
        coding_quadtree(cabac, contexts, sps, x, y, half, depth + 1, cus)?;
        if x + half < sps.pic_width {
            coding_quadtree(cabac, contexts, sps, x + half, y, half, depth + 1, cus)?;
        }
        if y + half < sps.pic_height {
            coding_quadtree(cabac, contexts, sps, x, y + half, half, depth + 1, cus)?;
        }
        if x + half < sps.pic_width && y + half < sps.pic_height {
            coding_quadtree(
                cabac,
                contexts,
                sps,
                x + half,
                y + half,
                half,
                depth + 1,
                cus,
            )?;
        }
    } else {
        // Leaf: decode coding unit
        let cu = decode_coding_unit(cabac, contexts, sps, x, y, size, depth)?;
        cus.push(cu);
    }

    Ok(())
}

/// Decode a coding unit (ITU-T H.265 Section 7.3.8.5).
fn decode_coding_unit(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
    sps: &Sps,
    x: u32,
    y: u32,
    size: u32,
    _depth: u32,
) -> Result<CuSyntax, HevcError> {
    // For I-slices: pred_mode is always Intra, no need to decode pred_mode_flag
    let pred_mode = PredMode::Intra;

    // Part mode: 2Nx2N or NxN (NxN only at min CU size)
    let part_mode = if size == sps.min_cb_size() {
        // Decode part_mode — for intra at min size, 0=2Nx2N, 1=NxN
        let ctx_idx = PART_MODE_CTX_OFFSET;
        let val = cabac.decode_bin(&mut contexts[ctx_idx])?;
        if val == 1 {
            PartMode::Part2Nx2N
        } else {
            PartMode::PartNxN
        }
    } else {
        PartMode::Part2Nx2N
    };

    // Decode intra prediction modes
    let num_pus = match part_mode {
        PartMode::Part2Nx2N => 1,
        PartMode::PartNxN => 4,
    };

    let mut prev_intra_luma_pred_flags = Vec::with_capacity(num_pus);
    for _ in 0..num_pus {
        let ctx_idx = PREV_INTRA_PRED_CTX_OFFSET;
        let flag = cabac.decode_bin(&mut contexts[ctx_idx])?;
        prev_intra_luma_pred_flags.push(flag != 0);
    }

    let mut intra_luma_modes = Vec::with_capacity(num_pus);
    for i in 0..num_pus {
        if prev_intra_luma_pred_flags[i] {
            // mpm_idx: 0, 1, or 2 (truncated unary, bypass)
            let mpm_idx = if cabac.decode_bypass()? == 0 {
                0
            } else if cabac.decode_bypass()? == 0 {
                1
            } else {
                2
            };
            intra_luma_modes.push(mpm_idx as u8); // Placeholder — actual MPM resolution happens in frame assembly
        } else {
            // rem_intra_luma_pred_mode: 5 bypass bits
            let mut rem = 0u8;
            for bit in 0..5 {
                rem |= (cabac.decode_bypass()? as u8) << (4 - bit);
            }
            // Signal that this is a rem mode by adding 32 (modes 0-31 encode rem, 0-2 encode mpm)
            intra_luma_modes.push(rem + 3); // Offset to distinguish from mpm_idx (0-2)
        }
    }

    // Intra chroma pred mode
    let ctx_idx = CHROMA_PRED_CTX_OFFSET;
    let chroma_flag = cabac.decode_bin(&mut contexts[ctx_idx])?;
    let intra_chroma_mode = if chroma_flag == 0 {
        4 // DM mode (derived from luma)
    } else {
        let b0 = cabac.decode_bypass()?;
        let b1 = cabac.decode_bypass()?;
        (b0 * 2 + b1) as u8
    };

    // rqt_root_cbf — for intra CUs, signals whether any residual exists
    // Per HEVC spec, rqt_root_cbf is NOT signaled for intra CUs — it's implicitly 1.
    // Instead, cbf flags are signaled per-TU in the transform tree.
    // However, for the simplest case (flat image), all cbf will be 0.
    let tu = decode_transform_tree(cabac, contexts, sps, x, y, size, 0, true, true)?;

    Ok(CuSyntax {
        x,
        y,
        size,
        pred_mode,
        part_mode,
        intra_luma_modes,
        intra_chroma_mode,
        tu: Some(tu),
    })
}

/// Decode transform tree recursion (ITU-T H.265 Section 7.3.8.7).
fn decode_transform_tree(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
    sps: &Sps,
    x: u32,
    y: u32,
    size: u32,
    depth: u32,
    _parent_cbf_cb: bool,
    _parent_cbf_cr: bool,
) -> Result<TuSyntax, HevcError> {
    let min_tu_size = 1u32 << sps.log2_min_luma_transform_block_size;
    let max_tu_depth = sps.max_transform_hierarchy_depth_intra as u32;

    // Decide whether to split the TU
    let split = if size > min_tu_size && depth < max_tu_depth {
        let ctx_idx = SPLIT_TU_CTX_OFFSET + 5 - (size as usize).trailing_zeros() as usize;
        cabac.decode_bin(&mut contexts[ctx_idx.min(contexts.len() - 1)])? != 0
    } else {
        false
    };

    if split {
        let half = size / 2;
        let mut children = Vec::with_capacity(4);
        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let child = decode_transform_tree(
                cabac,
                contexts,
                sps,
                x + dx * half,
                y + dy * half,
                half,
                depth + 1,
                true,
                true,
            )?;
            children.push(child);
        }
        Ok(TuSyntax {
            x,
            y,
            size,
            cbf_luma: children.iter().any(|c| c.cbf_luma),
            cbf_cb: children.iter().any(|c| c.cbf_cb),
            cbf_cr: children.iter().any(|c| c.cbf_cr),
            luma_coeffs: Vec::new(),
            children,
        })
    } else {
        // Leaf TU: decode CBF flags and residual coefficients
        // HEVC Section 7.3.8.7: cbf_cb and cbf_cr decoded before cbf_luma

        // Chroma CBF (4:2:0: decoded when chroma is present and depth allows)
        let cbf_cb = if sps.chroma_format_idc > 0 && _parent_cbf_cb {
            let ctx_idx = CBF_CHROMA_CTX_OFFSET + depth.min(3) as usize;
            if ctx_idx < contexts.len() {
                cabac.decode_bin(&mut contexts[ctx_idx])? != 0
            } else {
                false
            }
        } else {
            false
        };

        let cbf_cr = if sps.chroma_format_idc > 0 && _parent_cbf_cr {
            let ctx_idx = CBF_CHROMA_CTX_OFFSET + depth.min(3) as usize;
            if ctx_idx < contexts.len() {
                cabac.decode_bin(&mut contexts[ctx_idx])? != 0
            } else {
                false
            }
        } else {
            false
        };

        // cbf_luma: always decoded for leaf TU
        let cbf_luma_ctx = CBF_LUMA_CTX_OFFSET + if depth > 0 { 1 } else { 0 };
        let cbf_luma = if cbf_luma_ctx < contexts.len() {
            cabac.decode_bin(&mut contexts[cbf_luma_ctx])? != 0
        } else {
            false
        };

        let luma_coeffs = if cbf_luma {
            decode_residual_coeffs(cabac, contexts, size)?
        } else {
            Vec::new()
        };

        Ok(TuSyntax {
            x,
            y,
            size,
            cbf_luma,
            cbf_cb,
            cbf_cr,
            luma_coeffs,
            children: Vec::new(),
        })
    }
}

/// Decode residual coefficients for a transform block.
///
/// ITU-T H.265 Section 7.3.8.11.
/// This is a simplified version that decodes the coefficient levels
/// using CABAC significance map and level coding.
fn decode_residual_coeffs(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
    size: u32,
) -> Result<Vec<i16>, HevcError> {
    let num_coeffs = (size * size) as usize;
    let mut coeffs = vec![0i16; num_coeffs];

    let log2_size = (size as f32).log2() as usize;

    // Decode last significant coefficient position
    let (last_x, last_y) = decode_last_sig_coeff_pos(cabac, contexts, log2_size)?;

    if last_x == 0 && last_y == 0 {
        // Only DC coefficient
        let level = decode_coeff_level(cabac, contexts)?;
        let sign = cabac.decode_bypass()?;
        coeffs[0] = if sign != 0 { -level } else { level };
        return Ok(coeffs);
    }

    // Scan backwards from last significant coefficient
    // Using diagonal scan order (simplified)
    let scan = build_scan_order(size as usize);

    // Find the scan position of the last significant coefficient
    let last_scan_pos = scan
        .iter()
        .position(|&(x, y)| x == last_x as usize && y == last_y as usize)
        .unwrap_or(0);

    // Decode significance flags and levels for each position
    for scan_idx in (0..=last_scan_pos).rev() {
        let (sx, sy) = scan[scan_idx];
        let pos = sy * size as usize + sx;

        if scan_idx == last_scan_pos {
            // Last position is always significant
            let level = decode_coeff_level(cabac, contexts)?;
            let sign = cabac.decode_bypass()?;
            coeffs[pos] = if sign != 0 { -level } else { level };
        } else {
            // Decode significance flag
            let sig_ctx = SIG_COEFF_CTX_OFFSET;
            let sig = cabac.decode_bin(&mut contexts[sig_ctx])? != 0;
            if sig {
                let level = decode_coeff_level(cabac, contexts)?;
                let sign = cabac.decode_bypass()?;
                coeffs[pos] = if sign != 0 { -level } else { level };
            }
        }
    }

    Ok(coeffs)
}

/// Decode the last significant coefficient X and Y prefixes.
fn decode_last_sig_coeff_pos(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
    log2_size: usize,
) -> Result<(u32, u32), HevcError> {
    let max_prefix = (log2_size << 1) - 1;

    // X prefix (truncated unary)
    let mut last_x_prefix = 0u32;
    let ctx_base_x = LAST_SIG_X_CTX_OFFSET;
    for i in 0..max_prefix {
        let ctx_idx = ctx_base_x + i;
        if ctx_idx >= contexts.len() {
            break;
        }
        let val = cabac.decode_bin(&mut contexts[ctx_idx])?;
        if val == 0 {
            break;
        }
        last_x_prefix += 1;
    }

    // Y prefix (truncated unary)
    let mut last_y_prefix = 0u32;
    let ctx_base_y = LAST_SIG_Y_CTX_OFFSET;
    for i in 0..max_prefix {
        let ctx_idx = ctx_base_y + i;
        if ctx_idx >= contexts.len() {
            break;
        }
        let val = cabac.decode_bin(&mut contexts[ctx_idx])?;
        if val == 0 {
            break;
        }
        last_y_prefix += 1;
    }

    // Decode suffixes (bypass) if prefix > 3
    let last_x = if last_x_prefix > 3 {
        let suffix_len = (last_x_prefix - 2) >> 1;
        let mut suffix = 0u32;
        for _ in 0..suffix_len {
            suffix = (suffix << 1) | cabac.decode_bypass()?;
        }
        let base = ((2 + (last_x_prefix & 1)) << suffix_len) - 2;
        base + suffix
    } else {
        last_x_prefix
    };

    let last_y = if last_y_prefix > 3 {
        let suffix_len = (last_y_prefix - 2) >> 1;
        let mut suffix = 0u32;
        for _ in 0..suffix_len {
            suffix = (suffix << 1) | cabac.decode_bypass()?;
        }
        let base = ((2 + (last_y_prefix & 1)) << suffix_len) - 2;
        base + suffix
    } else {
        last_y_prefix
    };

    Ok((last_x, last_y))
}

/// Decode a single coefficient absolute level.
fn decode_coeff_level(
    cabac: &mut CabacDecoder,
    contexts: &mut [ContextModel],
) -> Result<i16, HevcError> {
    // coeff_abs_level_greater1_flag
    let gt1_ctx = GT1_CTX_OFFSET;
    let greater1 = cabac.decode_bin(&mut contexts[gt1_ctx])? != 0;

    if !greater1 {
        return Ok(1); // Level = 1
    }

    // coeff_abs_level_greater2_flag
    let gt2_ctx = GT2_CTX_OFFSET;
    let greater2 = cabac.decode_bin(&mut contexts[gt2_ctx])? != 0;

    if !greater2 {
        return Ok(2); // Level = 2
    }

    // coeff_abs_level_remaining (bypass, Exp-Golomb with Rice parameter)
    let rice_param = 0u32; // Simplified — real decoder adapts this
    let remaining = decode_coeff_abs_level_remaining(cabac, rice_param)?;

    Ok((3 + remaining) as i16)
}

/// Decode coeff_abs_level_remaining using truncated Rice + Exp-Golomb.
fn decode_coeff_abs_level_remaining(
    cabac: &mut CabacDecoder,
    rice_param: u32,
) -> Result<u32, HevcError> {
    // Prefix: unary code in bypass mode
    let mut prefix = 0u32;
    loop {
        let val = cabac.decode_bypass()?;
        if val == 0 {
            break;
        }
        prefix += 1;
        if prefix >= 20 {
            break;
        } // Safety limit
    }

    if prefix < 3 {
        // Truncated Rice
        let mut suffix = 0u32;
        for _ in 0..rice_param {
            suffix = (suffix << 1) | cabac.decode_bypass()?;
        }
        Ok((prefix << rice_param) + suffix)
    } else {
        // Exp-Golomb with offset
        let suffix_len = prefix - 3 + rice_param;
        let mut suffix = 0u32;
        for _ in 0..suffix_len {
            suffix = (suffix << 1) | cabac.decode_bypass()?;
        }
        Ok(((2 + prefix - 3) << rice_param) + suffix)
    }
}

/// Build a diagonal scan order for an NxN block.
fn build_scan_order(n: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(n * n);
    for diag in 0..2 * n - 1 {
        if diag % 2 == 0 {
            // Even diagonal: bottom-left to top-right
            let start_y = diag.min(n - 1);
            let start_x = diag - start_y;
            let mut x = start_x;
            let mut y = start_y;
            loop {
                order.push((x, y));
                if x + 1 >= n || y == 0 {
                    break;
                }
                x += 1;
                y -= 1;
            }
        } else {
            // Odd diagonal: top-right to bottom-left
            let start_x = diag.min(n - 1);
            let start_y = diag - start_x;
            let mut x = start_x;
            let mut y = start_y;
            loop {
                order.push((x, y));
                if y + 1 >= n || x == 0 {
                    break;
                }
                x -= 1;
                y += 1;
            }
        }
    }
    order
}

// ─── Context index offsets ──────────────────────────────────────────────────
//
// These are simplified context index mappings. A full implementation would
// use the complete HEVC context derivation from Tables 9-5 through 9-37.

fn get_split_cu_ctx(depth: u32) -> usize {
    SPLIT_CU_CTX_OFFSET + depth.min(2) as usize
}

const SPLIT_CU_CTX_OFFSET: usize = 0;
const PART_MODE_CTX_OFFSET: usize = 3;
const PREV_INTRA_PRED_CTX_OFFSET: usize = 4;
const CHROMA_PRED_CTX_OFFSET: usize = 5;
const SPLIT_TU_CTX_OFFSET: usize = 6;
const CBF_LUMA_CTX_OFFSET: usize = 10;
const SIG_COEFF_CTX_OFFSET: usize = 12;
const GT1_CTX_OFFSET: usize = 13;
const GT2_CTX_OFFSET: usize = 14;
const LAST_SIG_X_CTX_OFFSET: usize = 15;
const LAST_SIG_Y_CTX_OFFSET: usize = 33;
const CBF_CHROMA_CTX_OFFSET: usize = 51;

/// Total number of context models needed for syntax parsing.
pub const NUM_SYNTAX_CONTEXTS: usize = 55; // extended for chroma CBF

/// Initialize syntax context models for an I-slice at the given QP.
///
/// Uses the CABAC module's init tables from the HEVC spec (Tables 9-5 through 9-37).
pub fn init_syntax_contexts(qp: i32) -> Vec<ContextModel> {
    use crate::cabac;

    // I-slice init table index = 2
    let si = 2usize;

    // Build flat init value array matching our context layout
    let mut iv = vec![154u8; NUM_SYNTAX_CONTEXTS]; // CNU default

    // SPLIT_CU_FLAG: 3 contexts
    for (i, &v) in cabac::SPLIT_CU_FLAG_INIT[si].iter().enumerate() {
        iv[SPLIT_CU_CTX_OFFSET + i] = v;
    }
    // PART_MODE: 1 context
    iv[PART_MODE_CTX_OFFSET] = cabac::PART_MODE_INIT[si][0];
    // PREV_INTRA_LUMA_PRED_FLAG: 1 context
    iv[PREV_INTRA_PRED_CTX_OFFSET] = cabac::PREV_INTRA_LUMA_PRED_FLAG_INIT[si][0];
    // INTRA_CHROMA_PRED_MODE: 1 context
    iv[CHROMA_PRED_CTX_OFFSET] = cabac::INTRA_CHROMA_PRED_MODE_INIT[si][0];
    // SPLIT_TRANSFORM_FLAG: 3 contexts
    for (i, &v) in cabac::SPLIT_TRANSFORM_FLAG_INIT[si].iter().enumerate() {
        if SPLIT_TU_CTX_OFFSET + i < iv.len() {
            iv[SPLIT_TU_CTX_OFFSET + i] = v;
        }
    }
    // CBF_LUMA: 2 contexts
    for (i, &v) in cabac::CBF_LUMA_INIT[si].iter().enumerate() {
        if CBF_LUMA_CTX_OFFSET + i < iv.len() {
            iv[CBF_LUMA_CTX_OFFSET + i] = v;
        }
    }
    // SIG_COEFF_FLAG, GT1, GT2, LAST_SIG_X/Y: use spec init values for I-slice
    // These use the CODED_SUB_BLOCK_FLAG, SIG_COEFF_FLAG, etc. tables
    for (i, &v) in cabac::SIG_COEFF_FLAG_INIT[si].iter().enumerate() {
        if SIG_COEFF_CTX_OFFSET + i < iv.len() {
            iv[SIG_COEFF_CTX_OFFSET + i] = v;
        }
    }
    for (i, &v) in cabac::COEFF_ABS_LEVEL_GREATER1_FLAG_INIT[si]
        .iter()
        .enumerate()
    {
        if GT1_CTX_OFFSET + i < iv.len() {
            iv[GT1_CTX_OFFSET + i] = v;
        }
    }
    for (i, &v) in cabac::COEFF_ABS_LEVEL_GREATER2_FLAG_INIT[si]
        .iter()
        .enumerate()
    {
        if GT2_CTX_OFFSET + i < iv.len() {
            iv[GT2_CTX_OFFSET + i] = v;
        }
    }
    for (i, &v) in cabac::LAST_SIG_COEFF_X_PREFIX_INIT[si].iter().enumerate() {
        if LAST_SIG_X_CTX_OFFSET + i < iv.len() {
            iv[LAST_SIG_X_CTX_OFFSET + i] = v;
        }
    }
    for (i, &v) in cabac::LAST_SIG_COEFF_Y_PREFIX_INIT[si].iter().enumerate() {
        if LAST_SIG_Y_CTX_OFFSET + i < iv.len() {
            iv[LAST_SIG_Y_CTX_OFFSET + i] = v;
        }
    }
    // CBF_CHROMA: 5 contexts
    for (i, &v) in cabac::CBF_CHROMA_INIT[si].iter().enumerate() {
        if CBF_CHROMA_CTX_OFFSET + i < iv.len() {
            iv[CBF_CHROMA_CTX_OFFSET + i] = v;
        }
    }

    cabac::init_contexts(&iv, qp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_order_4x4() {
        let scan = build_scan_order(4);
        assert_eq!(scan.len(), 16);
        // First position should be (0,0)
        assert_eq!(scan[0], (0, 0));
        // Last position should be (3,3)
        assert_eq!(scan[15], (3, 3));
    }

    #[test]
    fn scan_order_covers_all() {
        for n in [2, 4, 8] {
            let scan = build_scan_order(n);
            assert_eq!(scan.len(), n * n, "scan order for {n}x{n}");
            // Verify all positions are covered
            let mut covered = vec![false; n * n];
            for &(x, y) in &scan {
                covered[y * n + x] = true;
            }
            for (i, &c) in covered.iter().enumerate() {
                assert!(
                    c,
                    "position ({}, {}) not covered in {n}x{n} scan",
                    i % n,
                    i / n
                );
            }
        }
    }

    #[test]
    fn init_syntax_contexts_doesnt_panic() {
        for qp in [0, 22, 37, 51] {
            let ctx = init_syntax_contexts(qp);
            assert_eq!(ctx.len(), NUM_SYNTAX_CONTEXTS);
        }
    }

    #[test]
    fn pred_mode_enum() {
        assert_eq!(PredMode::Intra, PredMode::Intra);
        assert_ne!(PredMode::Intra, PredMode::Inter);
    }

    #[test]
    fn part_mode_enum() {
        assert_eq!(PartMode::Part2Nx2N, PartMode::Part2Nx2N);
        assert_ne!(PartMode::Part2Nx2N, PartMode::PartNxN);
    }

    #[test]
    fn golden_syntax_parse() {
        crate::skip_if_no_fixtures!();

        // Load a real HEVC bitstream and verify we can at least parse the NAL structure
        let hevc_data = crate::testutil::load_fixture("flat_64x64_q22", "hevc").unwrap();
        let nals: Vec<&[u8]> = crate::nal::NalIterator::new(&hevc_data).collect();

        // Should have at least VPS + SPS + PPS + slice
        assert!(nals.len() >= 4, "expected >= 4 NALs, got {}", nals.len());

        // Find and parse SPS to get dimensions
        for nal_data in &nals {
            let nal = crate::nal::parse_nal_unit(nal_data).unwrap();
            if nal.nal_type == crate::types::NalUnitType::SpsNut {
                let sps = crate::params::parse_sps(&nal.rbsp).unwrap();
                assert_eq!(sps.pic_width, 64);
                assert_eq!(sps.pic_height, 64);
                // x265 ultrafast uses 32x32 CTU, so 64x64 frame = 4 CTUs
                assert!(
                    sps.ctu_size() >= 16,
                    "CTU size should be >= 16, got {}",
                    sps.ctu_size()
                );
                return;
            }
        }
        panic!("SPS not found");
    }
}
