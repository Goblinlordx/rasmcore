//! VP8 bitstream assembly — frame header, partitions, macroblock encoding.
//!
//! Assembles the complete VP8 frame from encoded macroblocks.
//! References: RFC 6386 Sections 4, 9, 13, 19.

#![allow(clippy::too_many_arguments)]

use crate::block::{self, MacroblockInfo};
use crate::boolcoder::BoolWriter;
use crate::dct;
use crate::filter;
use crate::predict;
use crate::quant::{self, SegmentQuant};
use crate::ratecontrol::EncodeParams;
use crate::rdo;
use crate::segment::{self, SegmentMap};
use crate::tables;
use crate::token;
use rasmcore_color::YuvImage;

/// Encode a VP8 key frame from YUV420 data.
///
/// Returns the raw VP8 frame data (without RIFF container).
pub fn encode_frame(yuv: &YuvImage, params: &EncodeParams) -> Vec<u8> {
    let width = yuv.width;
    let height = yuv.height;
    let (mb_w, mb_h) = block::mb_dimensions(width, height);

    // Pad YUV planes to macroblock boundaries
    let padded_w = mb_w as usize * 16;
    let padded_h = mb_h as usize * 16;
    let y_padded = pad_plane(&yuv.y, width as usize, height as usize, padded_w, padded_h);
    let uv_pad_w = mb_w as usize * 8;
    let uv_pad_h = mb_h as usize * 8;
    let uv_w = (width as usize).div_ceil(2);
    let uv_h = (height as usize).div_ceil(2);
    let u_padded = pad_plane(&yuv.u, uv_w, uv_h, uv_pad_w, uv_pad_h);
    let v_padded = pad_plane(&yuv.v, uv_w, uv_h, uv_pad_w, uv_pad_h);

    // Compute per-MB segment map (adaptive QP via activity analysis)
    let seg_map = segment::compute_segment_map(
        &y_padded,
        padded_w,
        mb_w as usize,
        mb_h as usize,
        params.qp_y,
    );

    // Encode all macroblocks — collect modes and token data
    let mut mb_infos = Vec::with_capacity((mb_w * mb_h) as usize);
    let mut token_writer = BoolWriter::with_capacity(padded_w * padded_h);

    // Reconstructed planes for prediction reference
    let mut recon_y = vec![128u8; padded_w * padded_h];
    let mut recon_u = vec![128u8; uv_pad_w * uv_pad_h];
    let mut recon_v = vec![128u8; uv_pad_w * uv_pad_h];

    // Complexity context tracking (matches decoder's top[]/left[] arrays)
    // 9 entries per MB: [0]=Y2, [1..5]=Y cols, [5..7]=U cols, [7..9]=V cols
    let mut top_ctx: Vec<[u8; 9]> = vec![[0u8; 9]; mb_w as usize];
    let mut left_ctx: [u8; 9] = [0u8; 9];

    for mb_row in 0..mb_h as usize {
        left_ctx = [0u8; 9]; // reset left context at start of each row
        for mb_col in 0..mb_w as usize {
            let seg_id = seg_map.get(mb_col, mb_row);
            let seg_quant = seg_map.quant(seg_id);
            // Derive RDO lambda from segment QP
            let seg_qp = (params.qp_y as i16 + seg_map.qp_deltas[seg_id as usize] as i16)
                .clamp(0, 127) as u8;
            let lambda = crate::rdo::lambda_from_qp(seg_qp);

            let mut mb = encode_macroblock(
                &y_padded,
                &u_padded,
                &v_padded,
                &recon_y,
                &recon_u,
                &recon_v,
                padded_w,
                uv_pad_w,
                mb_col,
                mb_row,
                seg_quant,
                lambda,
                &mut token_writer,
                &mut top_ctx[mb_col],
                &mut left_ctx,
            );
            mb.segment = seg_id;
            mb_infos.push(mb);

            // Update reconstructed planes (for next macroblock's prediction)
            reconstruct_macroblock(
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                padded_w,
                uv_pad_w,
                mb_col,
                mb_row,
                &y_padded,
                &u_padded,
                &v_padded,
                seg_quant,
                mb_infos.last().unwrap(),
            );
        }

        // Apply loop filter to this MB row after all MBs in the row are
        // reconstructed. This matches the decoder's timing: filter runs after
        // each row, before the next row uses these pixels as prediction ref.
        filter::apply_loop_filter_row(
            &mut recon_y,
            padded_w,
            mb_row,
            mb_w as usize,
            params.filter_level,
            params.filter_sharpness,
            params.filter_type,
        );
    }

    let token_data = token_writer.finish();

    // Apply loop filter to reconstructed Y plane
    if params.filter_level > 0 {
        filter::apply_loop_filter(
            &mut recon_y,
            padded_w,
            padded_h,
            params.filter_level,
            params.filter_sharpness,
            params.filter_type,
        );
    }
    // Build first partition (macroblock modes + segmentation header)
    let first_partition = encode_first_partition(&mb_infos, mb_w, mb_h, params, &seg_map);

    // Assemble frame
    assemble_frame(width, height, &first_partition, &token_data)
}

/// Assemble the VP8 frame from partitions.
fn assemble_frame(
    width: u32,
    height: u32,
    first_partition: &[u8],
    token_partition: &[u8],
) -> Vec<u8> {
    let first_part_size = first_partition.len() as u32;

    // Frame tag: 3 bytes (RFC 6386 Section 9.1)
    // Bits 0: frame_type (0 = key frame)
    // Bits 1-2: version (0)
    // Bit 3: show_frame (1)
    // Bits 4-23: first_part_size (19 bits)
    let frame_tag = (first_part_size << 5) | (1 << 4); // keyframe, version 0, show=1
    let tag_bytes = frame_tag.to_le_bytes();

    // Key frame header: 7 bytes (RFC 6386 Section 9.1)
    // Bytes 0-2: start code 0x9D 0x01 0x2A
    // Bytes 3-4: width (14 bits) + horizontal scale (2 bits)
    // Bytes 5-6: height (14 bits) + vertical scale (2 bits)
    let width_field = width as u16 & 0x3FFF; // no scaling
    let height_field = height as u16 & 0x3FFF;

    let mut frame = Vec::with_capacity(10 + first_partition.len() + token_partition.len());
    // Frame tag (3 bytes)
    frame.extend_from_slice(&tag_bytes[..3]);
    // Start code
    frame.extend_from_slice(&[0x9D, 0x01, 0x2A]);
    // Width + height (little-endian)
    frame.extend_from_slice(&width_field.to_le_bytes());
    frame.extend_from_slice(&height_field.to_le_bytes());
    // First partition
    frame.extend_from_slice(first_partition);
    // Token partition(s)
    frame.extend_from_slice(token_partition);

    frame
}

/// Encode the first partition: segment header, loop filter, quant params, MB modes.
fn encode_first_partition(
    mb_infos: &[MacroblockInfo],
    _mb_w: u32,
    _mb_h: u32,
    params: &EncodeParams,
    seg_map: &SegmentMap,
) -> Vec<u8> {
    let mut bw = BoolWriter::with_capacity(1024);

    // Color space and clamping (RFC 6386 Section 9.2)
    bw.put_bit(128, false); // color_space = 0 (YUV)
    bw.put_bit(128, false); // clamping_type = 0

    // Segmentation (RFC 6386 Section 9.3)
    if seg_map.enabled {
        bw.put_bit(128, true); // segmentation_enabled = true

        // segment_update_map = true (we're sending the full segment map)
        bw.put_bit(128, true);
        // segment_update_data = true (we're sending QP deltas)
        bw.put_bit(128, true);

        // segment_feature_mode: 0 = delta mode (deltas from base QP)
        bw.put_bit(128, false);

        // Per-segment quantizer deltas (4 segments)
        for i in 0..segment::NUM_SEGMENTS {
            let delta = seg_map.qp_deltas[i];
            if delta != 0 {
                bw.put_bit(128, true); // quantizer_update = true
                bw.put_literal(7, delta.unsigned_abs() as u32);
                bw.put_bit(128, delta < 0); // sign bit: true = negative
            } else {
                bw.put_bit(128, false); // quantizer_update = false
            }
        }

        // Per-segment loop filter deltas (4 segments) — all zero
        for _ in 0..segment::NUM_SEGMENTS {
            bw.put_bit(128, false); // lf_update = false
        }

        // Segment map probabilities (3 tree probs for 4-way classification)
        // Use uniform probability (128) for all tree nodes
        for _ in 0..3 {
            bw.put_bit(128, true); // prob_present = true
            bw.put_literal(8, 128); // uniform probability
        }
    } else {
        bw.put_bit(128, false); // segmentation_enabled = false
    }

    // Loop filter (RFC 6386 Section 9.4)
    bw.put_bit(128, params.filter_type == crate::filter::FilterType::Simple); // filter_type: false=normal, true=simple
    bw.put_literal(6, params.filter_level as u32);
    bw.put_literal(3, params.filter_sharpness as u32);

    // Loop filter adjustments — disabled
    bw.put_bit(128, false); // mode_ref_lf_delta_enabled = false

    // Token partition count (RFC 6386 Section 9.5)
    // log2(num_partitions) in 2 bits. 0 = 1 partition.
    bw.put_literal(2, 0); // 1 token partition

    // Quantizer (RFC 6386 Section 9.6)
    bw.put_literal(7, params.qp_y as u32); // y_ac_qi
    // All deltas = 0 (signaled by false flags)
    for _ in 0..5 {
        bw.put_bit(128, false); // no delta
    }

    // Refresh entropy probs (RFC 6386 Section 9.7)
    bw.put_bit(128, false); // refresh_entropy_probs = false

    // Refresh last frame buffer
    // For key frames, refresh_last is implicit (always true)

    // Token probability updates (RFC 6386 Section 13.4)
    // Write false for each flag using the proper update probabilities.
    // High probability values (near 255) mean "update is very unlikely",
    // which lets the bool coder compress "false" flags into very few bits.
    for &prob in &token::COEFF_UPDATE_PROBS {
        bw.put_bit(prob, false);
    }

    // mb_no_coeff_skip (RFC 6386 Section 9.10, verified against libvpx)
    let skip_count = mb_infos.iter().filter(|m| m.skip).count();
    let total_mbs = mb_infos.len().max(1);
    let enable_skip = skip_count > 0;
    let prob_skip = if enable_skip {
        ((total_mbs - skip_count) * 256 / total_mbs).clamp(1, 255) as u8
    } else {
        128
    };
    if enable_skip {
        bw.put_literal(1, 1);
        bw.put_literal(8, prob_skip as u32);
    } else {
        bw.put_literal(1, 0);
    }

    // Encode macroblock modes with B_PRED context tracking.
    // For B_PRED MBs, we need context from neighboring MBs' sub-block modes.
    let mb_w = mb_infos.last().map(|m| m.mb_x as usize + 1).unwrap_or(0);
    // Track bottom row modes of each MB column (for top context of next row)
    let mut top_bmode_ctx: Vec<[u8; 4]> = vec![[0u8; 4]; mb_w];
    let mut left_bmode_ctx: [u8; 4] = [0u8; 4];

    for mb in mb_infos {
        if mb.mb_x == 0 {
            left_bmode_ctx = [0u8; 4]; // Reset at row start
        }

        // Encode segment ID when segmentation is enabled (RFC 6386 Section 10.2)
        if seg_map.enabled {
            let seg = mb.segment;
            bw.put_bit(128, seg >= 2); // tree node 0: {0,1} vs {2,3}
            if seg >= 2 {
                bw.put_bit(128, seg == 3); // tree node 2
            } else {
                bw.put_bit(128, seg == 1); // tree node 1
            }
        }

        if enable_skip {
            bw.put_bit(prob_skip, mb.skip);
        }
        encode_mb_header(
            &mut bw,
            mb,
            &top_bmode_ctx[mb.mb_x as usize],
            &left_bmode_ctx,
        );

        if mb.y_mode == 4 {
            // Update context: bottom row of this MB's sub-block modes
            for col in 0..4 {
                top_bmode_ctx[mb.mb_x as usize][col] = mb.b_modes[12 + col]; // row 3
            }
            // Right column of this MB's sub-block modes
            for row in 0..4 {
                left_bmode_ctx[row] = mb.b_modes[row * 4 + 3]; // col 3
            }
        } else {
            // I16x16 mode: use the I16x16 mode as context for all positions
            // Per VP8 spec: DC_PRED maps to B_DC_PRED context
            let ctx_mode = match mb.y_mode {
                1 => 2, // V → B_VE
                2 => 3, // H → B_HE
                3 => 1, // TM → B_TM
                _ => 0, // DC → B_DC
            };
            top_bmode_ctx[mb.mb_x as usize] = [ctx_mode; 4];
            left_bmode_ctx = [ctx_mode; 4];
        }
    }

    bw.finish()
}

/// Encode a single macroblock header in the first partition.
fn encode_mb_header(
    bw: &mut BoolWriter,
    mb: &MacroblockInfo,
    above_bpred_modes: &[u8; 4],
    left_bpred_modes: &[u8; 4],
) {
    // Luma prediction mode (RFC 6386 Section 11.2)
    encode_intra_y_mode(bw, mb.y_mode);

    // If B_PRED, encode 16 sub-block modes with context
    if mb.y_mode == 4 {
        encode_bpred_modes(bw, &mb.b_modes, above_bpred_modes, left_bpred_modes);
    }

    // Chroma prediction mode
    encode_intra_uv_mode(bw, mb.uv_mode);
}

/// Encode luma 16x16 prediction mode for key frame.
///
/// Key-frame Y mode tree (vp8_kf_ymode_tree from libvpx):
///   `[-B_PRED, 2, 4, 6, -DC_PRED, -V_PRED, -H_PRED, -TM_PRED]`
///
///   Node 0: B_PRED vs rest         [prob = 145]
///   Node 1: (DC/V) vs (H/TM)      [prob = 156]
///   Node 2: DC vs V                [prob = 163]
///   Node 3: H vs TM               [prob = 128]
///
fn encode_intra_y_mode(bw: &mut BoolWriter, mode: u8) {
    if mode == 4 {
        // B_PRED: node 0 → false (B_PRED is left child)
        bw.put_bit(145, false);
    } else {
        // Not B_PRED: node 0 → true, then tree for I16x16 modes
        bw.put_bit(145, true);
        match mode {
            0 => {
                bw.put_bit(156, false);
                bw.put_bit(163, false);
            }
            1 => {
                bw.put_bit(156, false);
                bw.put_bit(163, true);
            }
            2 => {
                bw.put_bit(156, true);
                bw.put_bit(128, false);
            }
            3 => {
                bw.put_bit(156, true);
                bw.put_bit(128, true);
            }
            _ => unreachable!("invalid y_mode: {mode}"),
        }
    }
}

/// Encode a single 4x4 intra prediction mode using the B_PRED mode tree.
///
/// Uses context-dependent probabilities from KF_BMODE_PROB[top_mode][left_mode].
/// Tree format matches image-webp/libvpx: leaves are <= 0 (mode = -leaf),
/// positive values are jump indices.
fn encode_intra_4x4_mode(bw: &mut BoolWriter, mode: u8, top_mode: u8, left_mode: u8) {
    let probs = &tables::KF_BMODE_PROB[top_mode as usize][left_mode as usize];

    // Exact tree from image-webp vp8.rs KEYFRAME_BPRED_MODE_TREE.
    // Leaves: val <= 0, mode = -val. Internal: val > 0, jump to that index.
    // Pairs: [false_branch, true_branch] at each even index.
    const TREE: [i8; 18] = [
        0, 2, // idx 0: false→DC(0), true→jump[2]
        -1, 4, // idx 2: false→TM(1), true→jump[4]
        -2, 6, // idx 4: false→V(2), true→jump[6]
        8, 12, // idx 6: false→jump[8], true→jump[12]
        -3, 10, // idx 8: false→H(3), true→jump[10]
        -5, -6, // idx 10: false→RD(5), true→VR(6)
        -4, 14, // idx 12: false→LD(4), true→jump[14]
        -7, 16, // idx 14: false→VL(7), true→jump[16]
        -8, -9, // idx 16: false→HD(8), true→HU(9)
    ];

    let target_leaf = -(mode as i8); // leaf encoding: mode = -leaf_value
    let mut idx = 0usize;

    loop {
        let false_val = TREE[idx];
        let true_val = TREE[idx + 1];
        let prob = probs[idx / 2];

        // Check if target is in the FALSE branch (immediate leaf)
        if false_val <= 0 && false_val == target_leaf {
            bw.put_bit(prob, false);
            return;
        }
        // Check if target is in the TRUE branch (immediate leaf)
        if true_val <= 0 && true_val == target_leaf {
            bw.put_bit(prob, true);
            return;
        }

        // Neither is the target directly. Determine which subtree to enter.
        if false_val <= 0 {
            // FALSE is a different leaf — target must be in TRUE subtree
            bw.put_bit(prob, true);
            idx = true_val as usize;
        } else if true_val <= 0 {
            // TRUE is a different leaf — target must be in FALSE subtree
            bw.put_bit(prob, false);
            idx = false_val as usize;
        } else {
            // Both are internal nodes — search to determine which subtree
            if leaf_in_subtree(&TREE, false_val as usize, target_leaf) {
                bw.put_bit(prob, false);
                idx = false_val as usize;
            } else {
                bw.put_bit(prob, true);
                idx = true_val as usize;
            }
        }
    }
}

/// Check if a leaf value exists in a binary tree subtree.
fn leaf_in_subtree(tree: &[i8; 18], root: usize, target: i8) -> bool {
    let false_val = tree[root];
    let true_val = tree[root + 1];
    if (false_val <= 0 && false_val == target) || (true_val <= 0 && true_val == target) {
        return true;
    }
    if false_val > 0 && leaf_in_subtree(tree, false_val as usize, target) {
        return true;
    }
    if true_val > 0 && leaf_in_subtree(tree, true_val as usize, target) {
        return true;
    }
    false
}

/// Encode 16 sub-block modes for a B_PRED macroblock.
///
/// Each mode is encoded using context from the top and left neighbor modes.
/// For the first row, top context comes from the bottom row of the MB above.
/// For the first column, left context comes from the right column of the MB to the left.
fn encode_bpred_modes(
    bw: &mut BoolWriter,
    b_modes: &[u8; 16],
    above_modes: &[u8; 4], // top context: bottom row modes of MB above
    left_modes: &[u8; 4],  // left context: right column modes of MB to the left
) {
    // Track context within the macroblock.
    // top_ctx[col] = mode of the block above in this column
    // left_mode = mode of the block to the left in this row
    let mut top_ctx = *above_modes;

    for row in 0..4 {
        let mut left_mode = left_modes[row];
        for col in 0..4 {
            let sb = row * 4 + col;
            let mode = b_modes[sb];
            encode_intra_4x4_mode(bw, mode, top_ctx[col], left_mode);
            top_ctx[col] = mode;
            left_mode = mode;
        }
    }
}

/// Encode chroma prediction mode for key frame.
fn encode_intra_uv_mode(bw: &mut BoolWriter, mode: u8) {
    // Key frame chroma mode probabilities (RFC 6386 Section 11.3)
    match mode {
        0 => {
            // DC
            bw.put_bit(142, false);
        }
        1 => {
            // V
            bw.put_bit(142, true);
            bw.put_bit(114, false);
        }
        2 => {
            // H
            bw.put_bit(142, true);
            bw.put_bit(114, true);
            bw.put_bit(183, false);
        }
        3 => {
            // TM
            bw.put_bit(142, true);
            bw.put_bit(114, true);
            bw.put_bit(183, true);
        }
        _ => unreachable!("invalid uv_mode: {mode}"),
    }
}

/// Encode a single macroblock: choose prediction, compute residual, DCT, quantize + RDO prune.
fn encode_macroblock(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    recon_y: &[u8],
    recon_u: &[u8],
    recon_v: &[u8],
    y_stride: usize,
    uv_stride: usize,
    mb_col: usize,
    mb_row: usize,
    seg_quant: &SegmentQuant,
    lambda: f64,
    token_bw: &mut BoolWriter,
    top_ctx: &mut [u8; 9],
    left_ctx: &mut [u8; 9],
) -> MacroblockInfo {
    let y_off = mb_row * 16 * y_stride + mb_col * 16;
    let uv_off = mb_row * 8 * uv_stride + mb_col * 8;

    // Extract 16×16 luma block from source
    let mut y_block = [0u8; 256];
    for r in 0..16 {
        y_block[r * 16..r * 16 + 16]
            .copy_from_slice(&y_plane[y_off + r * y_stride..y_off + r * y_stride + 16]);
    }

    // Get prediction neighbors from reconstructed frame
    let (above_y, left_y, above_left_y) = get_y_neighbors(recon_y, y_stride, mb_col, mb_row);

    // ─── I16x16 mode evaluation (RD cost) ────────────────────────────────
    let (i16_mode, _) = predict::select_best_16x16_rd(
        &y_block,
        &above_y,
        &left_y,
        above_left_y,
        mb_row > 0,
        mb_col > 0,
        &seg_quant.y_ac,
        lambda,
    );

    let mut i16_pred = [0u8; 256];
    predict::predict_16x16(
        i16_mode,
        &above_y,
        &left_y,
        above_left_y,
        mb_row > 0,
        mb_col > 0,
        &mut i16_pred,
    );

    // ─── B_PRED mode evaluation with sequential reconstruction ──────────
    // Each sub-block's mode is selected using RD cost, and the reconstruction
    // is written to a scratch buffer so subsequent sub-blocks use correct neighbors.
    let mut b_modes = [0u8; 16];
    let mut bpred_preds = [[0u8; 16]; 16];

    // Copy current recon_y region into scratch buffer for sequential updates
    let mut scratch_y = recon_y.to_vec();

    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;

        let mut src_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                src_4x4[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }

        // Get neighbors from scratch buffer (has reconstructed previous sub-blocks)
        let (above_4, left_4, al, ar) =
            get_4x4_neighbors(&scratch_y, y_stride, mb_col, mb_row, sb_row, sb_col);

        // RD mode selection: try top SATD candidates with full encode trial
        let mode = predict::select_best_4x4_rd(
            &src_4x4,
            &above_4,
            &left_4,
            al,
            &ar,
            &seg_quant.y_dc,
            lambda,
        );
        b_modes[sb] = mode as u8;

        let mut pred = [0u8; 16];
        predict::predict_4x4(mode, &above_4, &left_4, al, &ar, &mut pred);
        bpred_preds[sb] = pred;

        // Reconstruct this sub-block into scratch buffer for next sub-block's neighbors
        let mut coeffs = [0i16; 16];
        dct::forward_dct(&src_4x4, &pred, &mut coeffs);
        let mut quantized = [0i16; 16];
        quant::quantize_block(&coeffs, &seg_quant.y_dc, &mut quantized);
        let mut dequant_tmp = [0i16; 16];
        quant::dequantize_block(&quantized, &seg_quant.y_dc, &mut dequant_tmp);
        let mut recon_block = [0u8; 16];
        dct::inverse_dct(&dequant_tmp, &pred, &mut recon_block);

        for r in 0..4 {
            for c in 0..4 {
                let row = mb_row * 16 + sb_row * 4 + r;
                let col = mb_col * 16 + sb_col * 4 + c;
                scratch_y[row * y_stride + col] = recon_block[r * 4 + c];
            }
        }
    }

    // ─── Mode decision: full RD cost comparison ─────────────────────────
    // Trial-encode both B_PRED and I16x16 paths, compute reconstruction-based
    // RD cost, and pick whichever has lower cost.

    // B_PRED RD: DCT→quant→RDO→bits + dequant→IDCT→SSD
    let mut bpred_ssd: u64 = 0;
    let mut bpred_bits: u32 = rdo::mode_header_cost_16x16(4); // B_PRED MB header
    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;
        let mut src_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                src_4x4[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }
        // Mode header cost for this sub-block's 4x4 mode
        bpred_bits += rdo::mode_header_cost_4x4(b_modes[sb]);

        // Forward DCT + quantize + RDO prune
        let mut coeffs = [0i16; 16];
        dct::forward_dct(&src_4x4, &bpred_preds[sb], &mut coeffs);
        let mut quantized = [0i16; 16];
        quant::quantize_block(&coeffs, &seg_quant.y_dc, &mut quantized);
        rdo::rdo_prune_block(&mut quantized, &coeffs, &seg_quant.y_dc, 3, lambda);

        // Estimate encoding cost (block_type 3 = Y-with-DC for B_PRED)
        bpred_bits += rdo::estimate_block_bits(&quantized, 3);

        // Reconstruct to measure SSD
        let mut dequant = [0i16; 16];
        quant::dequantize_block(&quantized, &seg_quant.y_dc, &mut dequant);
        let mut recon_block = [0u8; 16];
        dct::inverse_dct(&dequant, &bpred_preds[sb], &mut recon_block);

        for i in 0..16 {
            let d = src_4x4[i] as i32 - recon_block[i] as i32;
            bpred_ssd += (d * d) as u64;
        }
    }
    let bpred_rd = rdo::rd_cost(bpred_ssd, bpred_bits, lambda);

    // I16x16 RD: same trial encode approach for fair comparison
    let mut i16_ssd: u64 = 0;
    let mut i16_bits: u32 = rdo::mode_header_cost_16x16(i16_mode as u8);
    let mut i16_trial_ac = [[0i16; 16]; 16];
    let mut i16_dc_coeffs = [0i16; 16];
    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;
        let mut src_4x4 = [0u8; 16];
        let mut ref_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                src_4x4[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                ref_4x4[r * 4 + c] = i16_pred[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }
        let mut coeffs = [0i16; 16];
        dct::forward_dct(&src_4x4, &ref_4x4, &mut coeffs);
        i16_dc_coeffs[sb] = coeffs[0];
        let mut quantized = [0i16; 16];
        quant::quantize_block(&coeffs, &seg_quant.y_ac, &mut quantized);
        rdo::rdo_prune_block(&mut quantized, &coeffs, &seg_quant.y_ac, 0, lambda);
        quantized[0] = 0; // DC goes to Y2
        i16_trial_ac[sb] = quantized;
        i16_bits += rdo::estimate_block_bits(&quantized, 0);
    }
    // Y2 block (WHT of DC coefficients)
    let mut y2_trial = [0i16; 16];
    dct::forward_wht(&i16_dc_coeffs, &mut y2_trial);
    let mut y2_trial_q = [0i16; 16];
    quant::quantize_block(&y2_trial, &seg_quant.y2_dc, &mut y2_trial_q);
    rdo::rdo_prune_block(&mut y2_trial_q, &y2_trial, &seg_quant.y2_dc, 1, lambda);
    i16_bits += rdo::estimate_block_bits(&y2_trial_q, 1);

    // Reconstruct I16x16 to measure SSD
    let mut y2_dequant_trial = [0i16; 16];
    quant::dequantize_block(&y2_trial_q, &seg_quant.y2_dc, &mut y2_dequant_trial);
    let mut recon_dc_trial = [0i16; 16];
    dct::inverse_wht(&y2_dequant_trial, &mut recon_dc_trial);
    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;
        let mut dequant = [0i16; 16];
        quant::dequantize_block(&i16_trial_ac[sb], &seg_quant.y_ac, &mut dequant);
        dequant[0] = recon_dc_trial[sb];
        let mut ref_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                ref_4x4[r * 4 + c] = i16_pred[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }
        let mut recon_block = [0u8; 16];
        dct::inverse_dct(&dequant, &ref_4x4, &mut recon_block);
        for r in 0..4 {
            for c in 0..4 {
                let orig = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c] as i32;
                let rec = recon_block[r * 4 + c] as i32;
                i16_ssd += ((orig - rec) * (orig - rec)) as u64;
            }
        }
    }
    let i16_rd = rdo::rd_cost(i16_ssd, i16_bits, lambda);

    // libwebp-style mode decision: B_PRED must beat I16x16.
    // In libwebp, PickBestIntra4 initializes rd_best with H=211 (VP8BitCost(0,145))
    // and compares cumulative I4 cost against the I16 score. The i4_penalty
    // (1000 * q_i4^2) is implicitly included via the mode_costs.
    let use_bpred = bpred_rd < i16_rd;
    let final_y_mode: u8 = if use_bpred { 4 } else { i16_mode as u8 };

    // ─── Coefficient encoding ───────────────────────────────────────────
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut y2_quantized = [0i16; 16];

    if use_bpred {
        // B_PRED: each block encodes all 16 coefficients (DC at pos 0, no Y2)
        for sb in 0..16 {
            let sb_row = sb / 4;
            let sb_col = sb % 4;
            let mut src_block = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_block[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_block, &bpred_preds[sb], &mut coeffs);
            // B_PRED: DC at pos 0 uses y_dc quantizer (DC_TABLE), AC uses AC_TABLE
            quant::quantize_block(&coeffs, &seg_quant.y_dc, &mut y_coeffs[sb]);
            // RDO: prune costly coefficients (block_type 3 = Y-with-DC for B_PRED)
            crate::rdo::rdo_prune_block(&mut y_coeffs[sb], &coeffs, &seg_quant.y_dc, 3, lambda);
        }
        // No Y2 block for B_PRED
    } else {
        // I16x16: DC goes to Y2 block
        let mut y_dc_coeffs = [0i16; 16];
        for sb in 0..16 {
            let sb_row = sb / 4;
            let sb_col = sb % 4;
            let mut src_block = [0u8; 16];
            let mut ref_block = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_block[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                    ref_block[r * 4 + c] = i16_pred[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_block, &ref_block, &mut coeffs);
            y_dc_coeffs[sb] = coeffs[0];
            let mut quantized = [0i16; 16];
            quant::quantize_block(&coeffs, &seg_quant.y_ac, &mut quantized);
            // RDO: prune costly AC coefficients (block_type 0 = Y-AC after Y2)
            crate::rdo::rdo_prune_block(&mut quantized, &coeffs, &seg_quant.y_ac, 0, lambda);
            quantized[0] = 0; // DC goes to Y2
            y_coeffs[sb] = quantized;
        }
        let mut y2_coeffs = [0i16; 16];
        dct::forward_wht(&y_dc_coeffs, &mut y2_coeffs);
        quant::quantize_block(&y2_coeffs, &seg_quant.y2_dc, &mut y2_quantized);
        // RDO: prune costly Y2 coefficients (block_type 1 = Y2)
        crate::rdo::rdo_prune_block(&mut y2_quantized, &y2_coeffs, &seg_quant.y2_dc, 1, lambda);
    }

    // ─── Chroma (same for both modes) ───────────────────────────────────
    let (above_u, left_u, above_left_u) = get_uv_neighbors(recon_u, uv_stride, mb_col, mb_row);
    let (above_v, left_v, above_left_v) = get_uv_neighbors(recon_v, uv_stride, mb_col, mb_row);

    let mut u_block = [0u8; 64];
    let mut v_block = [0u8; 64];
    for r in 0..8 {
        u_block[r * 8..r * 8 + 8]
            .copy_from_slice(&u_plane[uv_off + r * uv_stride..uv_off + r * uv_stride + 8]);
        v_block[r * 8..r * 8 + 8]
            .copy_from_slice(&v_plane[uv_off + r * uv_stride..uv_off + r * uv_stride + 8]);
    }

    let uv_mode = predict::select_best_8x8(
        &u_block,
        &above_u,
        &left_u,
        above_left_u,
        mb_row > 0,
        mb_col > 0,
    );

    let mut pred_u = [0u8; 64];
    let mut pred_v = [0u8; 64];
    predict::predict_8x8(
        uv_mode,
        &above_u,
        &left_u,
        above_left_u,
        mb_row > 0,
        mb_col > 0,
        &mut pred_u,
    );
    predict::predict_8x8(
        uv_mode,
        &above_v,
        &left_v,
        above_left_v,
        mb_row > 0,
        mb_col > 0,
        &mut pred_v,
    );

    let mut uv_quantized = [[0i16; 16]; 8];
    let mut uv_idx = 0;
    for (plane_block, pred_block) in [(&u_block, &pred_u), (&v_block, &pred_v)] {
        for sb in 0..4 {
            let sb_row = sb / 2;
            let sb_col = sb % 2;
            let mut src_4x4 = [0u8; 16];
            let mut ref_4x4 = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_4x4[r * 4 + c] = plane_block[(sb_row * 4 + r) * 8 + sb_col * 4 + c];
                    ref_4x4[r * 4 + c] = pred_block[(sb_row * 4 + r) * 8 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_4x4, &ref_4x4, &mut coeffs);
            quant::quantize_block(&coeffs, &seg_quant.uv_ac, &mut uv_quantized[uv_idx]);
            // RDO: prune costly UV coefficients (block_type 2 = UV)
            crate::rdo::rdo_prune_block(
                &mut uv_quantized[uv_idx],
                &coeffs,
                &seg_quant.uv_ac,
                2,
                lambda,
            );
            uv_idx += 1;
        }
    }

    let all_zero = y2_quantized.iter().all(|&c| c == 0)
        && y_coeffs.iter().all(|block| block.iter().all(|&c| c == 0))
        && uv_quantized
            .iter()
            .all(|block| block.iter().all(|&c| c == 0));

    // ─── Token encoding ─────────────────────────────────────────────────
    if all_zero {
        if !use_bpred {
            top_ctx[0] = 0;
            left_ctx[0] = 0;
        }
        for i in 1..9 {
            top_ctx[i] = 0;
            left_ctx[i] = 0;
        }
    } else if use_bpred {
        // B_PRED: no Y2 block. Leave Y2 context UNCHANGED — the decoder
        // does not touch complexity[0] for B_PRED MBs, so the encoder must
        // preserve whatever the previous MB set.

        // Y blocks: plane=3 (Y-with-DC), all 16 coefficients including DC
        for y in 0..4 {
            let mut left = left_ctx[y + 1];
            for x in 0..4 {
                let sb = y * 4 + x;
                let complexity = (top_ctx[x + 1] + left).min(2) as usize;
                let has = token::encode_block(token_bw, &y_coeffs[sb], 3, complexity);
                let ctx_val = if has { 1 } else { 0 };
                left = ctx_val;
                top_ctx[x + 1] = ctx_val;
            }
            left_ctx[y + 1] = left;
        }
    } else {
        // I16x16: Y2 block + Y-AC blocks
        let y2_complexity = (top_ctx[0] + left_ctx[0]).min(2) as usize;
        let y2_has = token::encode_block(token_bw, &y2_quantized, 1, y2_complexity);
        top_ctx[0] = if y2_has { 1 } else { 0 };
        left_ctx[0] = if y2_has { 1 } else { 0 };

        for y in 0..4 {
            let mut left = left_ctx[y + 1];
            for x in 0..4 {
                let sb = y * 4 + x;
                let complexity = (top_ctx[x + 1] + left).min(2) as usize;
                let has = token::encode_block(token_bw, &y_coeffs[sb], 0, complexity);
                let ctx_val = if has { 1 } else { 0 };
                left = ctx_val;
                top_ctx[x + 1] = ctx_val;
            }
            left_ctx[y + 1] = left;
        }
    }

    // Chroma (same for both modes, but skipped when all_zero)
    if !all_zero {
        for y in 0..2 {
            let mut left = left_ctx[y + 5];
            for x in 0..2 {
                let sb = y * 2 + x;
                let complexity = (top_ctx[x + 5] + left).min(2) as usize;
                let has = token::encode_block(token_bw, &uv_quantized[sb], 2, complexity);
                let ctx_val = if has { 1 } else { 0 };
                left = ctx_val;
                top_ctx[x + 5] = ctx_val;
            }
            left_ctx[y + 5] = left;
        }
        for y in 0..2 {
            let mut left = left_ctx[y + 7];
            for x in 0..2 {
                let sb = 4 + y * 2 + x;
                let complexity = (top_ctx[x + 7] + left).min(2) as usize;
                let has = token::encode_block(token_bw, &uv_quantized[sb], 2, complexity);
                let ctx_val = if has { 1 } else { 0 };
                left = ctx_val;
                top_ctx[x + 7] = ctx_val;
            }
            left_ctx[y + 7] = left;
        }
    }

    MacroblockInfo {
        mb_x: mb_col as u32,
        mb_y: mb_row as u32,
        y_mode: final_y_mode,
        b_modes,
        uv_mode: uv_mode as u8,
        segment: 0,
        skip: all_zero,
    }
}

/// Get 4x4 prediction neighbors for a sub-block within a macroblock.
///
/// Returns (above[4], left[4], above_left, above_right[4]) from the
/// reconstructed frame for the sub-block at (sb_row, sb_col) within
/// the MB at (mb_col, mb_row).
fn get_4x4_neighbors(
    recon: &[u8],
    stride: usize,
    mb_col: usize,
    mb_row: usize,
    sb_row: usize,
    sb_col: usize,
) -> ([u8; 4], [u8; 4], u8, [u8; 4]) {
    let base_y = mb_row * 16 + sb_row * 4;
    let base_x = mb_col * 16 + sb_col * 4;

    // Above 4 pixels
    let mut above = [127u8; 4];
    if base_y > 0 {
        for i in 0..4 {
            above[i] = recon[(base_y - 1) * stride + base_x + i];
        }
    }

    // Left 4 pixels
    let mut left = [129u8; 4];
    if base_x > 0 {
        for i in 0..4 {
            left[i] = recon[(base_y + i) * stride + base_x - 1];
        }
    }

    // Above-left pixel
    let above_left = if base_y > 0 && base_x > 0 {
        recon[(base_y - 1) * stride + base_x - 1]
    } else if base_y > 0 {
        129
    } else {
        127
    };

    // Above-right 4 pixels (needed for LD, VL modes)
    let mut above_right = [127u8; 4];
    if base_y > 0 && base_x + 4 + 4 <= stride {
        for i in 0..4 {
            above_right[i] = recon[(base_y - 1) * stride + base_x + 4 + i];
        }
    } else if base_y > 0 {
        // Replicate last above pixel
        let last = recon[(base_y - 1) * stride + (base_x + 3).min(stride - 1)];
        above_right = [last; 4];
    }

    (above, left, above_left, above_right)
}

/// Reconstruct a macroblock for prediction reference by future MBs.
///
/// This mirrors what the decoder does: dequantize → inverse WHT (Y2) →
/// inverse DCT → add prediction. The reconstructed pixels must match exactly
/// what a decoder would produce, so the next macroblock's prediction is correct.
fn reconstruct_macroblock(
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    mb_col: usize,
    mb_row: usize,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    seg_quant: &SegmentQuant,
    mb: &MacroblockInfo,
) {
    let y_off = mb_row * 16 * y_stride + mb_col * 16;
    let uv_off = mb_row * 8 * uv_stride + mb_col * 8;

    // Extract 16x16 luma source block
    let mut y_block = [0u8; 256];
    for r in 0..16 {
        y_block[r * 16..r * 16 + 16]
            .copy_from_slice(&y_plane[y_off + r * y_stride..y_off + r * y_stride + 16]);
    }

    if mb.y_mode == 4 {
        // ─── B_PRED reconstruction ──────────────────────────────────────
        // Per-4x4 prediction, no Y2 WHT. Process blocks in raster order
        // so each block can use previously reconstructed neighbors.
        for sb in 0..16 {
            let sb_row = sb / 4;
            let sb_col = sb % 4;

            // Get 4x4 prediction from reconstructed frame (already has prior blocks)
            let (above_4, left_4, al, ar) =
                get_4x4_neighbors(recon_y, y_stride, mb_col, mb_row, sb_row, sb_col);

            let mode = predict::Intra4Mode::from_u8(mb.b_modes[sb]);
            let mut pred = [0u8; 16];
            predict::predict_4x4(mode, &above_4, &left_4, al, &ar, &mut pred);

            // Forward DCT + quantize (full 16 coefficients including DC)
            let mut src_block = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_block[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_block, &pred, &mut coeffs);
            let mut quantized = [0i16; 16];
            quant::quantize_block(&coeffs, &seg_quant.y_dc, &mut quantized);

            // Dequantize + inverse DCT (DC at position 0, no Y2 path)
            let mut dequant = [0i16; 16];
            quant::dequantize_block(&quantized, &seg_quant.y_dc, &mut dequant);
            let mut recon_block = [0u8; 16];
            dct::inverse_dct(&dequant, &pred, &mut recon_block);

            // Write to recon_y (MUST happen before next block uses this as neighbor)
            for r in 0..4 {
                for c in 0..4 {
                    let row = mb_row * 16 + sb_row * 4 + r;
                    let col = mb_col * 16 + sb_col * 4 + c;
                    recon_y[row * y_stride + col] = recon_block[r * 4 + c];
                }
            }
        }
    } else {
        // ─── I16x16 reconstruction (original path) ──────────────────────
        let (above_y, left_y, above_left_y) = get_y_neighbors(recon_y, y_stride, mb_col, mb_row);

        let mut pred_y = [0u8; 256];
        let y_mode_enum = match mb.y_mode {
            0 => predict::Intra16Mode::DC,
            1 => predict::Intra16Mode::V,
            2 => predict::Intra16Mode::H,
            _ => predict::Intra16Mode::TM,
        };
        predict::predict_16x16(
            y_mode_enum,
            &above_y,
            &left_y,
            above_left_y,
            mb_row > 0,
            mb_col > 0,
            &mut pred_y,
        );

        let mut y_quantized = [[0i16; 16]; 16];
        let mut y_dc_coeffs = [0i16; 16];
        for sb in 0..16 {
            let sb_row = sb / 4;
            let sb_col = sb % 4;
            let mut src_block = [0u8; 16];
            let mut ref_block = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_block[r * 4 + c] = y_block[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                    ref_block[r * 4 + c] = pred_y[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_block, &ref_block, &mut coeffs);
            y_dc_coeffs[sb] = coeffs[0];
            let mut quantized = [0i16; 16];
            quant::quantize_block(&coeffs, &seg_quant.y_ac, &mut quantized);
            quantized[0] = 0;
            y_quantized[sb] = quantized;
        }

        let mut y2_coeffs = [0i16; 16];
        dct::forward_wht(&y_dc_coeffs, &mut y2_coeffs);
        let mut y2_quantized = [0i16; 16];
        quant::quantize_block(&y2_coeffs, &seg_quant.y2_dc, &mut y2_quantized);
        let mut y2_dequant = [0i16; 16];
        quant::dequantize_block(&y2_quantized, &seg_quant.y2_dc, &mut y2_dequant);
        let mut recon_dc = [0i16; 16];
        dct::inverse_wht(&y2_dequant, &mut recon_dc);

        for sb in 0..16 {
            let sb_row = sb / 4;
            let sb_col = sb % 4;
            let mut dequant = [0i16; 16];
            quant::dequantize_block(&y_quantized[sb], &seg_quant.y_ac, &mut dequant);
            dequant[0] = recon_dc[sb];
            let mut ref_block = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    ref_block[r * 4 + c] = pred_y[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                }
            }
            let mut recon_block = [0u8; 16];
            dct::inverse_dct(&dequant, &ref_block, &mut recon_block);
            for r in 0..4 {
                for c in 0..4 {
                    let row = mb_row * 16 + sb_row * 4 + r;
                    let col = mb_col * 16 + sb_col * 4 + c;
                    recon_y[row * y_stride + col] = recon_block[r * 4 + c];
                }
            }
        }
    }

    // Reconstruct chroma (U and V)
    let (above_u, left_u, above_left_u) = get_uv_neighbors(recon_u, uv_stride, mb_col, mb_row);
    let (above_v, left_v, above_left_v) = get_uv_neighbors(recon_v, uv_stride, mb_col, mb_row);

    let mut pred_u = [0u8; 64];
    let mut pred_v = [0u8; 64];
    let uv_mode_enum = match mb.uv_mode {
        0 => predict::ChromaMode::DC,
        1 => predict::ChromaMode::V,
        2 => predict::ChromaMode::H,
        _ => predict::ChromaMode::TM,
    };
    predict::predict_8x8(
        uv_mode_enum,
        &above_u,
        &left_u,
        above_left_u,
        mb_row > 0,
        mb_col > 0,
        &mut pred_u,
    );
    predict::predict_8x8(
        uv_mode_enum,
        &above_v,
        &left_v,
        above_left_v,
        mb_row > 0,
        mb_col > 0,
        &mut pred_v,
    );

    let mut u_block = [0u8; 64];
    let mut v_block = [0u8; 64];
    for r in 0..8 {
        u_block[r * 8..r * 8 + 8]
            .copy_from_slice(&u_plane[uv_off + r * uv_stride..uv_off + r * uv_stride + 8]);
        v_block[r * 8..r * 8 + 8]
            .copy_from_slice(&v_plane[uv_off + r * uv_stride..uv_off + r * uv_stride + 8]);
    }

    for (plane_block, pred_block, recon_plane, stride) in [
        (&u_block, &pred_u, &mut *recon_u, uv_stride),
        (&v_block, &pred_v, &mut *recon_v, uv_stride),
    ] {
        for sb in 0..4 {
            let sb_row = sb / 2;
            let sb_col = sb % 2;
            let mut src_4x4 = [0u8; 16];
            let mut ref_4x4 = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_4x4[r * 4 + c] = plane_block[(sb_row * 4 + r) * 8 + sb_col * 4 + c];
                    ref_4x4[r * 4 + c] = pred_block[(sb_row * 4 + r) * 8 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_4x4, &ref_4x4, &mut coeffs);
            let mut quantized = [0i16; 16];
            quant::quantize_block(&coeffs, &seg_quant.uv_ac, &mut quantized);

            // Dequantize + inverse DCT for reconstruction
            let mut dequant = [0i16; 16];
            quant::dequantize_block(&quantized, &seg_quant.uv_ac, &mut dequant);
            let mut recon_block = [0u8; 16];
            dct::inverse_dct(&dequant, &ref_4x4, &mut recon_block);

            for r in 0..4 {
                for c in 0..4 {
                    let row = mb_row * 8 + sb_row * 4 + r;
                    let col = mb_col * 8 + sb_col * 4 + c;
                    recon_plane[row * stride + col] = recon_block[r * 4 + c];
                }
            }
        }
    }
}

/// Get Y prediction neighbors for a macroblock.
fn get_y_neighbors(
    recon: &[u8],
    stride: usize,
    mb_col: usize,
    mb_row: usize,
) -> ([u8; 16], [u8; 16], u8) {
    // VP8 spec: default top=127, left=129 (matching image-webp decoder)
    let mut above = [127u8; 16];
    let mut left = [129u8; 16];
    let mut above_left = if mb_row == 0 { 127u8 } else { 129u8 };

    if mb_row > 0 {
        let row_above = (mb_row * 16 - 1) * stride + mb_col * 16;
        above.copy_from_slice(&recon[row_above..row_above + 16]);
    }
    if mb_col > 0 {
        for r in 0..16 {
            left[r] = recon[(mb_row * 16 + r) * stride + mb_col * 16 - 1];
        }
    }
    if mb_row > 0 && mb_col > 0 {
        above_left = recon[(mb_row * 16 - 1) * stride + mb_col * 16 - 1];
    }

    (above, left, above_left)
}

/// Get UV prediction neighbors.
fn get_uv_neighbors(
    recon: &[u8],
    stride: usize,
    mb_col: usize,
    mb_row: usize,
) -> ([u8; 8], [u8; 8], u8) {
    // VP8 spec: default top=127, left=129 (matching image-webp decoder)
    let mut above = [127u8; 8];
    let mut left = [129u8; 8];
    let mut above_left = if mb_row == 0 { 127u8 } else { 129u8 };

    if mb_row > 0 {
        let row_above = (mb_row * 8 - 1) * stride + mb_col * 8;
        above.copy_from_slice(&recon[row_above..row_above + 8]);
    }
    if mb_col > 0 {
        for r in 0..8 {
            left[r] = recon[(mb_row * 8 + r) * stride + mb_col * 8 - 1];
        }
    }
    if mb_row > 0 && mb_col > 0 {
        above_left = recon[(mb_row * 8 - 1) * stride + mb_col * 8 - 1];
    }

    (above, left, above_left)
}

/// Pad a plane to the target dimensions by repeating edge pixels.
fn pad_plane(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut out = vec![128u8; dst_w * dst_h];
    for r in 0..dst_h {
        let src_row = r.min(src_h.saturating_sub(1));
        for c in 0..dst_w {
            let src_col = c.min(src_w.saturating_sub(1));
            out[r * dst_w + c] = src[src_row * src_w + src_col];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ratecontrol;

    #[test]
    fn frame_header_valid_structure() {
        let yuv = YuvImage {
            width: 16,
            height: 16,
            y: vec![128u8; 256],
            u: vec![128u8; 64],
            v: vec![128u8; 64],
        };
        let params = ratecontrol::quality_to_params(75);
        let frame = encode_frame(&yuv, &params);

        // Check frame tag (3 bytes)
        let tag = u32::from_le_bytes([frame[0], frame[1], frame[2], 0]);
        let frame_type = tag & 1;
        assert_eq!(frame_type, 0, "should be key frame");
        let show_frame = (tag >> 4) & 1;
        assert_eq!(show_frame, 1, "show_frame should be 1");

        // Check start code
        assert_eq!(&frame[3..6], &[0x9D, 0x01, 0x2A]);

        // Check dimensions
        let w = u16::from_le_bytes([frame[6], frame[7]]) & 0x3FFF;
        let h = u16::from_le_bytes([frame[8], frame[9]]) & 0x3FFF;
        assert_eq!(w, 16);
        assert_eq!(h, 16);
    }

    #[test]
    fn encode_1x1_produces_valid_frame() {
        let yuv = YuvImage {
            width: 1,
            height: 1,
            y: vec![128],
            u: vec![128],
            v: vec![128],
        };
        let params = ratecontrol::quality_to_params(75);
        let frame = encode_frame(&yuv, &params);
        assert!(frame.len() > 10, "frame should have header + data");
    }
}

#[cfg(test)]
mod bpred_tests {
    #[test]
    fn bpred_gradient_decodes() {
        // Test B_PRED with gradient content that triggers per-4x4 mode selection
        for &size in &[32u32, 256, 512] {
            let s = size as usize;
            let mut pixels = vec![0u8; s * s * 3];
            for y in 0..s {
                for x in 0..s {
                    let i = (y * s + x) * 3;
                    pixels[i] = (x * 255 / s) as u8;
                    pixels[i + 1] = (y * 255 / s) as u8;
                    pixels[i + 2] = 128;
                }
            }
            let config = crate::EncodeConfig {
                quality: 75,
                ..Default::default()
            };
            let webp =
                crate::encode(&pixels, size, size, crate::PixelFormat::Rgb8, &config).unwrap();
            let decoded = image::load_from_memory_with_format(&webp, image::ImageFormat::WebP);
            assert!(
                decoded.is_ok(),
                "{size}x{size} gradient failed: {:?}",
                decoded.err()
            );
        }
    }

    #[test]
    fn bpred_wrapping_gradient_decodes() {
        // Wrapping gradient — sharp 255→0 transition triggers B_PRED
        for &size in &[328u32, 512] {
            let s = size as usize;
            let mut pixels = vec![0u8; s * s * 3];
            for y in 0..s {
                for x in 0..s {
                    let i = (y * s + x) * 3;
                    pixels[i] = (x % 256) as u8;
                    pixels[i + 1] = (y % 256) as u8;
                    pixels[i + 2] = 128;
                }
            }
            let webp = crate::encode(
                &pixels,
                size,
                size,
                crate::PixelFormat::Rgb8,
                &crate::EncodeConfig {
                    quality: 75,
                    ..Default::default()
                },
            )
            .unwrap();
            let decoded = image::load_from_memory_with_format(&webp, image::ImageFormat::WebP);
            assert!(
                decoded.is_ok(),
                "{size}x{size} wrap gradient failed: {:?}",
                decoded.err()
            );
        }
    }
}
