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
use crate::token;
use rasmcore_color::YuvImage;

/// Encode a VP8 key frame from YUV420 data.
///
/// Returns the raw VP8 frame data (without RIFF container).
pub fn encode_frame(yuv: &YuvImage, params: &EncodeParams) -> Vec<u8> {
    let width = yuv.width;
    let height = yuv.height;
    let (mb_w, mb_h) = block::mb_dimensions(width, height);
    let seg_quant = quant::build_segment_quant(params.qp_y);

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
            let mb = encode_macroblock(
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
                &seg_quant,
                &mut token_writer,
                &mut top_ctx[mb_col],
                &mut left_ctx,
            );
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
                &seg_quant,
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

    // Build first partition (macroblock modes)
    let first_partition = encode_first_partition(&mb_infos, mb_w, mb_h, params);

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
) -> Vec<u8> {
    let mut bw = BoolWriter::with_capacity(1024);

    // Color space and clamping (RFC 6386 Section 9.2)
    bw.put_bit(128, false); // color_space = 0 (YUV)
    bw.put_bit(128, false); // clamping_type = 0

    // Segmentation (RFC 6386 Section 9.3) — disabled
    bw.put_bit(128, false); // segmentation_enabled = false

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

    // Macroblock header: mb_no_coeff_skip flag
    // Disable mb_no_coeff_skip — always encode all tokens for every MB.
    // This is simpler and avoids skip signaling bugs. Less efficient but correct.
    bw.put_bit(128, false); // mb_no_coeff_skip = false

    // Encode macroblock modes
    for mb in mb_infos {
        encode_mb_header(&mut bw, mb);
    }

    bw.finish()
}

/// Encode a single macroblock header in the first partition.
fn encode_mb_header(bw: &mut BoolWriter, mb: &MacroblockInfo) {
    // No skip flag when mb_no_coeff_skip = false

    // Luma prediction mode (RFC 6386 Section 11.2)
    // Key frame Y mode is coded with a fixed tree
    encode_intra_y_mode(bw, mb.y_mode);

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
/// We only use DC(0), V(1), H(2), TM(3) — never B_PRED(4).
fn encode_intra_y_mode(bw: &mut BoolWriter, mode: u8) {
    // Node 0: skip B_PRED (we never use it)
    bw.put_bit(145, true); // not B_PRED → node 1

    match mode {
        0 => {
            // DC_PRED: node1 left → node2 left
            bw.put_bit(156, false); // → node 2
            bw.put_bit(163, false); // → DC
        }
        1 => {
            // V_PRED: node1 left → node2 right
            bw.put_bit(156, false); // → node 2
            bw.put_bit(163, true); // → V
        }
        2 => {
            // H_PRED: node1 right → node3 left
            bw.put_bit(156, true); // → node 3
            bw.put_bit(128, false); // → H
        }
        3 => {
            // TM_PRED: node1 right → node3 right
            bw.put_bit(156, true); // → node 3
            bw.put_bit(128, true); // → TM
        }
        _ => unreachable!("invalid y_mode: {mode}"),
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

/// Encode a single macroblock: choose prediction, compute residual, DCT, quantize.
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

    // Select best 16×16 Y mode
    let y_mode = predict::select_best_16x16(
        &y_block,
        &above_y,
        &left_y,
        above_left_y,
        mb_row > 0,
        mb_col > 0,
    );

    // Generate prediction
    let mut pred_y = [0u8; 256];
    predict::predict_16x16(
        y_mode,
        &above_y,
        &left_y,
        above_left_y,
        mb_row > 0,
        mb_col > 0,
        &mut pred_y,
    );

    // Compute residual, DCT, quantize for all 16 Y sub-blocks
    let mut y_coeffs = [[0i16; 16]; 16];
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
        y_dc_coeffs[sb] = coeffs[0]; // Raw DC for WHT path (NOT quantized)
        let mut quantized = [0i16; 16];
        quant::quantize_block(&coeffs, &seg_quant.y_ac, &mut quantized);
        quantized[0] = 0; // DC goes to Y2 block, not Y-AC
        y_coeffs[sb] = quantized;
    }

    // WHT on DC coefficients (Y2 block)
    let mut y2_coeffs = [0i16; 16];
    dct::forward_wht(&y_dc_coeffs, &mut y2_coeffs);
    let mut y2_quantized = [0i16; 16];
    quant::quantize_block(&y2_coeffs, &seg_quant.y2_dc, &mut y2_quantized);

    // Chroma: select mode, predict, compute coefficients
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

    // Compute all 8 chroma sub-block coefficients
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
            uv_idx += 1;
        }
    }

    // Encode tokens with inter-MB complexity context tracking.
    // Matches decoder's read_residual_data (vp8.rs lines 1575-1650).

    // Y2 block: plane=1, complexity from top[0] + left[0]
    let y2_complexity = (top_ctx[0] + left_ctx[0]).min(2) as usize;
    let y2_has = token::encode_block(token_bw, &y2_quantized, 1, y2_complexity);
    top_ctx[0] = if y2_has { 1 } else { 0 };
    left_ctx[0] = if y2_has { 1 } else { 0 };

    // Y blocks: plane=0, y=0..3 (outer), x=0..3 (inner)
    // complexity from top[x+1] + left[y+1]
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

    // U blocks: plane=2, y=0..1 (outer), x=0..1 (inner)
    // complexity from top[x+5] + left[y+5]
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

    // V blocks: plane=2, y=0..1 (outer), x=0..1 (inner)
    // complexity from top[x+7] + left[y+7]
    for y in 0..2 {
        let mut left = left_ctx[y + 7];
        for x in 0..2 {
            let sb = 4 + y * 2 + x; // V blocks are indices 4-7 in uv_quantized
            let complexity = (top_ctx[x + 7] + left).min(2) as usize;
            let has = token::encode_block(token_bw, &uv_quantized[sb], 2, complexity);
            let ctx_val = if has { 1 } else { 0 };
            left = ctx_val;
            top_ctx[x + 7] = ctx_val;
        }
        left_ctx[y + 7] = left;
    }

    let all_zero = y2_quantized.iter().all(|&c| c == 0)
        && y_coeffs.iter().all(|block| block.iter().all(|&c| c == 0))
        && uv_quantized
            .iter()
            .all(|block| block.iter().all(|&c| c == 0));

    MacroblockInfo {
        mb_x: mb_col as u32,
        mb_y: mb_row as u32,
        y_mode: y_mode as u8,
        b_modes: [0; 16],
        uv_mode: uv_mode as u8,
        segment: 0,
        skip: all_zero,
    }
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

    // Get prediction neighbors (same as encode path)
    let (above_y, left_y, above_left_y) = get_y_neighbors(recon_y, y_stride, mb_col, mb_row);

    // Generate prediction (same mode as encode)
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

    // Forward DCT + quantize each Y sub-block (re-encode to get quantized coeffs)
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
        y_dc_coeffs[sb] = coeffs[0]; // Raw DC for WHT path (NOT quantized)
        let mut quantized = [0i16; 16];
        quant::quantize_block(&coeffs, &seg_quant.y_ac, &mut quantized);
        quantized[0] = 0; // DC goes to Y2
        y_quantized[sb] = quantized;
    }

    // WHT on raw DC coefficients, quantize, then dequantize + inverse WHT
    let mut y2_coeffs = [0i16; 16];
    dct::forward_wht(&y_dc_coeffs, &mut y2_coeffs);
    let mut y2_quantized = [0i16; 16];
    quant::quantize_block(&y2_coeffs, &seg_quant.y2_dc, &mut y2_quantized);

    // Dequantize Y2 and inverse WHT to get reconstructed DC coefficients
    let mut y2_dequant = [0i16; 16];
    quant::dequantize_block(&y2_quantized, &seg_quant.y2_dc, &mut y2_dequant);
    let mut recon_dc = [0i16; 16];
    dct::inverse_wht(&y2_dequant, &mut recon_dc);

    // Reconstruct each Y sub-block: dequantize AC + restored DC → inverse DCT
    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;

        // Dequantize AC coefficients
        let mut dequant = [0i16; 16];
        quant::dequantize_block(&y_quantized[sb], &seg_quant.y_ac, &mut dequant);
        // Restore DC from inverse WHT output
        dequant[0] = recon_dc[sb];

        // Get prediction block for this sub-block
        let mut ref_block = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                ref_block[r * 4 + c] = pred_y[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }

        // Inverse DCT: adds dequantized residual to prediction
        let mut recon_block = [0u8; 16];
        dct::inverse_dct(&dequant, &ref_block, &mut recon_block);

        // Write reconstructed pixels to recon_y
        for r in 0..4 {
            for c in 0..4 {
                let row = mb_row * 16 + sb_row * 4 + r;
                let col = mb_col * 16 + sb_col * 4 + c;
                recon_y[row * y_stride + col] = recon_block[r * 4 + c];
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
