//! CABAC syntax element encoding — encode CU/TU/coefficient syntax via CABAC.
//!
//! This is the encoding counterpart to `syntax.rs` (decoder). Each function
//! encodes a specific HEVC syntax element using the CABAC encoder.
//!
//! Ref: x265 4.1 encoder/entropy.cpp — all code* functions
//! Ref: ITU-T H.265 Section 7.3.8 (coding unit/transform unit syntax)

use super::cabac_enc::CabacEncoder;
use crate::cabac::ContextModel;
use crate::syntax::{
    CBF_CHROMA_CTX_OFFSET, CBF_LUMA_CTX_OFFSET, CHROMA_PRED_CTX_OFFSET, PART_MODE_CTX_OFFSET,
    PREV_INTRA_PRED_CTX_OFFSET, SPLIT_TU_CTX_OFFSET,
};

/// Encode split_cu_flag.
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeSplitFlag
pub fn encode_split_cu_flag(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    ctx_idx: usize,
    split: bool,
) {
    enc.encode_bin(split as u32, &mut contexts[ctx_idx]);
}

/// Encode part_mode for intra CU at minimum CU size.
/// bin=1 → Part2Nx2N, bin=0 → PartNxN.
///
/// Ref: x265 4.1 encoder/entropy.cpp — codePartSize
pub fn encode_part_mode_intra(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    is_2nx2n: bool,
) {
    let ctx_idx = PART_MODE_CTX_OFFSET;
    enc.encode_bin(is_2nx2n as u32, &mut contexts[ctx_idx]);
}

/// Encode prev_intra_luma_pred_flag + mpm_idx or rem_intra_luma_pred_mode.
///
/// If mode is in MPM list: flag=1 + mpm_idx (truncated unary, bypass)
/// If mode is not in MPM list: flag=0 + rem_mode (5 bypass bits)
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeIntraDirLumaAng
pub fn encode_intra_luma_mode(enc: &mut CabacEncoder, contexts: &mut [ContextModel], raw_mode: u8) {
    if raw_mode < 3 {
        // MPM mode: flag=1, then mpm_idx via truncated unary bypass
        let ctx_idx = PREV_INTRA_PRED_CTX_OFFSET;
        enc.encode_bin(1, &mut contexts[ctx_idx]);
        let mpm_idx = raw_mode;
        if mpm_idx == 0 {
            enc.encode_bypass(0);
        } else if mpm_idx == 1 {
            enc.encode_bypass(1);
            enc.encode_bypass(0);
        } else {
            enc.encode_bypass(1);
            enc.encode_bypass(1);
        }
    } else {
        // Rem mode: flag=0, then 5-bit value via bypass
        let ctx_idx = PREV_INTRA_PRED_CTX_OFFSET;
        enc.encode_bin(0, &mut contexts[ctx_idx]);
        let rem = raw_mode - 3;
        for bit in (0..5).rev() {
            enc.encode_bypass(((rem >> bit) & 1) as u32);
        }
    }
}

/// Encode intra_chroma_pred_mode.
///
/// bin=0 → DM mode (derived from luma), bin=1 + 2 bypass bits → explicit mode.
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeIntraDirChroma
pub fn encode_intra_chroma_mode(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    chroma_mode: u8,
) {
    let ctx_idx = CHROMA_PRED_CTX_OFFSET;
    if chroma_mode == 4 {
        // DM mode
        enc.encode_bin(0, &mut contexts[ctx_idx]);
    } else {
        enc.encode_bin(1, &mut contexts[ctx_idx]);
        enc.encode_bypass((chroma_mode >> 1) as u32 & 1);
        enc.encode_bypass(chroma_mode as u32 & 1);
    }
}

/// Encode split_transform_flag.
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeTransformSubdivFlag
pub fn encode_split_transform_flag(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    size: u32,
    split: bool,
) {
    let ctx_idx = SPLIT_TU_CTX_OFFSET + 5 - (size as usize).trailing_zeros() as usize;
    let ctx_idx = ctx_idx.min(contexts.len() - 1);
    enc.encode_bin(split as u32, &mut contexts[ctx_idx]);
}

/// Encode cbf_luma.
///
/// Context: depth==0 → ctxInc=1, else ctxInc=0
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeQtCbfLuma
pub fn encode_cbf_luma(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    depth: u32,
    cbf: bool,
) {
    let ctx_idx = CBF_LUMA_CTX_OFFSET + if depth == 0 { 1 } else { 0 };
    enc.encode_bin(cbf as u32, &mut contexts[ctx_idx]);
}

/// Encode cbf_chroma (Cb or Cr).
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeQtCbfChroma
pub fn encode_cbf_chroma(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    depth: u32,
    cbf: bool,
) {
    let ctx_idx = CBF_CHROMA_CTX_OFFSET + depth.min(3) as usize;
    if ctx_idx < contexts.len() {
        enc.encode_bin(cbf as u32, &mut contexts[ctx_idx]);
    }
}

/// Encode last significant coefficient X/Y position.
///
/// Exactly mirrors the decoder's `decode_last_sig_coeff_pos()` in syntax.rs.
/// Uses the same context offset and shift formulas.
///
/// Ref: ITU-T H.265 Section 7.3.8.11 — last_sig_coeff_x/y_prefix
pub fn encode_last_sig_coeff_pos(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    log2_size: usize,
    last_x: u32,
    last_y: u32,
) {
    let max_prefix = (log2_size << 1) - 1;

    // Context derivation — MUST match decoder exactly
    // Ref: syntax.rs decode_last_sig_coeff_pos lines 1114-1115
    let ctx_offset = 3 * (log2_size - 2) + ((log2_size - 1) >> 2);
    let ctx_shift = (log2_size + 1) >> 2;

    // Encode X prefix (truncated unary)
    let last_x_prefix = pos_to_prefix(last_x);
    for i in 0..last_x_prefix as usize {
        let ctx_idx = crate::syntax::LAST_SIG_X_CTX_OFFSET + ctx_offset + (i >> ctx_shift);
        if ctx_idx < contexts.len() {
            enc.encode_bin(1, &mut contexts[ctx_idx]);
        }
    }
    if (last_x_prefix as usize) < max_prefix {
        let ctx_idx = crate::syntax::LAST_SIG_X_CTX_OFFSET
            + ctx_offset
            + (last_x_prefix as usize >> ctx_shift);
        if ctx_idx < contexts.len() {
            enc.encode_bin(0, &mut contexts[ctx_idx]);
        }
    }

    // Encode Y prefix
    let last_y_prefix = pos_to_prefix(last_y);
    for i in 0..last_y_prefix as usize {
        let ctx_idx = crate::syntax::LAST_SIG_Y_CTX_OFFSET + ctx_offset + (i >> ctx_shift);
        if ctx_idx < contexts.len() {
            enc.encode_bin(1, &mut contexts[ctx_idx]);
        }
    }
    if (last_y_prefix as usize) < max_prefix {
        let ctx_idx = crate::syntax::LAST_SIG_Y_CTX_OFFSET
            + ctx_offset
            + (last_y_prefix as usize >> ctx_shift);
        if ctx_idx < contexts.len() {
            enc.encode_bin(0, &mut contexts[ctx_idx]);
        }
    }

    // Encode X suffix (bypass) if prefix > 3
    if last_x_prefix > 3 {
        let suffix_len = (last_x_prefix >> 1) - 1;
        let suffix = last_x - ((2 + (last_x_prefix & 1)) << suffix_len);
        for bit in (0..suffix_len).rev() {
            enc.encode_bypass((suffix >> bit) & 1);
        }
    }

    // Encode Y suffix
    if last_y_prefix > 3 {
        let suffix_len = (last_y_prefix >> 1) - 1;
        let suffix = last_y - ((2 + (last_y_prefix & 1)) << suffix_len);
        for bit in (0..suffix_len).rev() {
            enc.encode_bypass((suffix >> bit) & 1);
        }
    }
}

/// Convert a last significant coefficient position to its prefix code.
///
/// Inverse of the decoder's mapping:
///   prefix < 4: pos = prefix
///   prefix >= 4: pos = ((2 + (prefix & 1)) << ((prefix >> 1) - 1)) + suffix
///
/// Ref: ITU-T H.265 Section 7.4.9.11
fn pos_to_prefix(pos: u32) -> u32 {
    // Positions 0-3 map directly to prefix 0-3
    if pos < 4 {
        return pos;
    }
    // For pos >= 4: find prefix such that the base of that prefix group <= pos
    // Group bases: prefix 4→4, 5→6, 6→8, 7→12, 8→16, 9→24
    let mut prefix = 4u32;
    loop {
        let suffix_len = (prefix >> 1) - 1;
        let base = (2 + (prefix & 1)) << suffix_len;
        let next_prefix = prefix + 1;
        let next_base = (2 + (next_prefix & 1)) << ((next_prefix >> 1) - 1);
        if pos < next_base {
            return prefix;
        }
        prefix = next_prefix;
        if prefix >= 20 {
            return prefix; // safety limit
        }
    }
}

/// Encode residual coefficients for a transform block.
///
/// Full encoding counterpart to `decode_residual_coeffs()` in syntax.rs.
/// Mirrors the decoder's sub-block processing exactly in the encode direction.
///
/// Ref: x265 4.1 encoder/entropy.cpp — codeCoeffNxN
/// Ref: ITU-T H.265 Section 7.3.8.11
pub fn encode_residual_coeffs(
    enc: &mut CabacEncoder,
    contexts: &mut [ContextModel],
    coeffs: &[i16],
    size: u32,
    sign_data_hiding_enabled: bool,
) {
    let log2_size = (size as f32).log2() as usize;
    let c_idx: usize = 0;

    let sb_width = if log2_size <= 2 { 1 } else { size as usize / 4 };
    let sub_scan = crate::syntax::build_scan_order(sb_width);
    let coeff_scan_4x4 = crate::syntax::build_scan_order(4);
    let scan_4x4_size = if log2_size <= 2 { size as usize } else { 4 };

    // Find last significant coefficient in sub-block scan order
    let mut last_sub_scan_pos = 0usize;
    let mut last_coeff_scan_pos = 0usize;
    let mut found = false;
    for (si, &(sx, sy)) in sub_scan.iter().enumerate() {
        let bx = sx * (if log2_size <= 2 { 1 } else { 4 });
        let by = sy * (if log2_size <= 2 { 1 } else { 4 });
        for (ci, &(cx, cy)) in coeff_scan_4x4.iter().enumerate() {
            if cx >= scan_4x4_size || cy >= scan_4x4_size {
                continue;
            }
            let tx = bx + cx;
            let ty = by + cy;
            if tx < size as usize && ty < size as usize && coeffs[ty * size as usize + tx] != 0 {
                last_sub_scan_pos = si;
                last_coeff_scan_pos = ci;
                found = true;
            }
        }
    }
    if !found {
        return;
    }

    // Encode last significant position
    let (lsx, lsy) = sub_scan[last_sub_scan_pos];
    let (lcx, lcy) = coeff_scan_4x4[last_coeff_scan_pos];
    let last_x = if log2_size <= 2 { lcx } else { lsx * 4 + lcx };
    let last_y = if log2_size <= 2 { lcy } else { lsy * 4 + lcy };
    encode_last_sig_coeff_pos(enc, contexts, log2_size, last_x as u32, last_y as u32);

    #[rustfmt::skip]
    let ctx_idx_map_4x4: [usize; 16] = [0,1,4,5,2,3,4,5,6,6,8,8,7,7,8,8];
    let mut coded_sub_block_neighbors = vec![0u8; sb_width * sb_width];
    let mut prev_c1 = 1u32;

    for sub_idx in (0..=last_sub_scan_pos).rev() {
        let (sub_x, sub_y) = sub_scan[sub_idx];
        let is_last_sub = sub_idx == last_sub_scan_pos;
        let is_dc_sub = sub_idx == 0;
        let bx = sub_x * (if log2_size <= 2 { 1 } else { 4 });
        let by = sub_y * (if log2_size <= 2 { 1 } else { 4 });
        let coeff_end = if is_last_sub { last_coeff_scan_pos } else { 15 };

        // Collect abs levels and signs from coefficient array
        let mut abs_c = [0u32; 16];
        let mut sign_c = [0u32; 16];
        let mut has_nz = false;
        for ci in 0..=coeff_end {
            let (cx, cy) = coeff_scan_4x4[ci];
            if cx >= scan_4x4_size || cy >= scan_4x4_size {
                continue;
            }
            let tx = bx + cx;
            let ty = by + cy;
            if tx < size as usize && ty < size as usize {
                let v = coeffs[ty * size as usize + tx];
                abs_c[ci] = v.unsigned_abs() as u32;
                sign_c[ci] = if v < 0 { 1 } else { 0 };
                if v != 0 {
                    has_nz = true;
                }
            }
        }

        // coded_sub_block_flag
        let coded = if is_last_sub || is_dc_sub {
            true
        } else {
            let prev_csbf = coded_sub_block_neighbors[sub_x + sub_y * sb_width];
            let csbf_ctx = (prev_csbf & 1) | (prev_csbf >> 1);
            let ctx = crate::syntax::CODED_SUB_BLOCK_FLAG_CTX_OFFSET
                + csbf_ctx as usize
                + if c_idx != 0 { 2 } else { 0 };
            enc.encode_bin(has_nz as u32, &mut contexts[ctx]);
            has_nz
        };
        if coded {
            if sub_x > 0 {
                coded_sub_block_neighbors[(sub_x - 1) + sub_y * sb_width] |= 1;
            }
            if sub_y > 0 {
                coded_sub_block_neighbors[sub_x + (sub_y - 1) * sb_width] |= 2;
            }
        }
        if !coded {
            continue;
        }

        let prev_csbf = coded_sub_block_neighbors[sub_x + sub_y * sb_width];

        // sig_coeff_flags
        let mut sig = [false; 16];
        let mut num_sig = 0u32;
        let mut first_sp: usize = 16;
        let mut last_sp: usize = 0;
        for ci in (0..=coeff_end).rev() {
            let is_sig = abs_c[ci] != 0;
            if is_last_sub && ci == last_coeff_scan_pos {
                sig[ci] = true;
                num_sig += 1;
                if ci < first_sp {
                    first_sp = ci;
                }
                if num_sig == 1 {
                    last_sp = ci;
                }
                continue;
            }
            if ci == 0 && num_sig == 0 && !is_dc_sub && !is_last_sub {
                sig[ci] = true;
                num_sig += 1;
                first_sp = ci;
                last_sp = ci;
                continue;
            }
            let (cx, cy) = coeff_scan_4x4[ci];
            let sc = crate::syntax::derive_sig_coeff_ctx(
                log2_size,
                c_idx,
                0,
                cx,
                cy,
                sub_x,
                sub_y,
                prev_csbf,
                sb_width,
                &ctx_idx_map_4x4,
            );
            let ctx = crate::syntax::SIG_COEFF_CTX_OFFSET + sc.min(41);
            enc.encode_bin(is_sig as u32, &mut contexts[ctx]);
            sig[ci] = is_sig;
            if is_sig {
                num_sig += 1;
                if ci < first_sp {
                    first_sp = ci;
                }
                if num_sig == 1 {
                    last_sp = ci;
                }
            }
        }
        if num_sig == 0 {
            continue;
        }

        let sign_hidden = sign_data_hiding_enabled && (last_sp as isize - first_sp as isize) > 3;

        // gt1 flags
        let mut ctx_set = if sub_idx == 0 || c_idx > 0 {
            0u32
        } else {
            2u32
        };
        if prev_c1 == 0 {
            ctx_set += 1;
        }
        let mut g1ctx = 1u32;
        let mut gt1_count = 0u32;
        let mut first_gt1: Option<usize> = None;
        for ci in (0..=coeff_end).rev() {
            if !sig[ci] {
                continue;
            }
            if gt1_count < 8 {
                let gt1 = abs_c[ci] > 1;
                let ctx = (crate::syntax::GT1_CTX_OFFSET + (ctx_set * 4 + g1ctx.min(3)) as usize)
                    .min(crate::syntax::GT1_CTX_OFFSET + 23);
                enc.encode_bin(gt1 as u32, &mut contexts[ctx]);
                if gt1 {
                    if first_gt1.is_none() {
                        first_gt1 = Some(ci);
                    }
                    g1ctx = 0;
                } else if g1ctx > 0 && g1ctx < 3 {
                    g1ctx += 1;
                }
                gt1_count += 1;
            }
        }
        prev_c1 = if g1ctx == 0 { 0 } else { 1 };

        // gt2 flag
        if let Some(g2ci) = first_gt1 {
            let gt2 = abs_c[g2ci] > 2;
            let ctx = (crate::syntax::GT2_CTX_OFFSET + ctx_set as usize)
                .min(crate::syntax::GT2_CTX_OFFSET + 5);
            enc.encode_bin(gt2 as u32, &mut contexts[ctx]);
        }

        // sign flags
        for ci in (0..=coeff_end).rev() {
            if sig[ci] {
                if !(sign_hidden && ci == first_sp) {
                    enc.encode_bypass(sign_c[ci]);
                }
            }
        }

        // coeff_abs_level_remaining
        let mut first_8 = [false; 16];
        {
            let mut cnt = 0u32;
            for cj in (0..=coeff_end).rev() {
                if sig[cj] {
                    if cnt < 8 {
                        first_8[cj] = true;
                    }
                    cnt += 1;
                }
            }
        }
        let mut rice = 0u32;
        for ci in (0..=coeff_end).rev() {
            if !sig[ci] {
                continue;
            }
            let (base, needs) = if first_8[ci] {
                if Some(ci) == first_gt1 {
                    let b = if abs_c[ci] > 2 { 3u32 } else { 2 };
                    (b, abs_c[ci] > 2)
                } else if abs_c[ci] > 1 {
                    (2u32, true)
                } else {
                    (1u32, false)
                }
            } else {
                (1u32, true)
            };
            if needs {
                let rem = abs_c[ci] - base;
                encode_coeff_abs_level_remaining(enc, rem, rice);
                if base + rem > 3 * (1u32 << rice) {
                    rice = (rice + 1).min(4);
                }
            }
        }
    }
}

/// Encode coeff_abs_level_remaining using truncated Rice + Exp-Golomb.
///
/// Must produce bytes that decode_coeff_abs_level_remaining reconstructs as `value`.
///
/// Decoder formula:
///   prefix < 3: result = (prefix << rice_param) + suffix
///   prefix >= 3: result = ((1 << (prefix-3)) + 2) << rice_param + suffix
///
/// Ref: syntax.rs decode_coeff_abs_level_remaining
fn encode_coeff_abs_level_remaining(enc: &mut CabacEncoder, value: u32, rice_param: u32) {
    let threshold = 3u32 << rice_param; // = 3 * (1 << rice_param)

    if value < threshold {
        // Truncated Rice: prefix = value >> rice_param, suffix = low rice_param bits
        let prefix = value >> rice_param;
        for _ in 0..prefix {
            enc.encode_bypass(1);
        }
        enc.encode_bypass(0);
        // Rice suffix: rice_param bits of (value & mask)
        let suffix = value & ((1u32 << rice_param) - 1);
        for bit in (0..rice_param).rev() {
            enc.encode_bypass((suffix >> bit) & 1);
        }
    } else {
        // Exp-Golomb escape.
        // Decoder: result = ((1 << eg_order) + 2) << rice_param + suffix
        // Find eg_order such that ((1<<eg_order)+2)<<rice <= value < ((1<<(eg_order+1))+2)<<rice
        let mut eg_order = 0u32;
        loop {
            let range_start = ((1u32 << eg_order) + 2) << rice_param;
            let next_start = ((1u32 << (eg_order + 1)) + 2) << rice_param;
            if value < next_start {
                // value is in this eg_order's range
                let suffix = value - range_start;
                let prefix = 3 + eg_order; // unary prefix length

                // Encode unary prefix (prefix ones + 1 zero)
                for _ in 0..prefix {
                    enc.encode_bypass(1);
                }
                enc.encode_bypass(0);

                // Encode suffix: (eg_order + rice_param) bits
                let suffix_len = eg_order + rice_param;
                for bit in (0..suffix_len).rev() {
                    enc.encode_bypass((suffix >> bit) & 1);
                }
                return;
            }
            eg_order += 1;
            if eg_order > 20 {
                break; // safety
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacDecoder;
    use crate::syntax;

    #[test]
    fn encode_decode_intra_mode_mpm() {
        let qp = 26;
        let mut enc_ctx = syntax::init_syntax_contexts(qp);
        let mut dec_ctx = syntax::init_syntax_contexts(qp);

        // Encode mpm_idx = 0, 1, 2
        for mpm_idx in 0..3u8 {
            let mut enc = CabacEncoder::new();
            let mut enc_c = enc_ctx.clone();

            encode_intra_luma_mode(&mut enc, &mut enc_c, mpm_idx);
            enc.encode_terminate(1);
            let data = enc.finish_and_get_bytes();

            // Decode and verify
            let mut dec = CabacDecoder::new(&data).unwrap();
            let mut dec_c = dec_ctx.clone();

            let ctx_idx = PREV_INTRA_PRED_CTX_OFFSET;
            let flag = dec.decode_bin(&mut dec_c[ctx_idx]).unwrap();
            assert_eq!(flag, 1, "prev_intra should be 1 for MPM");

            let decoded_idx = if dec.decode_bypass().unwrap() == 0 {
                0
            } else if dec.decode_bypass().unwrap() == 0 {
                1
            } else {
                2
            };
            assert_eq!(
                decoded_idx, mpm_idx,
                "mpm_idx mismatch: expected {mpm_idx}, got {decoded_idx}"
            );
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn encode_decode_intra_mode_rem() {
        let qp = 26;

        for rem in 0..32u8 {
            let mut enc = CabacEncoder::new();
            let mut enc_ctx = syntax::init_syntax_contexts(qp);

            encode_intra_luma_mode(&mut enc, &mut enc_ctx, rem + 3);
            enc.encode_terminate(1);
            let data = enc.finish_and_get_bytes();

            let mut dec = CabacDecoder::new(&data).unwrap();
            let mut dec_ctx = syntax::init_syntax_contexts(qp);

            let ctx_idx = PREV_INTRA_PRED_CTX_OFFSET;
            let flag = dec.decode_bin(&mut dec_ctx[ctx_idx]).unwrap();
            assert_eq!(flag, 0, "prev_intra should be 0 for rem mode");

            let mut decoded_rem = 0u8;
            for bit in (0..5).rev() {
                decoded_rem |= (dec.decode_bypass().unwrap() as u8) << bit;
            }
            assert_eq!(
                decoded_rem, rem,
                "rem mode mismatch: expected {rem}, got {decoded_rem}"
            );
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn encode_decode_chroma_mode() {
        let qp = 26;

        // DM mode (4)
        {
            let mut enc = CabacEncoder::new();
            let mut ctx = syntax::init_syntax_contexts(qp);
            encode_intra_chroma_mode(&mut enc, &mut ctx, 4);
            enc.encode_terminate(1);
            let data = enc.finish_and_get_bytes();

            let mut dec = CabacDecoder::new(&data).unwrap();
            let mut dec_ctx = syntax::init_syntax_contexts(qp);
            let flag = dec
                .decode_bin(&mut dec_ctx[CHROMA_PRED_CTX_OFFSET])
                .unwrap();
            assert_eq!(flag, 0, "DM mode should have flag=0");
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }

        // Explicit modes (0-3)
        for mode in 0..4u8 {
            let mut enc = CabacEncoder::new();
            let mut ctx = syntax::init_syntax_contexts(qp);
            encode_intra_chroma_mode(&mut enc, &mut ctx, mode);
            enc.encode_terminate(1);
            let data = enc.finish_and_get_bytes();

            let mut dec = CabacDecoder::new(&data).unwrap();
            let mut dec_ctx = syntax::init_syntax_contexts(qp);
            let flag = dec
                .decode_bin(&mut dec_ctx[CHROMA_PRED_CTX_OFFSET])
                .unwrap();
            assert_eq!(flag, 1, "explicit chroma mode should have flag=1");
            let b0 = dec.decode_bypass().unwrap();
            let b1 = dec.decode_bypass().unwrap();
            let decoded = (b0 * 2 + b1) as u8;
            assert_eq!(decoded, mode, "chroma mode mismatch");
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn encode_decode_cbf() {
        let qp = 26;

        // CBF luma at depth 0 and depth 1
        for depth in 0..2u32 {
            for cbf in [false, true] {
                let mut enc = CabacEncoder::new();
                let mut ctx = syntax::init_syntax_contexts(qp);
                encode_cbf_luma(&mut enc, &mut ctx, depth, cbf);
                enc.encode_terminate(1);
                let data = enc.finish_and_get_bytes();

                let mut dec = CabacDecoder::new(&data).unwrap();
                let mut dec_ctx = syntax::init_syntax_contexts(qp);
                let cbf_ctx = CBF_LUMA_CTX_OFFSET + if depth == 0 { 1 } else { 0 };
                let decoded = dec.decode_bin(&mut dec_ctx[cbf_ctx]).unwrap() != 0;
                assert_eq!(decoded, cbf, "cbf_luma mismatch at depth {depth}");
                assert_eq!(dec.decode_terminate().unwrap(), 1);
            }
        }
    }
}
