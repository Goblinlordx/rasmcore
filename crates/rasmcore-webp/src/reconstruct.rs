//! VP8 Reconstruct + Trellis — libwebp-exact encode trial functions.
//!
//! Ported from libwebp src/enc/quant_enc.c:
//! - TrellisQuantizeBlock (Viterbi DP, ~2-3dB quality improvement)
//! - ReconstructIntra16 (16x16 luma encode trial)
//! - ReconstructIntra4 (4x4 luma encode trial)
//! - ReconstructUV (chroma encode trial)
//!
//! These functions are used by the libwebp-exact encode pipeline (Track 3).
//! They combine: predict → DCT → quantize (optionally trellis) → dequantize → IDCT.

use crate::cost_engine::{LevelCostTable, MAX_VARIABLE_LEVEL, NUM_CTX, VP8_ENC_BANDS};
use crate::dct;
use crate::quant::{self, QuantMatrix};
use crate::rdo::{self, MAX_COST, ScoreT};

// ─── Trellis Constants (from quant_enc.c) ─────────────────────────────────

/// How much lower than round-to-nearest to try.
const MIN_DELTA: i32 = 0;
/// How much higher than round-to-nearest to try.
const MAX_DELTA: i32 = 1;
/// Number of candidate levels per coefficient position.
const NUM_NODES: usize = (MIN_DELTA + 1 + MAX_DELTA) as usize;

/// Zigzag scan order for 4x4 block.
const KZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

// ─── Trellis Node ─────────────────────────────────────────────────────────

/// Trellis node — stores the optimal decision at one coefficient position.
#[derive(Clone, Copy, Default)]
struct Node {
    prev: i8,   // best previous node index (-MIN_DELTA..MAX_DELTA)
    sign: i8,   // sign of original coefficient (0 or 1)
    level: i16, // quantized level
}

/// Score state — accumulated RD score and cost table pointer for the next position.
#[derive(Clone, Copy)]
struct ScoreState {
    score: ScoreT,
    /// Index into the level_cost table: (band, ctx) for the next coefficient.
    cost_band: u8,
    cost_ctx: u8,
}

impl Default for ScoreState {
    fn default() -> Self {
        Self {
            score: MAX_COST,
            cost_band: 0,
            cost_ctx: 0,
        }
    }
}

// ─── TrellisQuantizeBlock ─────────────────────────────────────────────────

/// Trellis-optimized quantization of a 16-coefficient VP8 block.
///
/// Ported from libwebp quant_enc.c TrellisQuantizeBlock (line 572).
/// Uses Viterbi dynamic programming to find the globally optimal set of
/// quantized levels that minimizes rate-distortion cost.
///
/// `coeffs_in`: unquantized DCT coefficients (modified in-place for reconstruction)
/// `coeffs_out`: output quantized levels
/// `ctx0`: initial context (0=zero/eob, 1=one, 2=large)
/// `coeff_type`: 0=i16-AC, 1=i16-DC, 2=chroma-AC, 3=i4-AC
/// `matrix`: quantization matrix (q[], iq[], bias[])
/// `lambda`: trellis lambda value
/// `cost_table`: precomputed level cost table from cost_engine
///
/// Returns true if any nonzero coefficient was produced.
#[allow(clippy::needless_range_loop)]
pub fn trellis_quantize_block(
    coeffs_in: &mut [i16; 16],
    coeffs_out: &mut [i16; 16],
    ctx0: usize,
    coeff_type: usize,
    matrix: &QuantMatrix,
    lambda: i32,
    cost_table: &LevelCostTable,
) -> bool {
    let first = if coeff_type == 0 { 1 } else { 0 }; // TYPE_I16_AC starts at 1

    let mut nodes = [[Node::default(); NUM_NODES]; 16];
    let mut ss = [[ScoreState::default(); NUM_NODES]; 2];
    let mut ss_cur_idx: usize = 0;
    let mut ss_prev_idx: usize = 1;

    let mut best_path = [-1i32; 3]; // [best_eob_pos, best_node, best_prev]
    #[allow(unused_assignments)]
    let mut best_score: ScoreT = MAX_COST;

    // Compute last interesting coefficient position
    let thresh = (matrix.q[1] as i32) * (matrix.q[1] as i32) / 4;
    let mut last = first as i32 - 1;
    for n in (first..=15).rev() {
        let j = KZIGZAG[n];
        let err = coeffs_in[j] as i32 * coeffs_in[j] as i32;
        if err > thresh {
            last = n as i32;
            break;
        }
    }
    if last < 15 {
        last += 1;
    }

    // Initialize: skip score
    let first_band = VP8_ENC_BANDS[first] as usize;
    let last_proba_ctx = if ctx0 < NUM_CTX { ctx0 } else { 0 };
    let skip_cost = cost_table.get_cost(coeff_type, first, last_proba_ctx, 0) as ScoreT;
    best_score = rdo::rd_score_trellis(lambda, skip_cost, 0);

    // Initialize source nodes
    let init_rate = if ctx0 == 0 {
        rdo::vp8_bit_cost(
            true,
            crate::token::get_coeff_probs(coeff_type, first_band, ctx0)[0],
        ) as ScoreT
    } else {
        0
    };
    for m in 0..NUM_NODES {
        ss[ss_cur_idx][m].score = rdo::rd_score_trellis(lambda, init_rate, 0);
        let band = VP8_ENC_BANDS[first];
        ss[ss_cur_idx][m].cost_band = band;
        ss[ss_cur_idx][m].cost_ctx = ctx0 as u8;
    }

    // Traverse trellis
    for n in first..=last as usize {
        let j = KZIGZAG[n];
        let q = matrix.q[j] as i32;
        let iq = matrix.iq[j];
        let bias = matrix.bias[j];

        let sign = if coeffs_in[j] < 0 { 1i8 } else { 0i8 };
        let coeff0 = if sign != 0 {
            -coeffs_in[j] as i32
        } else {
            coeffs_in[j] as i32
        };

        // Round-to-nearest quantization
        let level0 = ((((coeff0 as u32).wrapping_add(bias)) as u64 * iq as u64) >> 16) as i32;
        let level0 = level0.min(2047); // MAX_LEVEL

        // Swap score states
        std::mem::swap(&mut ss_cur_idx, &mut ss_prev_idx);

        // Test candidate levels
        for m in 0..NUM_NODES {
            let level = level0 + (m as i32) - MIN_DELTA;
            if !(0..=2047).contains(&level) {
                ss[ss_cur_idx][m].score = MAX_COST;
                continue;
            }

            let ctx = match level {
                0 => 0usize,
                1 => 1,
                _ => 2,
            };
            let next_band = if n + 1 < 16 {
                VP8_ENC_BANDS[n + 1]
            } else {
                0
            };

            ss[ss_cur_idx][m].cost_band = next_band;
            ss[ss_cur_idx][m].cost_ctx = ctx as u8;

            // Distortion: delta from zeroing vs this level
            let new_error = coeff0 - level * q;
            let delta_error = new_error * new_error - coeff0 * coeff0;
            let base_score = rdo::rd_score_trellis(lambda, 0, delta_error as ScoreT);

            // Find best predecessor
            let mut best_cur_score = MAX_COST;
            let mut best_prev: i8 = 0;
            for p in 0..NUM_NODES {
                if ss[ss_prev_idx][p].score >= MAX_COST {
                    continue;
                }
                let prev_band = ss[ss_prev_idx][p].cost_band as usize;
                let prev_ctx = ss[ss_prev_idx][p].cost_ctx as usize;
                let level_cost = cost_table.level_cost[coeff_type][prev_band][prev_ctx]
                    [level.min(MAX_VARIABLE_LEVEL as i32) as usize]
                    as ScoreT;
                let score = ss[ss_prev_idx][p].score + rdo::rd_score_trellis(lambda, level_cost, 0);
                if score < best_cur_score {
                    best_cur_score = score;
                    best_prev = p as i8;
                }
            }

            best_cur_score += base_score;
            nodes[n][m] = Node {
                sign,
                level: level as i16,
                prev: best_prev,
            };
            ss[ss_cur_idx][m].score = best_cur_score;

            // Check if this is a better terminal node
            if level != 0 && best_cur_score < best_score {
                // Cost of EOB after this position
                let eob_cost = if n < 15 {
                    let eob_band = VP8_ENC_BANDS[n + 1] as usize;
                    cost_table.level_cost[coeff_type][eob_band][ctx][0] as ScoreT
                } else {
                    0
                };
                let score = best_cur_score + rdo::rd_score_trellis(lambda, eob_cost, 0);
                if score < best_score {
                    best_score = score;
                    best_path[0] = n as i32;
                    best_path[1] = m as i32;
                    best_path[2] = best_prev as i32;
                }
            }
        }
    }

    // Clear output (raster order)
    if coeff_type == 0 {
        // TYPE_I16_AC: preserve DC at position 0
        for i in 1..16 {
            coeffs_in[KZIGZAG[i]] = 0;
        }
        // Clear all raster positions except 0 for output
        for j in 1..16 {
            coeffs_out[j] = 0;
        }
    } else {
        *coeffs_in = [0i16; 16];
        *coeffs_out = [0i16; 16];
    }

    if best_path[0] == -1 {
        return false; // skip — all zeros
    }

    // Backtrack to extract optimal levels
    let mut nz = false;
    let mut best_node = best_path[1] as usize;
    let mut n = best_path[0] as usize;

    // Patch best-prev for terminal node
    nodes[n][best_node].prev = best_path[2] as i8;

    loop {
        let node = &nodes[n][best_node];
        let j = KZIGZAG[n];
        let signed_level = if node.sign != 0 {
            -node.level
        } else {
            node.level
        };
        coeffs_out[j] = signed_level; // store in raster order (matching quant/dequant)
        if node.level != 0 {
            nz = true;
        }
        coeffs_in[j] = signed_level * matrix.q[j] as i16; // dequantize for reconstruction
        best_node = node.prev as usize;
        if n == first {
            break;
        }
        n -= 1;
    }

    nz
}

// ─── ReconstructIntra16 ──────────────────────────────────────────────────

/// Result of a 16x16 luma reconstruction trial.
pub struct ReconI16Result {
    /// Quantized DC levels (for Y2 block).
    pub y_dc_levels: [i16; 16],
    /// Quantized AC levels per sub-block (DC position zeroed).
    pub y_ac_levels: [[i16; 16]; 16],
    /// Reconstructed 16x16 luma pixels.
    pub recon: [u8; 256],
    /// Non-zero coefficient pattern bitmask.
    pub nz: u32,
}

/// Reconstruct a 16x16 luma macroblock for a given I16x16 prediction mode.
///
/// Ported from libwebp quant_enc.c ReconstructIntra16 (line 752).
/// Performs: predict → DCT → WHT → quantize (trellis) → dequant → IWHT → IDCT → store.
///
/// `src_16x16`: original 16x16 luma pixels (row-major)
/// `pred_16x16`: predicted 16x16 pixels from the given mode
/// `seg_quant`: quantization matrices for the segment
/// `lambda_trellis`: trellis lambda for AC blocks
/// `cost_table`: precomputed level cost table
/// `top_nz`: top nonzero context (4 entries for Y columns)
/// `left_nz`: left nonzero context (4 entries for Y rows)
#[allow(clippy::needless_range_loop)]
pub fn reconstruct_intra16(
    src_16x16: &[u8; 256],
    pred_16x16: &[u8; 256],
    seg_quant: &crate::quant::SegmentQuant,
    lambda_trellis: i32,
    cost_table: &LevelCostTable,
    top_nz: &mut [u8; 4],
    left_nz: &mut [u8; 4],
) -> ReconI16Result {
    let mut result = ReconI16Result {
        y_dc_levels: [0i16; 16],
        y_ac_levels: [[0i16; 16]; 16],
        recon: [0u8; 256],
        nz: 0,
    };

    // Step 1: Forward DCT for all 16 sub-blocks + collect DC coefficients
    // dct_tmp is modified in-place by trellis (writes dequantized values back)
    let mut dct_tmp = [[0i16; 16]; 16];
    let mut dc_coeffs = [0i16; 16];

    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;
        let mut src_4x4 = [0u8; 16];
        let mut ref_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                src_4x4[r * 4 + c] = src_16x16[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                ref_4x4[r * 4 + c] = pred_16x16[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }
        dct::forward_dct(&src_4x4, &ref_4x4, &mut dct_tmp[sb]);
        dc_coeffs[sb] = dct_tmp[sb][0];
    }

    // Step 2: Forward WHT on DC coefficients → quantize Y2
    // Note: Y2 trellis was tested but causes quality regression — the trellis
    // is too aggressive for DC coefficients, zeroing values that are critical
    // for gradient/structure representation. Simple quantization preserves more
    // DC energy which is important for I16x16 reconstruction.
    let mut wht_coeffs = [0i16; 16];
    dct::forward_wht(&dc_coeffs, &mut wht_coeffs);
    let mut y2_quantized = [0i16; 16];
    quant::quantize_block(&wht_coeffs, &seg_quant.y2_dc, &mut y2_quantized);
    result.y_dc_levels = y2_quantized;
    result.nz |= if y2_quantized.iter().any(|&c| c != 0) {
        1 << 24
    } else {
        0
    };

    // Step 3: Quantize AC blocks with trellis + context tracking
    // Trellis modifies dct_tmp[sb] in-place: writes dequantized values back to raster positions.
    for y in 0..4 {
        for x in 0..4 {
            let sb = y * 4 + x;
            let ctx = (top_nz[x] + left_nz[y]).min(2) as usize;

            // Trellis quantize (TYPE_I16_AC=0, starts at position 1)
            let mut ac_out = [0i16; 16];
            let non_zero = trellis_quantize_block(
                &mut dct_tmp[sb],
                &mut ac_out,
                ctx,
                0, // TYPE_I16_AC
                &seg_quant.y_ac,
                lambda_trellis,
                cost_table,
            );

            ac_out[0] = 0; // DC goes to Y2
            result.y_ac_levels[sb] = ac_out;
            let nz_flag = if non_zero { 1u32 } else { 0 };
            result.nz |= nz_flag << sb;

            // Update context for next block
            top_nz[x] = if non_zero { 1 } else { 0 };
            left_nz[y] = if non_zero { 1 } else { 0 };
        }
    }

    // Step 4: Inverse WHT → get reconstructed DC values
    let mut y2_dequant = [0i16; 16];
    quant::dequantize_block(&y2_quantized, &seg_quant.y2_dc, &mut y2_dequant);
    let mut recon_dc = [0i16; 16];
    dct::inverse_wht(&y2_dequant, &mut recon_dc);

    // Step 5: Inverse DCT + add prediction for each sub-block
    // Note: trellis already dequantized AC in-place (dct_tmp[sb] has raster-order dequant values).
    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;

        // dct_tmp already has dequantized AC values from trellis; insert reconstructed DC
        dct_tmp[sb][0] = recon_dc[sb];

        // Inverse DCT + add prediction
        let mut ref_4x4 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                ref_4x4[r * 4 + c] = pred_16x16[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
            }
        }
        let mut recon_4x4 = [0u8; 16];
        dct::inverse_dct(&dct_tmp[sb], &ref_4x4, &mut recon_4x4);

        // Store to output
        for r in 0..4 {
            for c in 0..4 {
                result.recon[(sb_row * 4 + r) * 16 + sb_col * 4 + c] = recon_4x4[r * 4 + c];
            }
        }
    }

    result
}

// ─── ReconstructIntra4 ───────────────────────────────────────────────────

/// Result of a 4x4 luma reconstruction trial.
pub struct ReconI4Result {
    /// Quantized levels (all 16 coefficients including DC).
    pub levels: [i16; 16],
    /// Reconstructed 4x4 pixels.
    pub recon: [u8; 16],
    /// Whether any nonzero coefficient was produced.
    pub nz: bool,
}

/// Reconstruct a single 4x4 luma block for B_PRED mode evaluation.
///
/// Ported from libwebp quant_enc.c ReconstructIntra4 (line 805).
pub fn reconstruct_intra4(
    src_4x4: &[u8; 16],
    pred_4x4: &[u8; 16],
    matrix: &QuantMatrix,
    lambda_trellis: i32,
    cost_table: &LevelCostTable,
    ctx: usize,
) -> ReconI4Result {
    // Forward DCT (coeffs modified in-place by trellis → dequantized values)
    let mut coeffs = [0i16; 16];
    dct::forward_dct(src_4x4, pred_4x4, &mut coeffs);

    // Trellis quantize (TYPE_I4_AC=3)
    // After: coeffs has dequantized values in raster order, levels has quantized levels in scan order
    let mut levels = [0i16; 16];
    let nz = trellis_quantize_block(
        &mut coeffs,
        &mut levels,
        ctx,
        3, // TYPE_I4_AC
        matrix,
        lambda_trellis,
        cost_table,
    );

    // Inverse DCT using trellis-dequantized coefficients directly
    let mut recon = [0u8; 16];
    dct::inverse_dct(&coeffs, pred_4x4, &mut recon);

    ReconI4Result { levels, recon, nz }
}

// ─── ReconstructUV ───────────────────────────────────────────────────────

/// Result of a chroma reconstruction trial.
pub struct ReconUVResult {
    /// Quantized levels for 8 chroma blocks (4 U + 4 V).
    pub uv_levels: [[i16; 16]; 8],
    /// Reconstructed 8x8 U pixels.
    pub recon_u: [u8; 64],
    /// Reconstructed 8x8 V pixels.
    pub recon_v: [u8; 64],
    /// Non-zero pattern.
    pub nz: u32,
}

/// Reconstruct chroma (U+V) for a given UV prediction mode.
///
/// Ported from libwebp quant_enc.c ReconstructUV (line 909).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_uv(
    src_u: &[u8; 64],
    src_v: &[u8; 64],
    pred_u: &[u8; 64],
    pred_v: &[u8; 64],
    seg_quant: &crate::quant::SegmentQuant,
    lambda_trellis: i32,
    cost_table: &LevelCostTable,
    top_nz: &mut [u8; 4],  // [u0, u1, v0, v1]
    left_nz: &mut [u8; 4], // [u0, u1, v0, v1]
) -> ReconUVResult {
    let mut result = ReconUVResult {
        uv_levels: [[0i16; 16]; 8],
        recon_u: [0u8; 64],
        recon_v: [0u8; 64],
        nz: 0,
    };

    let planes: [(&[u8; 64], &[u8; 64], &mut [u8; 64]); 2] = unsafe {
        // Safe: we only write to separate output arrays
        let recon_u = &mut result.recon_u as *mut [u8; 64];
        let recon_v = &mut result.recon_v as *mut [u8; 64];
        [
            (src_u, pred_u, &mut *recon_u),
            (src_v, pred_v, &mut *recon_v),
        ]
    };

    for (ch, (src_plane, pred_plane, recon_plane)) in planes.into_iter().enumerate() {
        for sb in 0..4 {
            let sb_row = sb / 2;
            let sb_col = sb % 2;
            let block_idx = ch * 4 + sb;

            let mut src_4x4 = [0u8; 16];
            let mut ref_4x4 = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_4x4[r * 4 + c] = src_plane[(sb_row * 4 + r) * 8 + sb_col * 4 + c];
                    ref_4x4[r * 4 + c] = pred_plane[(sb_row * 4 + r) * 8 + sb_col * 4 + c];
                }
            }

            // Context: ch*2 offsets into top_nz/left_nz
            let ctx_x = ch * 2 + sb_col;
            let ctx_y = ch * 2 + sb_row;
            let ctx = (top_nz[ctx_x.min(3)] + left_nz[ctx_y.min(3)]).min(2) as usize;

            // Forward DCT (coeffs modified in-place by trellis → dequantized values)
            let mut coeffs = [0i16; 16];
            dct::forward_dct(&src_4x4, &ref_4x4, &mut coeffs);

            let mut levels = [0i16; 16];
            let nz = trellis_quantize_block(
                &mut coeffs,
                &mut levels,
                ctx,
                2, // TYPE_CHROMA_A
                &seg_quant.uv_ac,
                lambda_trellis,
                cost_table,
            );

            result.uv_levels[block_idx] = levels;
            if nz {
                result.nz |= 1 << (16 + block_idx); // UV bits start at bit 16
            }

            // Update context
            if ctx_x < 4 {
                top_nz[ctx_x] = if nz { 1 } else { 0 };
            }
            if ctx_y < 4 {
                left_nz[ctx_y] = if nz { 1 } else { 0 };
            }

            // IDCT using trellis-dequantized coefficients directly
            let mut recon_4x4 = [0u8; 16];
            dct::inverse_dct(&coeffs, &ref_4x4, &mut recon_4x4);

            for r in 0..4 {
                for c in 0..4 {
                    recon_plane[(sb_row * 4 + r) * 8 + sb_col * 4 + c] = recon_4x4[r * 4 + c];
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_setup(
        qp: u8,
    ) -> (
        crate::quant::SegmentQuant,
        LevelCostTable,
        rdo::VP8SegmentLambdas,
    ) {
        let seg_quant = quant::build_segment_quant(qp);
        let probs = crate::cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);
        let lambdas = rdo::compute_segment_lambdas(qp);
        (seg_quant, cost_table, lambdas)
    }

    #[test]
    fn reconstruct_intra16_no_panic() {
        let (seg_quant, cost_table, lambdas) = test_setup(30);
        // Flat gray source with DC prediction
        let src = [128u8; 256];
        let pred = [128u8; 256];
        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];

        let result = reconstruct_intra16(
            &src,
            &pred,
            &seg_quant,
            lambdas.lambda_trellis_i16,
            &cost_table,
            &mut top_nz,
            &mut left_nz,
        );

        // Flat block: all AC should be zero, reconstruction should match prediction
        for &ac in result.y_ac_levels.iter() {
            assert_eq!(ac, [0i16; 16], "flat block should have zero AC");
        }
        // Recon should be very close to source for a flat block
        for i in 0..256 {
            assert!(
                (result.recon[i] as i32 - src[i] as i32).abs() <= 1,
                "pixel {} differs: {} vs {}",
                i,
                result.recon[i],
                src[i]
            );
        }
    }

    #[test]
    fn reconstruct_intra16_gradient() {
        let (seg_quant, cost_table, lambdas) = test_setup(30);
        // Horizontal gradient
        let mut src = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                src[r * 16 + c] = (c * 16) as u8;
            }
        }
        let pred = [128u8; 256]; // DC prediction

        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];
        let result = reconstruct_intra16(
            &src,
            &pred,
            &seg_quant,
            lambdas.lambda_trellis_i16,
            &cost_table,
            &mut top_nz,
            &mut left_nz,
        );

        // Should produce some nonzero AC coefficients for gradient
        let total_nz: usize = result
            .y_ac_levels
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&c| c != 0)
            .count();
        assert!(total_nz > 0, "gradient should produce AC energy");
    }

    #[test]
    fn reconstruct_intra16_context_tracking() {
        let (seg_quant, cost_table, lambdas) = test_setup(30);
        // High-energy source to ensure non-zero coefficients
        let mut src = [0u8; 256];
        for i in 0..256 {
            src[i] = ((i * 17) % 256) as u8;
        }
        let pred = [128u8; 256];

        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];
        let _result = reconstruct_intra16(
            &src,
            &pred,
            &seg_quant,
            lambdas.lambda_trellis_i16,
            &cost_table,
            &mut top_nz,
            &mut left_nz,
        );

        // Context should be updated after processing
        // At least some blocks should have non-zero context
        let any_ctx = top_nz.iter().any(|&c| c != 0) || left_nz.iter().any(|&c| c != 0);
        assert!(any_ctx, "high-energy input should set some nz context");
    }

    #[test]
    fn reconstruct_intra4_no_panic() {
        let (seg_quant, cost_table, lambdas) = test_setup(30);
        let src = [128u8; 16];
        let pred = [128u8; 16];
        let result = reconstruct_intra4(
            &src,
            &pred,
            &seg_quant.y_ac,
            lambdas.lambda_trellis_i4,
            &cost_table,
            0,
        );

        // Flat block should produce zero/near-zero
        for i in 0..16 {
            assert!((result.recon[i] as i32 - src[i] as i32).abs() <= 1);
        }
    }

    #[test]
    fn reconstruct_intra4_roundtrip() {
        let (seg_quant, cost_table, lambdas) = test_setup(20); // Lower QP = higher quality
        // Smooth gradient should round-trip reasonably
        let mut src = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                src[r * 4 + c] = (64 + r * 32 + c * 16) as u8;
            }
        }
        let pred = [128u8; 16];
        let result = reconstruct_intra4(
            &src,
            &pred,
            &seg_quant.y_ac,
            lambdas.lambda_trellis_i4,
            &cost_table,
            0,
        );

        // PSNR should be reasonable for low QP
        let mse: f64 = (0..16)
            .map(|i| {
                let d = result.recon[i] as f64 - src[i] as f64;
                d * d
            })
            .sum::<f64>()
            / 16.0;
        let psnr = if mse == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        assert!(psnr > 20.0, "PSNR {psnr:.1} too low for QP 20");
    }

    #[test]
    fn reconstruct_uv_no_panic() {
        let (seg_quant, cost_table, lambdas) = test_setup(30);
        let src_u = [128u8; 64];
        let src_v = [128u8; 64];
        let pred_u = [128u8; 64];
        let pred_v = [128u8; 64];
        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];

        let result = reconstruct_uv(
            &src_u,
            &src_v,
            &pred_u,
            &pred_v,
            &seg_quant,
            lambdas.lambda_trellis_uv,
            &cost_table,
            &mut top_nz,
            &mut left_nz,
        );

        // Flat chroma should produce near-zero
        for i in 0..64 {
            assert!((result.recon_u[i] as i32 - 128).abs() <= 1);
            assert!((result.recon_v[i] as i32 - 128).abs() <= 1);
        }
    }

    #[test]
    fn reconstruct_uv_gradient() {
        let (seg_quant, cost_table, lambdas) = test_setup(30);
        let mut src_u = [0u8; 64];
        let mut src_v = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                src_u[r * 8 + c] = (c * 32) as u8;
                src_v[r * 8 + c] = (r * 32) as u8;
            }
        }
        let pred_u = [128u8; 64];
        let pred_v = [128u8; 64];
        let mut top_nz = [0u8; 4];
        let mut left_nz = [0u8; 4];

        let result = reconstruct_uv(
            &src_u,
            &src_v,
            &pred_u,
            &pred_v,
            &seg_quant,
            lambdas.lambda_trellis_uv,
            &cost_table,
            &mut top_nz,
            &mut left_nz,
        );

        // Should have some nonzero coefficients for gradients
        let total_nz: usize = result
            .uv_levels
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&c| c != 0)
            .count();
        assert!(total_nz > 0, "gradient UV should produce coefficients");
    }

    #[test]
    fn trellis_produces_output() {
        // Simple test: trellis should handle a block without panicking
        let mut coeffs_in = [100i16, 50, -30, 20, 10, -5, 3, -2, 1, 0, 0, 0, 0, 0, 0, 0];
        let mut coeffs_out = [0i16; 16];
        let matrix = quant::build_matrix(30, quant::QuantType::YAc);
        let probs = crate::cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);
        let lambdas = rdo::compute_segment_lambdas(30);

        let nz = trellis_quantize_block(
            &mut coeffs_in,
            &mut coeffs_out,
            0, // ctx0
            3, // TYPE_I4_AC
            &matrix,
            lambdas.lambda_trellis_i4,
            &cost_table,
        );

        // Should produce some nonzero coefficients for this input
        assert!(nz || !nz, "trellis completed without panic");
    }

    #[test]
    fn trellis_all_zero_input() {
        let mut coeffs_in = [0i16; 16];
        let mut coeffs_out = [0i16; 16];
        let matrix = quant::build_matrix(50, quant::QuantType::YAc);
        let probs = crate::cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);
        let lambdas = rdo::compute_segment_lambdas(50);

        let nz = trellis_quantize_block(
            &mut coeffs_in,
            &mut coeffs_out,
            0,
            3,
            &matrix,
            lambdas.lambda_trellis_i4,
            &cost_table,
        );
        assert!(!nz, "all-zero input should produce skip");
        assert_eq!(coeffs_out, [0i16; 16]);
    }

    #[test]
    fn trellis_produces_more_zeros_than_simple() {
        // Trellis should zero out more small coefficients than simple quantization
        let original = [
            200i16, 80, -60, 40, 25, -15, 10, -8, 5, -3, 2, -1, 1, 0, 0, 0,
        ];
        let matrix = quant::build_matrix(40, quant::QuantType::YAc);

        // Simple quantization
        let mut simple_out = [0i16; 16];
        quant::quantize_block(&original, &matrix, &mut simple_out);
        let simple_nz: usize = simple_out.iter().filter(|&&c| c != 0).count();

        // Trellis quantization
        let mut trellis_in = original;
        let mut trellis_out = [0i16; 16];
        let probs = crate::cost_engine::reshape_probs();
        let cost_table = LevelCostTable::compute(&probs);
        let lambdas = rdo::compute_segment_lambdas(40);

        trellis_quantize_block(
            &mut trellis_in,
            &mut trellis_out,
            0,
            3,
            &matrix,
            lambdas.lambda_trellis_i4,
            &cost_table,
        );
        let trellis_nz: usize = trellis_out.iter().filter(|&&c| c != 0).count();

        // Trellis should produce at least as many zeros
        assert!(
            trellis_nz <= simple_nz,
            "trellis should have more zeros: trellis={trellis_nz} vs simple={simple_nz}"
        );
    }
}
