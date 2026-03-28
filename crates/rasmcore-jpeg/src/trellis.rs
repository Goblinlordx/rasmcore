//! Trellis quantization — rate-distortion optimized JPEG compression.
//!
//! Uses dynamic programming (Viterbi algorithm) to find the set of
//! quantized DCT coefficients that minimizes rate-distortion cost:
//!
//!   J = D + lambda * R
//!
//! where:
//!   D = sum of squared errors (original - dequantized)^2
//!   R = estimated Huffman bit count for the coefficient sequence
//!   lambda = controls quality-size tradeoff
//!
//! Achieves 5-15% file size reduction at the same visual quality
//! compared to simple rounding quantization.
//!
//! Reference: mozjpeg trellis quantization (BSD-licensed algorithm).

use crate::entropy::{
    AC_CHROMA_CODE_LENGTHS, AC_LUMA_CODE_LENGTHS, DC_CHROMA_CODE_LENGTHS, DC_LUMA_CODE_LENGTHS,
};
use crate::quantize::ZIGZAG;

/// Maximum number of candidate values per coefficient position.
const MAX_CANDIDATES: usize = 3;

/// Trellis quantize a single 8x8 DCT block.
///
/// Input: raw (unquantized) DCT coefficients in zigzag order, quantization table.
/// Output: optimally quantized coefficients in zigzag order.
///
/// The lambda parameter controls quality-size tradeoff:
///   - Higher lambda → more aggressive quantization → smaller files
///   - Lower lambda → less aggressive → larger files, better quality
///
/// A good default: `lambda = (quant_step / 2.0)^2` where quant_step
/// is the average quantization step size.
pub fn trellis_quantize(
    dct_coeffs: &[i32; 64],
    quant_table: &[u16; 64],
    lambda: f64,
    is_luma: bool,
) -> [i16; 64] {
    trellis_quantize_with_codes(dct_coeffs, quant_table, lambda, is_luma, None)
}

/// Trellis quantize with optional custom AC code lengths for rate estimation.
///
/// When `custom_ac_codes` is provided, uses those code lengths instead of the
/// static ITU-T Annex K tables. This enables more accurate rate estimation
/// when optimized Huffman tables are available.
pub fn trellis_quantize_with_codes(
    dct_coeffs: &[i32; 64],
    quant_table: &[u16; 64],
    lambda: f64,
    is_luma: bool,
    custom_ac_codes: Option<&[u8; 256]>,
) -> [i16; 64] {
    let ac_code_lengths = match custom_ac_codes {
        Some(codes) => codes,
        None => {
            if is_luma {
                &AC_LUMA_CODE_LENGTHS
            } else {
                &AC_CHROMA_CODE_LENGTHS
            }
        }
    };

    // Pre-compute per-coefficient weight: 1/Q[i]^2 (normalized distortion)
    let mut coeff_weight = [0.0f64; 64];
    for i in 0..64 {
        let q = quant_table[i] as f64;
        coeff_weight[i] = if q > 0.0 { 1.0 / (q * q) } else { 1.0 };
    }

    let mut output = [0i16; 64];

    // DC coefficient: use simple rounding (trellis only optimizes AC)
    let q0 = quant_table[ZIGZAG[0]] as i32;
    let dc = dct_coeffs[ZIGZAG[0]];
    output[0] = if dc >= 0 {
        ((dc + q0 / 2) / q0) as i16
    } else {
        ((dc - q0 / 2) / q0) as i16
    };

    // AC coefficients: trellis optimization
    trellis_optimize_ac(
        dct_coeffs,
        quant_table,
        lambda,
        ac_code_lengths,
        &coeff_weight,
        &mut output,
    );

    output
}

/// Compute content-adaptive lambda from DCT coefficients and quantization table.
///
/// Uses mozjpeg-aligned formula:
///   block_energy = mean_squared_AC = sum(ac_coeff^2) / 63
///   lambda = 2^14.75 / (2^16.5 + block_energy)
///
/// Flat blocks → high lambda → more zeros → smaller files.
/// Textured blocks → low lambda → preserve detail → better quality.
///
/// The `scale` parameter allows user-adjustable quality-size tradeoff:
/// - scale = 1.0 matches mozjpeg default
/// - scale > 1.0 more aggressive (smaller files)
/// - scale < 1.0 less aggressive (higher quality)
pub fn adaptive_lambda(dct_coeffs: &[i32; 64], scale: f64) -> f64 {
    // Compute block energy: mean squared AC coefficient
    let mut energy: f64 = 0.0;
    for pos in 1..64 {
        let c = dct_coeffs[ZIGZAG[pos]] as f64;
        energy += c * c;
    }
    let mean_sq = energy / 63.0;

    // mozjpeg formula: 2^14.75 / (2^16.5 + block_energy)
    // 2^14.75 ≈ 27554.5, 2^16.5 ≈ 92681.9
    let lambda = 27554.5 / (92681.9 + mean_sq);

    lambda * scale
}

/// Compute the default lambda parameter.
///
/// Returns fixed 1.0 for backwards compatibility. Use [`adaptive_lambda`]
/// for content-adaptive trellis quantization.
pub fn default_lambda(_quant_table: &[u16; 64]) -> f64 {
    1.0
}

/// Estimate Huffman bits for an AC (run, size) symbol pair.
fn estimate_ac_bits(run: u8, size: u8, ac_code_lengths: &[u8; 256]) -> u32 {
    if size == 0 && run == 0 {
        // EOB
        return ac_code_lengths[0x00] as u32;
    }
    let symbol = ((run as usize) << 4) | (size as usize);
    if symbol >= 256 {
        return 32; // Invalid — penalize heavily
    }
    let huff_bits = ac_code_lengths[symbol] as u32;
    if huff_bits == 0 {
        return 32; // Symbol not in standard table — penalize
    }
    // Total bits = Huffman code + magnitude bits
    huff_bits + size as u32
}

/// Estimate bits for a ZRL (16 zeros) symbol.
fn estimate_zrl_bits(ac_code_lengths: &[u8; 256]) -> u32 {
    ac_code_lengths[0xF0] as u32
}

/// Get the magnitude category for a coefficient value.
fn magnitude_category(val: i32) -> u8 {
    let abs_val = val.unsigned_abs();
    if abs_val == 0 {
        return 0;
    }
    32 - abs_val.leading_zeros() as u8
}

/// DP cell with proper backtrack pointers for Viterbi reconstruction.
#[derive(Clone, Copy)]
struct DpCell {
    /// Accumulated RD cost to reach this (position, run_state).
    cost: f64,
    /// The coefficient level chosen at this position.
    level: i16,
    /// The run_state at the PREVIOUS position that led here.
    prev_run: u16,
}

const INF: f64 = f64::MAX / 2.0;

/// Trellis optimization for AC coefficients using Viterbi algorithm
/// with proper backtrack pointers.
///
/// State = zero run length (0..MAX_RUN). At each position, we evaluate
/// candidate levels (zero, round-down, round-up) and propagate the
/// minimum-cost path through the DP lattice.
#[allow(clippy::needless_range_loop)]
fn trellis_optimize_ac(
    dct_coeffs: &[i32; 64],
    quant_table: &[u16; 64],
    lambda: f64,
    ac_code_lengths: &[u8; 256],
    coeff_weight: &[f64; 64],
    output: &mut [i16; 64],
) {
    const NUM_AC: usize = 63;
    const MAX_RUN: usize = 64;

    // Generate candidates for each AC position
    let mut candidates: Vec<Vec<i16>> = Vec::with_capacity(NUM_AC);
    for pos in 1..64 {
        let zigzag_pos = ZIGZAG[pos];
        let coeff = dct_coeffs[zigzag_pos];
        let q = quant_table[zigzag_pos] as i32;

        let rounded = if coeff >= 0 {
            (coeff + q / 2) / q
        } else {
            (coeff - q / 2) / q
        };

        let mut cands = Vec::with_capacity(MAX_CANDIDATES);
        cands.push(0); // always consider zero

        if rounded != 0 {
            cands.push(rounded as i16);
        }
        if coeff > 0 && rounded > 1 {
            cands.push((rounded - 1) as i16);
        } else if coeff < 0 && rounded < -1 {
            cands.push((rounded + 1) as i16);
        }
        candidates.push(cands);
    }

    // DP table: dp[pos][run_state] with backtrack pointers
    let mut dp = vec![
        vec![
            DpCell {
                cost: INF,
                level: 0,
                prev_run: 0,
            };
            MAX_RUN
        ];
        NUM_AC
    ];

    // Initialize position 0 (first AC coefficient, zigzag position 1)
    {
        let zigzag_pos = ZIGZAG[1];
        let q = quant_table[zigzag_pos] as i32;
        let original = dct_coeffs[zigzag_pos];

        for &cand in &candidates[0] {
            let dequant = cand as i32 * q;
            // Per-coefficient weighted distortion: SSD * weight[zigzag_pos]
            let dist = ((original - dequant) as f64).powi(2) * coeff_weight[zigzag_pos];

            if cand == 0 {
                let total = dist;
                if total < dp[0][1].cost {
                    dp[0][1] = DpCell {
                        cost: total,
                        level: 0,
                        prev_run: 0,
                    };
                }
            } else {
                let size = magnitude_category(cand as i32);
                let rate = estimate_ac_bits(0, size, ac_code_lengths) as f64;
                let total = dist + lambda * rate;
                if total < dp[0][0].cost {
                    dp[0][0] = DpCell {
                        cost: total,
                        level: cand,
                        prev_run: 0,
                    };
                }
            }
        }
    }

    // Forward Viterbi pass
    for pos in 1..NUM_AC {
        let zigzag_pos = ZIGZAG[pos + 1];
        let q = quant_table[zigzag_pos] as i32;
        let original = dct_coeffs[zigzag_pos];

        for &cand in &candidates[pos] {
            let dequant = cand as i32 * q;
            // Per-coefficient weighted distortion: SSD * weight[zigzag_pos]
            let dist = ((original - dequant) as f64).powi(2) * coeff_weight[zigzag_pos];

            if cand == 0 {
                // Zero: extend run from any previous state
                for prev_run in 0..MAX_RUN.min(pos + 1) {
                    if dp[pos - 1][prev_run].cost >= INF {
                        continue;
                    }
                    let new_run = prev_run + 1;
                    if new_run >= MAX_RUN {
                        continue;
                    }
                    // ZRL cost for runs that cross 16-boundary
                    let zrl_cost = if new_run >= 16 && prev_run < 16 {
                        lambda * estimate_zrl_bits(ac_code_lengths) as f64
                    } else {
                        0.0
                    };
                    let total = dp[pos - 1][prev_run].cost + dist + zrl_cost;
                    if total < dp[pos][new_run].cost {
                        dp[pos][new_run] = DpCell {
                            cost: total,
                            level: 0,
                            prev_run: prev_run as u16,
                        };
                    }
                }
            } else {
                // Non-zero: emit (run, size) symbol, reset run to 0
                for prev_run in 0..MAX_RUN.min(pos + 1) {
                    if dp[pos - 1][prev_run].cost >= INF {
                        continue;
                    }
                    let run = prev_run;
                    let mut extra = 0.0;
                    let mut effective_run = run;
                    while effective_run >= 16 {
                        extra += lambda * estimate_zrl_bits(ac_code_lengths) as f64;
                        effective_run -= 16;
                    }
                    let size = magnitude_category(cand as i32);
                    let rate = estimate_ac_bits(effective_run as u8, size, ac_code_lengths) as f64
                        + extra / lambda.max(1e-10); // normalize extra back
                    let total = dp[pos - 1][prev_run].cost + dist + lambda * rate;
                    if total < dp[pos][0].cost {
                        dp[pos][0] = DpCell {
                            cost: total,
                            level: cand,
                            prev_run: prev_run as u16,
                        };
                    }
                }
            }
        }
    }

    // Find best final state (considering EOB cost)
    let mut best_cost = INF;
    let mut best_run = 0usize;
    for run in 0..MAX_RUN {
        if dp[NUM_AC - 1][run].cost >= INF {
            continue;
        }
        let eob_cost = estimate_ac_bits(0, 0, ac_code_lengths) as f64;
        let total = dp[NUM_AC - 1][run].cost + lambda * eob_cost;
        if total < best_cost {
            best_cost = total;
            best_run = run;
        }
    }

    // Backtrack: follow prev_run pointers to reconstruct optimal path
    let mut run_at = vec![0usize; NUM_AC];
    run_at[NUM_AC - 1] = best_run;
    for pos in (1..NUM_AC).rev() {
        run_at[pos - 1] = dp[pos][run_at[pos]].prev_run as usize;
    }

    // Extract levels from the optimal path
    for pos in 0..NUM_AC {
        output[pos + 1] = dp[pos][run_at[pos]].level;
    }
}

// ─── DC Trellis ─────────────────────────────────────────────────────────────
//
// Optimizes DC coefficient values across all blocks in raster order,
// considering DPCM differential encoding cost. Runs as a post-pass after
// per-block AC trellis optimization.

/// Optimize DC coefficients across a sequence of blocks using DPCM-aware DP.
///
/// For each block, evaluates 3 candidate DC values (round-down, round-exact, round-up)
/// and selects the combination that minimizes total `distortion + lambda * DPCM_bits`
/// across all blocks in sequence.
///
/// `blocks` are modified in-place (only position 0 = DC coefficient changes).
/// `dct_coeffs` are the original unquantized DCT blocks (for distortion computation).
/// `quant_table` provides the DC quantizer step.
pub fn dc_trellis_pass(
    blocks: &mut [[i16; 64]],
    dct_coeffs: &[[i32; 64]],
    quant_table: &[u16; 64],
    lambda: f64,
    is_luma: bool,
) {
    if blocks.is_empty() {
        return;
    }

    let dc_code_lengths = if is_luma {
        &DC_LUMA_CODE_LENGTHS
    } else {
        &DC_CHROMA_CODE_LENGTHS
    };
    let q0 = quant_table[ZIGZAG[0]] as i32;
    if q0 == 0 {
        return;
    }

    let n = blocks.len();

    // Generate DC candidates for each block
    let mut candidates: Vec<Vec<i16>> = Vec::with_capacity(n);
    for i in 0..n {
        let orig_dc = dct_coeffs[i][ZIGZAG[0]];
        let rounded = if orig_dc >= 0 {
            (orig_dc + q0 / 2) / q0
        } else {
            (orig_dc - q0 / 2) / q0
        } as i16;

        let mut cands = vec![rounded];
        if rounded > i16::MIN + 1 {
            cands.push(rounded - 1);
        }
        if rounded < i16::MAX {
            cands.push(rounded + 1);
        }
        candidates.push(cands);
    }

    // DP: dp[candidate_idx] = (cost, backtrack_candidate_idx_for_prev_block)
    // We track the best cost for each candidate of the current block
    let max_cands = 3;

    let mut prev_costs: Vec<f64> = vec![INF; max_cands];
    let mut backtrack: Vec<Vec<usize>> = Vec::with_capacity(n); // [block][candidate] -> prev_candidate

    // Initialize first block (DPCM predictor = 0)
    let mut first_bt = vec![0usize; candidates[0].len()];
    for (ci, &cand) in candidates[0].iter().enumerate() {
        let orig_dc = dct_coeffs[0][ZIGZAG[0]];
        let dequant = cand as i32 * q0;
        let dist = ((orig_dc - dequant) as f64).powi(2);
        let diff = cand as i32; // DPCM with predictor 0
        let cat = magnitude_category(diff);
        let rate = dc_code_lengths[cat as usize] as f64 + cat as f64;
        prev_costs[ci] = dist + lambda * rate;
        first_bt[ci] = 0;
    }
    backtrack.push(first_bt);

    // Forward pass
    for block_idx in 1..n {
        let mut curr_costs = vec![INF; candidates[block_idx].len()];
        let mut curr_bt = vec![0usize; candidates[block_idx].len()];

        for (ci, &cand) in candidates[block_idx].iter().enumerate() {
            let orig_dc = dct_coeffs[block_idx][ZIGZAG[0]];
            let dequant = cand as i32 * q0;
            let dist = ((orig_dc - dequant) as f64).powi(2);

            // Try each previous candidate
            for (pi, &prev_cand) in candidates[block_idx - 1].iter().enumerate() {
                if prev_costs[pi] >= INF {
                    continue;
                }
                let diff = cand as i32 - prev_cand as i32;
                let cat = magnitude_category(diff);
                let rate = dc_code_lengths[cat as usize] as f64 + cat as f64;
                let total = prev_costs[pi] + dist + lambda * rate;

                if total < curr_costs[ci] {
                    curr_costs[ci] = total;
                    curr_bt[ci] = pi;
                }
            }
        }

        prev_costs = curr_costs;
        backtrack.push(curr_bt);
    }

    // Find best final candidate
    let mut best_ci = 0;
    let mut best_cost = INF;
    for (ci, &cost) in prev_costs.iter().enumerate() {
        if ci < candidates[n - 1].len() && cost < best_cost {
            best_cost = cost;
            best_ci = ci;
        }
    }

    // Backtrack to extract optimal DC values
    let mut chosen = vec![0usize; n];
    chosen[n - 1] = best_ci;
    for i in (1..n).rev() {
        chosen[i - 1] = backtrack[i][chosen[i]];
    }

    // Apply optimized DC values
    for i in 0..n {
        blocks[i][0] = candidates[i][chosen[i]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::{self, QuantPreset};

    fn make_test_block() -> [i32; 64] {
        let mut block = [0i32; 64];
        for i in 0..64 {
            block[i] = ((i as i32 * 17 + 3) % 500) - 250;
        }
        block
    }

    #[test]
    fn trellis_produces_valid_output() {
        let dct = make_test_block();
        let qt = quantize::luma_quant_table(75, QuantPreset::Robidoux, false);
        let lambda = default_lambda(&qt);
        let result = trellis_quantize(&dct, &qt, lambda, true);

        // Output should have DC coefficient
        assert!(result[0] != 0 || dct[ZIGZAG[0]].abs() < qt[ZIGZAG[0]] as i32);

        // All values should be within reasonable range
        for &v in &result {
            assert!(v.abs() < 2048, "coefficient too large: {v}");
        }
    }

    #[test]
    fn trellis_has_more_zeros_than_simple() {
        let dct = make_test_block();
        let qt = quantize::luma_quant_table(50, QuantPreset::Robidoux, false);
        let lambda = default_lambda(&qt);

        // Simple quantization
        let mut simple = [0i16; 64];
        let mut dct_arr = [0i32; 64];
        for i in 0..64 {
            dct_arr[i] = dct[i];
        }
        quantize::quantize(&dct_arr, &qt, &mut simple);

        // Trellis quantization
        let trellis = trellis_quantize(&dct, &qt, lambda, true);

        let simple_zeros = simple.iter().filter(|&&v| v == 0).count();
        let trellis_zeros = trellis.iter().filter(|&&v| v == 0).count();

        // Trellis should produce at least as many zeros (often more)
        assert!(
            trellis_zeros >= simple_zeros,
            "trellis should have >= zeros: trellis={trellis_zeros}, simple={simple_zeros}"
        );
    }

    #[test]
    fn trellis_dc_matches_simple_before_dc_pass() {
        let dct = make_test_block();
        let qt = quantize::luma_quant_table(75, QuantPreset::AnnexK, false);
        let lambda = default_lambda(&qt);

        let mut simple = [0i16; 64];
        quantize::quantize(&dct, &qt, &mut simple);

        let trellis = trellis_quantize(&dct, &qt, lambda, true);

        // DC should match before DC trellis pass (AC trellis doesn't touch DC)
        assert_eq!(trellis[0], simple[0], "DC should match simple quantization");
    }

    #[test]
    fn dc_trellis_produces_valid_output() {
        let qt = quantize::luma_quant_table(50, QuantPreset::Robidoux, false);
        let lambda = default_lambda(&qt);

        // Create 4 blocks with varied DC values
        let mut dct_blocks = Vec::new();
        let mut quant_blocks = Vec::new();
        for i in 0..4 {
            let mut dct = [0i32; 64];
            dct[ZIGZAG[0]] = (i as i32 * 100) + 50; // DC values: 50, 150, 250, 350
            dct_blocks.push(dct);
            let zz = trellis_quantize(&dct, &qt, lambda, true);
            quant_blocks.push(zz);
        }

        let dc_before: Vec<i16> = quant_blocks.iter().map(|b| b[0]).collect();
        dc_trellis_pass(&mut quant_blocks, &dct_blocks, &qt, lambda, true);
        let dc_after: Vec<i16> = quant_blocks.iter().map(|b| b[0]).collect();

        // DC values should be within ±1 of original (candidates are round±1)
        for i in 0..4 {
            assert!(
                (dc_after[i] - dc_before[i]).abs() <= 1,
                "block {i}: DC changed from {} to {} (max ±1 allowed)",
                dc_before[i],
                dc_after[i]
            );
        }
    }

    #[test]
    fn lambda_is_constant() {
        let qt_high = quantize::luma_quant_table(90, QuantPreset::Robidoux, false);
        let qt_low = quantize::luma_quant_table(25, QuantPreset::Robidoux, false);

        // Lambda matches mozjpeg default: constant 1.0
        assert_eq!(default_lambda(&qt_high), 1.0);
        assert_eq!(default_lambda(&qt_low), 1.0);
    }

    #[test]
    fn trellis_zero_block_stays_zero() {
        let dct = [0i32; 64];
        let qt = quantize::luma_quant_table(50, QuantPreset::Robidoux, false);
        let lambda = default_lambda(&qt);
        let result = trellis_quantize(&dct, &qt, lambda, true);

        for &v in &result {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn default_lambda_is_positive() {
        let qt = quantize::luma_quant_table(50, QuantPreset::Robidoux, false);
        let lambda = default_lambda(&qt);
        assert!(lambda > 0.0);
    }

    #[test]
    fn magnitude_category_values() {
        assert_eq!(magnitude_category(0), 0);
        assert_eq!(magnitude_category(1), 1);
        assert_eq!(magnitude_category(-1), 1);
        assert_eq!(magnitude_category(2), 2);
        assert_eq!(magnitude_category(3), 2);
        assert_eq!(magnitude_category(4), 3);
        assert_eq!(magnitude_category(7), 3);
        assert_eq!(magnitude_category(255), 8);
        assert_eq!(magnitude_category(-255), 8);
    }

    #[test]
    fn adaptive_lambda_flat_vs_textured() {
        // Flat block: all AC = 0 → high lambda (aggressive zeroing OK)
        let mut flat = [0i32; 64];
        flat[0] = 1000; // DC only
        let lambda_flat = adaptive_lambda(&flat, 1.0);

        // Textured block: AC energy everywhere → low lambda (preserve detail)
        let mut textured = [0i32; 64];
        textured[0] = 1000;
        for i in 1..64 {
            textured[i] = 200;
        }
        let lambda_textured = adaptive_lambda(&textured, 1.0);

        assert!(
            lambda_flat > lambda_textured,
            "flat block lambda ({lambda_flat:.6}) should be > textured ({lambda_textured:.6})"
        );
    }

    #[test]
    fn adaptive_lambda_scale_factor() {
        let block = make_test_block();
        let l1 = adaptive_lambda(&block, 1.0);
        let l2 = adaptive_lambda(&block, 2.0);
        assert!(
            (l2 - 2.0 * l1).abs() < 1e-10,
            "scale=2.0 should double lambda"
        );
    }

    #[test]
    fn adaptive_lambda_positive_range() {
        // Lambda should always be positive for any block content
        let mut block = [0i32; 64];
        assert!(
            adaptive_lambda(&block, 1.0) > 0.0,
            "zero block lambda should be positive"
        );

        for i in 0..64 {
            block[i] = i32::MAX / 100;
        }
        assert!(
            adaptive_lambda(&block, 1.0) > 0.0,
            "huge block lambda should be positive"
        );
    }

    #[test]
    fn trellis_with_custom_codes() {
        let block = make_test_block();
        let qt = quantize::luma_quant_table(75, QuantPreset::Robidoux, false);

        // Standard trellis
        let std_result = trellis_quantize(&block, &qt, 1.0, true);

        // Trellis with same codes as static (should give identical result)
        let custom_result =
            trellis_quantize_with_codes(&block, &qt, 1.0, true, Some(&AC_LUMA_CODE_LENGTHS));

        assert_eq!(
            std_result, custom_result,
            "same codes should give same result"
        );
    }

    #[test]
    fn per_coeff_weighting_zeros_high_frequency() {
        // With per-coefficient 1/Q^2 weighting, high-frequency positions
        // (large Q values) should be more aggressively zeroed
        let mut block = [0i32; 64];
        // Put moderate energy in all positions
        for i in 0..64 {
            block[i] = 100;
        }
        let qt = quantize::luma_quant_table(50, QuantPreset::Robidoux, false);
        let lambda = adaptive_lambda(&block, 1.0);
        let result = trellis_quantize(&block, &qt, lambda, true);

        // Count non-zero coefficients in low-freq (pos 1-10) vs high-freq (pos 50-63)
        let low_nz: usize = result[1..=10].iter().filter(|&&c| c != 0).count();
        let high_nz: usize = result[50..64].iter().filter(|&&c| c != 0).count();

        // High-frequency should have fewer non-zeros (larger Q → more zeroing)
        assert!(
            low_nz >= high_nz,
            "low-freq non-zeros ({low_nz}) should be >= high-freq ({high_nz})"
        );
    }
}
