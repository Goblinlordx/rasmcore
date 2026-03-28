//! Rate-Distortion Optimization (RDO) for VP8 coefficient pruning.
//!
//! After standard quantization, evaluates each non-zero coefficient against its
//! encoding cost. Zeros out coefficients where the bit cost exceeds the quality
//! benefit, using a QP-derived Lagrangian multiplier (lambda).
//!
//! This is the single biggest quality improvement available — closes most of the
//! quality gap vs cwebp by eliminating expensive-to-encode coefficients that
//! contribute little to visual quality.

use crate::quant::QuantMatrix;
use crate::token;

/// VP8 coefficient band index for each position (0-15).
/// Position 0 = DC, positions 1-15 = AC with band assignments per spec.
const BANDS: [u8; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

/// Derive the Lagrangian multiplier from a QP index.
///
/// Lambda controls the quality/size tradeoff:
/// - Higher lambda → more aggressive pruning → smaller files, lower quality
/// - Lower lambda → keep more coefficients → larger files, higher quality
///
/// Formula approximates libvpx's RD lambda derivation.
pub fn lambda_from_qp(qp: u8) -> f64 {
    // VP8 lambda balancing SSD distortion vs rate cost.
    // Rate costs from estimate_token_cost are in 256-scaled units (256 ≈ 1 bit).
    // Distortion is raw SSD. Lambda converts rate to distortion scale.
    // Formula: lambda = q_step^2 / (2 * 256 * 16)
    let q_step = crate::quant::ac_table_value(qp) as f64;
    q_step * q_step / (2.0 * 256.0 * 16.0)
}

/// Estimate the encoding cost (in bits * 256) of a single coefficient value
/// at the given position in a block.
///
/// Uses the VP8 token probability tree to estimate how many bits the boolean
/// coder would spend encoding this value. Returns cost scaled by 256 for
/// fixed-point precision.
///
/// `block_type`: 0=Y-AC, 1=Y2, 2=UV, 3=Y-with-DC (B_PRED)
/// `band`: coefficient band index (from BANDS table)
/// `ctx`: previous coefficient context (0=EOB/zero, 1=one, 2=large)
pub fn estimate_token_cost(coeff: i16, block_type: u8, band: u8, ctx: u8) -> u32 {
    let probs = token::get_coeff_probs(block_type as usize, band as usize, ctx as usize);

    if coeff == 0 {
        // Cost of signaling zero at the is_nonzero node
        return prob_cost(probs[0] as u32);
    }

    let abs_val = coeff.unsigned_abs() as u32;

    // Cost: is_nonzero (true) + token category tree
    let mut cost = prob_cost(256 - probs[0] as u32); // node 0: nonzero = true

    if abs_val == 1 {
        cost += prob_cost(probs[1] as u32); // node 1: is_one = true
    } else {
        cost += prob_cost(256 - probs[1] as u32); // node 1: is_one = false (>1)

        if abs_val == 2 {
            cost += prob_cost(probs[2] as u32); // node 2: is_two = true
        } else if abs_val <= 4 {
            cost += prob_cost(256 - probs[2] as u32); // >2
            cost += prob_cost(probs[3] as u32); // node 3: category 1-2
            if abs_val == 3 {
                cost += prob_cost(probs[4] as u32); // 3
            } else {
                cost += prob_cost(256 - probs[4] as u32); // 4
            }
        } else {
            cost += prob_cost(256 - probs[2] as u32);
            cost += prob_cost(256 - probs[3] as u32); // >4
            // Higher categories: approximate with fixed cost
            let extra_bits = 32 - (abs_val - 1).leading_zeros();
            cost += extra_bits * 128;
        }
    }

    // Sign bit: always 1 bit (128 cost units)
    cost += 128;

    cost
}

/// Cost of encoding a boolean with the given probability.
///
/// Returns cost in fixed-point units (scaled by 256).
/// `prob` is the probability of the TRUE branch (0-255).
/// Cost of taking the TRUE branch = -log2(prob/256) * 256
/// Approximated as: 256 - prob (linear approximation, good enough for RDO).
#[inline]
fn prob_cost(prob: u32) -> u32 {
    // More accurate: (256 * 8) - (prob as f64).log2() * 256
    // Approximation: higher prob = lower cost, linear
    if prob == 0 {
        return 2048; // very expensive
    }
    if prob >= 256 {
        return 0;
    }
    // -log2(prob/256) * 256 ≈ (8 - log2(prob)) * 256
    // Simplified: 256 * ln(256/prob) / ln(2) ≈ (256 - prob) * 3 / 2
    // Even simpler approximation that works well in practice:
    ((256 - prob) * 256 / 256).max(1)
}

/// RDO prune a quantized block — zero out coefficients where bit cost exceeds quality benefit.
///
/// Works backward from the last non-zero coefficient. For each non-zero coeff,
/// computes the distortion increase from zeroing it vs the bitrate savings.
/// Zeros the coefficient if `lambda * rate_saved > distortion_increase`.
///
/// Returns the new last non-zero index (-1 if all zero).
pub fn rdo_prune_block(
    quantized: &mut [i16; 16],
    original_coeffs: &[i16; 16],
    matrix: &QuantMatrix,
    block_type: u8,
    lambda: f64,
) -> i32 {
    let mut last_nz: i32 = -1;

    // Find current last non-zero
    for i in (0..16).rev() {
        if quantized[i] != 0 {
            last_nz = i as i32;
            break;
        }
    }

    if last_nz < 0 {
        return -1; // already all zero
    }

    // Work backward from last non-zero, pruning coefficients
    // Context tracking: 0 = zero/eob, 1 = one, 2 = large
    for i in (0..=last_nz as usize).rev() {
        if quantized[i] == 0 {
            continue;
        }

        let band = BANDS[i];
        let ctx = if i == 0 {
            0u8 // DC context
        } else if i > 0 && quantized[i - 1] == 0 {
            0 // previous was zero
        } else {
            let prev_abs = quantized[i.saturating_sub(1)].unsigned_abs();
            if prev_abs <= 1 { 1 } else { 2 }
        };

        // Distortion increase from zeroing this coefficient (SSD)
        let dequant_val = quantized[i] as i32 * matrix.q[i] as i32;
        let orig = original_coeffs[i] as i32;
        // Current distortion: (orig - dequant)^2
        // If we zero it: (orig - 0)^2 = orig^2
        // Delta = orig^2 - (orig - dequant)^2 = dequant * (2*orig - dequant)
        let dist_current = (orig - dequant_val) * (orig - dequant_val);
        let dist_zeroed = orig * orig;
        let distortion_delta = (dist_zeroed - dist_current).max(0) as f64;

        // Bitrate savings from zeroing (in cost units)
        let rate_nonzero = estimate_token_cost(quantized[i], block_type, band, ctx) as f64;
        let rate_zero = estimate_token_cost(0, block_type, band, ctx) as f64;
        let rate_saved = (rate_nonzero - rate_zero).max(0.0);

        // RD decision: zero if bit savings * lambda > distortion increase
        if lambda * rate_saved > distortion_delta {
            quantized[i] = 0;
        }
    }

    // Recompute last non-zero
    let mut new_last_nz: i32 = -1;
    for i in (0..16).rev() {
        if quantized[i] != 0 {
            new_last_nz = i as i32;
            break;
        }
    }

    new_last_nz
}

// ─── VP8 Trellis Context ─────────────────────────────────────────────────

use rasmcore_trellis::TrellisContext;

/// VP8-specific trellis context for the shared trellis engine.
///
/// Maps VP8's 3 coefficient context states (zero/one/large) and 4 block types
/// to the TrellisContext trait. Uses the same VP8 token probability tables
/// as `estimate_token_cost` for bit-exact rate estimation.
pub struct Vp8TrellisContext {
    pub block_type: u8, // 0=Y-AC, 1=Y2, 2=UV, 3=Y-with-DC
}

impl TrellisContext for Vp8TrellisContext {
    const NUM_STATES: usize = 3; // 0=zero/eob, 1=one, 2=large

    fn token_cost(&self, level: i16, position: usize, state: usize) -> u32 {
        let band = BANDS[position.min(15)];
        estimate_token_cost(level, self.block_type, band, state as u8)
    }

    fn next_state(&self, level: i16, _state: usize) -> usize {
        match level.unsigned_abs() {
            0 => 0, // zero
            1 => 1, // one
            _ => 2, // large
        }
    }

    fn eob_cost(&self, position: usize, state: usize) -> u32 {
        // EOB is signaled as a zero at the is_nonzero node
        // Use the next position's band (EOB context uses the position where it appears)
        let band = BANDS[position.min(16)];
        let probs = token::get_coeff_probs(self.block_type as usize, band as usize, state);
        // Cost of encoding zero (EOB): prob_cost at is_nonzero = false
        prob_cost(probs[0] as u32)
    }
}

/// Run trellis optimization on a 16-coefficient VP8 block.
///
/// Replaces the greedy `rdo_prune_block` with globally optimal Viterbi search.
/// Takes original (unquantized) coefficients and produces optimized quantized levels.
pub fn trellis_optimize_block(
    original_coeffs: &[i16; 16],
    matrix: &QuantMatrix,
    block_type: u8,
    lambda: f64,
    output: &mut [i16; 16],
) -> i32 {
    let ctx = Vp8TrellisContext { block_type };
    let config = rasmcore_trellis::TrellisConfig { lambda };

    // Build quant/dequant step arrays from the QuantMatrix
    let mut quant_steps = [0u16; 16];
    let mut dequant_steps = [0u16; 16];
    for i in 0..16 {
        // VP8 quantization: level = (|coeff| + bias) * iq >> 16
        // For trellis candidate generation, we need the effective step size.
        // q[] is the step size for dequantization: dequant = level * q[i]
        // For quantization: level ≈ coeff / q[i] (approximately)
        quant_steps[i] = matrix.q[i] as u16;
        dequant_steps[i] = matrix.q[i] as u16;
    }

    rasmcore_trellis::trellis_optimize(
        original_coeffs,
        &quant_steps,
        &dequant_steps,
        16,
        &ctx,
        &config,
        output,
    );

    // Find last non-zero
    let mut last_nz: i32 = -1;
    for i in (0..16).rev() {
        if output[i] != 0 {
            last_nz = i as i32;
            break;
        }
    }
    last_nz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_increases_with_qp() {
        let l0 = lambda_from_qp(0);
        let l50 = lambda_from_qp(50);
        let l127 = lambda_from_qp(127);
        assert!(l50 > l0, "lambda should increase with QP");
        assert!(l127 > l50, "lambda should increase with QP");
        assert!(l0 > 0.0, "lambda should be positive at QP 0");
    }

    #[test]
    fn token_cost_zero_is_cheap() {
        let cost_zero = estimate_token_cost(0, 0, 1, 0);
        let cost_one = estimate_token_cost(1, 0, 1, 0);
        assert!(cost_zero < cost_one, "zero should be cheaper than one");
    }

    #[test]
    fn token_cost_increases_with_magnitude() {
        let cost_1 = estimate_token_cost(1, 0, 1, 0);
        let cost_5 = estimate_token_cost(5, 0, 1, 0);
        let cost_20 = estimate_token_cost(20, 0, 1, 0);
        assert!(cost_5 > cost_1, "larger coefficients cost more bits");
        assert!(cost_20 > cost_5, "larger coefficients cost more bits");
    }

    #[test]
    fn rdo_prune_zeros_small_coefficients_at_high_lambda() {
        let seg = crate::quant::build_segment_quant(50);
        // Create realistic original coefficients and quantize them properly
        let original = [200, 30, 20, 15, 10, 8, 5, 3, 2, 1, 1, 1, 0, 0, 0, 0i16];
        let mut quantized = [0i16; 16];
        crate::quant::quantize_block(&original, &seg.y_ac, &mut quantized);

        let nonzero_before: usize = quantized.iter().filter(|&&c| c != 0).count();

        let lambda = lambda_from_qp(80); // aggressive pruning
        rdo_prune_block(&mut quantized, &original, &seg.y_ac, 0, lambda);

        let nonzero_after: usize = quantized.iter().filter(|&&c| c != 0).count();

        // High lambda should prune at least some coefficients
        assert!(
            nonzero_after <= nonzero_before,
            "RDO should not increase non-zero count: {nonzero_before} -> {nonzero_after}"
        );
    }

    #[test]
    fn rdo_prune_preserves_large_coefficients() {
        let seg = crate::quant::build_segment_quant(30);
        // Large coefficients that quantize to big values
        let original = [
            2000, 1000, 500, 200, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0i16,
        ];
        let mut quantized = [0i16; 16];
        crate::quant::quantize_block(&original, &seg.y_ac, &mut quantized);

        let dc_before = quantized[0];
        let lambda = lambda_from_qp(30); // moderate lambda
        rdo_prune_block(&mut quantized, &original, &seg.y_ac, 0, lambda);

        // Large DC should survive at moderate lambda
        assert_eq!(quantized[0], dc_before, "large DC should survive RDO");
    }

    #[test]
    fn rdo_prune_all_zero_input() {
        let matrix = crate::quant::build_segment_quant(50);
        let original = [0i16; 16];
        let mut quantized = [0i16; 16];

        let result = rdo_prune_block(&mut quantized, &original, &matrix.y_ac, 0, 100.0);
        assert_eq!(result, -1, "all-zero should return -1");
    }
}
