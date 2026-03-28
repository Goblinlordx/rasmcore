//! Codec-agnostic trellis quantization engine.
//!
//! Uses Viterbi dynamic programming to find globally optimal quantization levels
//! across all coefficients in a block, considering inter-coefficient context
//! dependencies. Codec-specific token costs plugged in via [`TrellisContext`] trait.
//!
//! # Architecture
//!
//! The Viterbi algorithm searches over candidate quantization levels at each
//! coefficient position, tracking the optimal cost path through context states.
//! At each position it evaluates: `distortion(level) + lambda * token_cost(level, state)`
//! plus the accumulated cost from the previous position.
//!
//! The codec provides:
//! - Token encoding costs via [`TrellisContext::token_cost`]
//! - Context state transitions via [`TrellisContext::next_state`]
//! - End-of-block costs via [`TrellisContext::eob_cost`]
//!
//! # Usage
//!
//! ```
//! use rasmcore_trellis::{TrellisContext, TrellisConfig, trellis_optimize};
//!
//! // Implement TrellisContext for your codec (VP8, JPEG, HEVC, ...)
//! // Then call trellis_optimize() to get optimal quantization levels.
//! ```

/// Trait for codec-specific token cost models.
///
/// Each codec (VP8, JPEG, HEVC) implements this trait to provide
/// encoding cost information that the Viterbi search uses to evaluate
/// candidate quantization levels.
pub trait TrellisContext {
    /// Number of context states tracked by the codec.
    ///
    /// VP8: 3 (zero, one, large)
    /// JPEG: 16 (zero run length 0-15)
    /// HEVC: ~5 (significance levels)
    const NUM_STATES: usize;

    /// Initial context state at the start of a block.
    fn initial_state(&self) -> usize {
        0
    }

    /// Bit cost (in fixed-point units, 256 = 1 bit) to encode `level` at
    /// `position` given the previous coefficient's context `state`.
    ///
    /// `level` is the quantized coefficient value (can be 0, positive, or negative).
    fn token_cost(&self, level: i16, position: usize, state: usize) -> u32;

    /// Context state after encoding `level`. Used to propagate context
    /// through the Viterbi DP.
    fn next_state(&self, level: i16, state: usize) -> usize;

    /// Cost (in fixed-point units) of the end-of-block (EOB) token at
    /// `position` in context `state`. The encoder signals EOB after the
    /// last non-zero coefficient.
    fn eob_cost(&self, position: usize, state: usize) -> u32;

    /// Generate candidate quantization levels for a coefficient at `position`.
    ///
    /// Override this for codec-specific quantization formulas (e.g., VP8's
    /// bias+iq approach). Default uses generic integer division.
    fn candidates(
        &self,
        original_coeff: i16,
        quant_step: u16,
        dequant_step: u16,
        _position: usize,
    ) -> Vec<Candidate> {
        generate_candidates(original_coeff, quant_step, dequant_step)
    }
}

/// Configuration for the trellis optimizer.
#[derive(Debug, Clone)]
pub struct TrellisConfig {
    /// Lagrangian multiplier balancing distortion vs rate.
    /// Higher = more aggressive (smaller files, lower quality).
    /// Scale: lambda * rate_cost_units should be comparable to SSD distortion.
    pub lambda: f64,
}

/// A candidate quantization level for a single coefficient position.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// Quantized level.
    pub level: i16,
    /// Distortion (SSD) from choosing this level: (original - dequant(level))^2
    pub distortion: i64,
}

/// Generate candidate quantization levels for a coefficient.
///
/// Returns 2-3 candidates: zero, round-down, round-up (if different from round-down).
fn generate_candidates(original_coeff: i16, quant_step: u16, dequant_step: u16) -> Vec<Candidate> {
    let mut candidates = Vec::with_capacity(3);
    let orig = original_coeff as i64;
    let q = quant_step as i64;
    let dq = dequant_step as i64;

    // Candidate 0: zero (always an option)
    candidates.push(Candidate {
        level: 0,
        distortion: orig * orig,
    });

    if q == 0 || original_coeff == 0 {
        return candidates;
    }

    // Round to nearest quantization level
    let sign: i64 = if orig < 0 { -1 } else { 1 };
    let abs_orig = orig.abs();

    // Round-down: truncate toward zero
    let round_down = (abs_orig / q) as i16;
    // Round-up: round away from zero
    let round_up = round_down + 1;

    if round_down > 0 {
        let level = (sign * round_down as i64) as i16;
        let recon = level as i64 * dq;
        candidates.push(Candidate {
            level,
            distortion: (orig - recon) * (orig - recon),
        });
    }

    // Round-up (if different from round-down)
    if round_up as i64 * dq <= 32767 {
        let level = (sign * round_up as i64) as i16;
        let recon = level as i64 * dq;
        candidates.push(Candidate {
            level,
            distortion: (orig - recon) * (orig - recon),
        });
    }

    candidates
}

/// DP state at one position: cost and backtrack info per context state.
#[derive(Clone, Copy)]
struct DpState {
    /// Total RD cost accumulated to reach this state.
    cost: i64,
    /// Backtrack: which state we came from.
    prev_state: u8,
    /// Backtrack: which candidate level was chosen.
    level: i16,
}

const INFINITY_COST: i64 = i64::MAX / 2;

/// Find globally optimal quantization levels using Viterbi dynamic programming.
///
/// # Arguments
/// * `original_coeffs` — original (unquantized) DCT/transform coefficients
/// * `quant_steps` — quantizer step size per position (for candidate generation)
/// * `dequant_steps` — dequantizer step size per position (for distortion computation)
/// * `block_size` — number of coefficients (16 for VP8, 64 for JPEG, etc.)
/// * `ctx` — codec-specific cost model implementing [`TrellisContext`]
/// * `config` — lambda and other settings
/// * `output` — receives the optimized quantized levels
///
/// The algorithm is provably optimal: it finds the minimum-cost quantization
/// given the cost model. Validation against reference encoders (cwebp, mozjpeg)
/// happens in the codec integration tracks.
pub fn trellis_optimize<C: TrellisContext>(
    original_coeffs: &[i16],
    quant_steps: &[u16],
    dequant_steps: &[u16],
    block_size: usize,
    ctx: &C,
    config: &TrellisConfig,
    output: &mut [i16],
) {
    assert!(block_size <= original_coeffs.len());
    assert!(block_size <= quant_steps.len());
    assert!(block_size <= dequant_steps.len());
    assert!(block_size <= output.len());
    assert!(C::NUM_STATES <= 16, "max 16 context states supported");

    let n_states = C::NUM_STATES;

    // Fixed-point lambda: multiply by 256 to stay in i64 integer domain.
    // Rate costs from token_cost() are in 1/256-bit units, so lambda_fp * rate
    // gives distortion-comparable cost without any floating point.
    let lambda_fp: i64 = (config.lambda * 256.0) as i64;

    // DP tables: [position][state]
    // Flat array for cache-friendly sequential access
    let mut dp: Vec<DpState> = vec![
        DpState {
            cost: INFINITY_COST,
            prev_state: 0,
            level: 0,
        };
        (block_size + 1) * n_states
    ];

    // Initialize: position 0 starts from initial_state with cost 0
    let init_state = ctx.initial_state();
    dp[init_state].cost = 0;

    // Pre-compute suffix distortion for EOB evaluation:
    // suffix_dist[i] = sum of original[j]^2 for j in i..block_size
    // Eliminates the inner loop in EOB cost computation.
    let mut suffix_dist = vec![0i64; block_size + 1];
    for i in (0..block_size).rev() {
        suffix_dist[i] = suffix_dist[i + 1] + original_coeffs[i] as i64 * original_coeffs[i] as i64;
    }

    // Best EOB cost seen so far
    let mut best_eob_cost = INFINITY_COST;
    let mut best_eob_pos: usize = 0;
    let mut best_eob_state: usize = init_state;

    // Check if ending immediately (all zero) is best
    let eob_rate = ctx.eob_cost(0, init_state) as i64;
    let eob_at_start = ((lambda_fp * eob_rate) >> 8) + suffix_dist[0];
    if eob_at_start < best_eob_cost {
        best_eob_cost = eob_at_start;
        best_eob_pos = 0;
        best_eob_state = init_state;
    }

    // Pre-allocate candidate arrays (max 3 candidates, padded to 4 for SIMD)
    let mut cand_levels = [0i16; 4];
    let mut cand_dists = [0i64; 4];

    // Forward Viterbi pass
    for pos in 0..block_size {
        let candidates = ctx.candidates(
            original_coeffs[pos],
            quant_steps[pos],
            dequant_steps[pos],
            pos,
        );
        let n_cand = candidates.len();

        // Pack candidates into SIMD-friendly arrays
        for (i, c) in candidates.iter().enumerate() {
            cand_levels[i] = c.level;
            cand_dists[i] = c.distortion;
        }
        // Pad unused slots with infinity distortion
        for i in n_cand..4 {
            cand_levels[i] = 0;
            cand_dists[i] = INFINITY_COST;
        }

        let curr_base = pos * n_states;
        let next_base = (pos + 1) * n_states;

        // Pre-compute rate costs for all (candidate, state) pairs
        // Layout: rate_table[cand_idx * n_states + prev_state]
        let mut rate_table = [0i64; 4 * 16]; // max 4 candidates x 16 states
        let mut next_state_table = [0u8; 4 * 16];
        for ci in 0..n_cand {
            for s in 0..n_states {
                rate_table[ci * n_states + s] = ctx.token_cost(cand_levels[ci], pos, s) as i64;
                next_state_table[ci * n_states + s] = ctx.next_state(cand_levels[ci], s) as u8;
            }
        }

        // DP inner loop: for each candidate, for each prev_state, compute cost
        for ci in 0..n_cand {
            let dist = cand_dists[ci];
            let level = cand_levels[ci];

            for prev_s in 0..n_states {
                let prev_cost = dp[curr_base + prev_s].cost;
                if prev_cost >= INFINITY_COST {
                    continue;
                }

                // Fixed-point RD cost: distortion + (lambda * rate) >> 8
                let rate = rate_table[ci * n_states + prev_s];
                let rd_cost = dist + ((lambda_fp * rate) >> 8);
                let total = prev_cost + rd_cost;

                let next_s = next_state_table[ci * n_states + prev_s] as usize;

                if total < dp[next_base + next_s].cost {
                    dp[next_base + next_s] = DpState {
                        cost: total,
                        prev_state: prev_s as u8,
                        level,
                    };
                }
            }
        }

        // Check EOB after this position using pre-computed suffix distortion
        for s in 0..n_states {
            let cost = dp[next_base + s].cost;
            if cost >= INFINITY_COST {
                continue;
            }

            let eob_rate = ctx.eob_cost(pos + 1, s) as i64;
            let total = cost + ((lambda_fp * eob_rate) >> 8) + suffix_dist[pos + 1];

            if total < best_eob_cost {
                best_eob_cost = total;
                best_eob_pos = pos + 1;
                best_eob_state = s;
            }
        }
    }

    // Backtrack to extract optimal levels
    // First, zero everything after EOB position
    for i in best_eob_pos..block_size {
        output[i] = 0;
    }

    // Backtrack from best_eob_state at best_eob_pos
    let mut state = best_eob_state;
    for pos in (0..best_eob_pos).rev() {
        let base = (pos + 1) * n_states;
        let entry = dp[base + state];
        output[pos] = entry.level;
        state = entry.prev_state as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock context for testing ────────────────────────────────────────

    /// Simple mock: 2 states (zero/nonzero). Non-zero costs 100, zero costs 10.
    /// EOB costs 5. This makes the optimizer prefer zeros unless distortion is high.
    struct MockContext;

    impl TrellisContext for MockContext {
        const NUM_STATES: usize = 2;

        fn token_cost(&self, level: i16, _pos: usize, _state: usize) -> u32 {
            if level == 0 {
                10
            } else {
                100 + level.unsigned_abs() as u32 * 20
            }
        }

        fn next_state(&self, level: i16, _state: usize) -> usize {
            if level == 0 { 0 } else { 1 }
        }

        fn eob_cost(&self, _pos: usize, _state: usize) -> u32 {
            5
        }
    }

    /// Context where zero-runs are rewarded (cheaper when previous was also zero).
    struct RunLengthContext;

    impl TrellisContext for RunLengthContext {
        const NUM_STATES: usize = 2; // 0=after_zero, 1=after_nonzero

        fn token_cost(&self, level: i16, _pos: usize, state: usize) -> u32 {
            if level == 0 {
                if state == 0 { 5 } else { 20 } // cheap to extend a zero run
            } else {
                if state == 0 { 150 } else { 80 } // expensive to break a zero run
            }
        }

        fn next_state(&self, level: i16, _state: usize) -> usize {
            if level == 0 { 0 } else { 1 }
        }

        fn eob_cost(&self, _pos: usize, state: usize) -> u32 {
            if state == 0 { 2 } else { 30 }
        }
    }

    fn run_trellis(
        originals: &[i16],
        q_step: u16,
        ctx: &impl TrellisContext,
        lambda: f64,
    ) -> Vec<i16> {
        let n = originals.len();
        let quant_steps = vec![q_step; n];
        let dequant_steps = vec![q_step; n]; // same for simplicity
        let config = TrellisConfig { lambda };
        let mut output = vec![0i16; n];
        trellis_optimize(
            originals,
            &quant_steps,
            &dequant_steps,
            n,
            ctx,
            &config,
            &mut output,
        );
        output
    }

    // ── Provable optimality tests ───────────────────────────────────────

    #[test]
    fn all_zero_input_produces_all_zero_output() {
        let result = run_trellis(&[0, 0, 0, 0], 10, &MockContext, 1.0);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn large_coefficient_survives() {
        // Original = 1000 with q_step=10 → round_down=100, round_up=101
        // Distortion of zeroing: 1000^2 = 1_000_000
        // Rate cost of keeping: 100 + 100*20 = 2100, lambda*2100 at lambda=1 = 2100
        // 2100 << 1_000_000 → keep it
        let result = run_trellis(&[1000, 0, 0, 0], 10, &MockContext, 1.0);
        assert_ne!(result[0], 0, "large coefficient should survive");
    }

    #[test]
    fn tiny_coefficient_zeroed_at_high_lambda() {
        // Original = 5, q_step=10 → round_down=0, round_up=1
        // Zero distortion: 5^2=25. Round_up distortion: (5-10)^2=25.
        // Zero is free (dist=25, rate=10). Round_up costs rate=120 → 120*lambda.
        // At high lambda, zero wins.
        let result = run_trellis(&[5, 0, 0, 0], 10, &MockContext, 10.0);
        assert_eq!(result[0], 0, "tiny coeff should be zeroed at high lambda");
    }

    #[test]
    fn trellis_finds_zero_run_optimal_path() {
        // RunLengthContext rewards zero-runs. With mixed small coefficients,
        // trellis should prefer zeroing some to create longer runs vs greedy
        // which would keep each independently.
        let originals = [50i16, 8, 8, 50]; // small coeffs at pos 1,2
        let result = run_trellis(&originals, 10, &RunLengthContext, 2.0);

        // Positions 1 and 2 (value 8, q_step 10 → round to 0 or 1) should be
        // zeroed because extending the zero run is cheaper with RunLengthContext
        assert_eq!(
            result[1], 0,
            "trellis should zero pos 1 for run-length benefit"
        );
        assert_eq!(
            result[2], 0,
            "trellis should zero pos 2 for run-length benefit"
        );
        // Positions 0 and 3 (value 50) should survive
        assert_ne!(result[0], 0, "pos 0 should survive");
        assert_ne!(result[3], 0, "pos 3 should survive");
    }

    #[test]
    fn trellis_produces_valid_output() {
        // Trellis should produce quantized values that are valid candidates
        // (either zero, round-down, or round-up of each original coefficient).
        let originals = [50i16, 12, 12, 50];
        let q_step = 10u16;
        let trellis = run_trellis(&originals, q_step, &RunLengthContext, 2.0);

        for (i, &level) in trellis.iter().enumerate() {
            if level == 0 {
                continue; // zero is always valid
            }
            let expected_round = originals[i].abs() / q_step as i16;
            let abs_level = level.abs();
            assert!(
                abs_level == expected_round || abs_level == expected_round + 1,
                "pos {i}: level {level} should be 0, +-{expected_round}, or +-{}",
                expected_round + 1
            );
        }
    }

    #[test]
    fn identity_cost_matches_round_to_nearest() {
        // When all token costs are equal (no context benefit), trellis should
        // produce the same result as round-to-nearest quantization.
        struct FlatContext;
        impl TrellisContext for FlatContext {
            const NUM_STATES: usize = 1;
            fn token_cost(&self, level: i16, _pos: usize, _state: usize) -> u32 {
                if level == 0 { 0 } else { 100 }
            }
            fn next_state(&self, _level: i16, _state: usize) -> usize {
                0
            }
            fn eob_cost(&self, _pos: usize, _state: usize) -> u32 {
                0
            }
        }

        let originals = [45i16, 25, 15, 5];
        let q_step = 10u16;
        let result = run_trellis(&originals, q_step, &FlatContext, 0.5);

        // With flat costs and low lambda, round-to-nearest wins
        // 45/10=4 (recon=40, dist=25), 25/10=2 (recon=20, dist=25)
        // 15/10=1 (recon=10, dist=25), 5/10=0 or 1 (depends on lambda)
        assert_eq!(result[0], 4, "45 should round to 4");
        assert_eq!(result[1], 2, "25 should round to 2");
    }

    #[test]
    fn block_size_16() {
        // VP8-sized block
        let originals = [100, 50, 30, 20, 10, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0i16];
        let result = run_trellis(&originals, 10, &MockContext, 1.0);
        assert_eq!(result.len(), 16);
        assert_ne!(result[0], 0, "DC should survive");
    }

    #[test]
    fn block_size_64() {
        // JPEG-sized block
        let mut originals = [0i16; 64];
        originals[0] = 500;
        originals[1] = 100;
        originals[2] = 50;
        let result = run_trellis(&originals, 10, &MockContext, 1.0);
        assert_eq!(result.len(), 64);
        assert_ne!(result[0], 0, "DC should survive");
    }

    #[test]
    fn negative_coefficients_handled() {
        let originals = [-100i16, 50, -30, 20];
        let result = run_trellis(&originals, 10, &MockContext, 1.0);
        // Negative coefficient should produce negative quantized level
        assert!(
            result[0] < 0,
            "negative original should give negative level"
        );
    }

    #[test]
    fn eob_optimization_zeros_trailing() {
        // Last few coefficients are tiny — trellis should zero them for EOB savings
        let originals = [200, 100, 3, 2, 1, 1, 0, 0i16];
        let result = run_trellis(&originals, 10, &MockContext, 2.0);
        // Trailing tiny coefficients should be zeroed
        let last_nz = result.iter().rposition(|&c| c != 0).unwrap_or(0);
        assert!(
            last_nz <= 2,
            "trailing tiny coefficients should be zeroed, last_nz={last_nz}"
        );
    }
}
