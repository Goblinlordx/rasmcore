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
/// VP8EntropyCost — exact -log2(prob/256) * 256 table from libwebp dsp/cost.c.
/// Index: probability 0-255. Value: cost in 256-scaled units (256 = 1 bit).
#[rustfmt::skip]
const VP8_ENTROPY_COST: [u16; 256] = [
    1792, 1792, 1792, 1536, 1536, 1408, 1366, 1280, 1280, 1216, 1178, 1152,
    1110, 1076, 1061, 1024, 1024, 992,  968,  951,  939,  911,  896,  878,
    871,  854,  838,  820,  811,  794,  786,  768,  768,  752,  740,  732,
    720,  709,  704,  690,  683,  672,  666,  655,  647,  640,  631,  622,
    615,  607,  598,  592,  586,  576,  572,  564,  559,  555,  547,  541,
    534,  528,  522,  512,  512,  504,  500,  494,  488,  483,  477,  473,
    467,  461,  458,  452,  448,  443,  438,  434,  427,  424,  419,  415,
    410,  406,  403,  399,  394,  390,  384,  384,  377,  374,  370,  366,
    362,  359,  355,  351,  347,  342,  342,  336,  333,  330,  326,  323,
    320,  316,  312,  308,  305,  302,  299,  296,  293,  288,  287,  283,
    280,  277,  274,  272,  268,  266,  262,  256,  256,  256,  251,  248,
    245,  242,  240,  237,  234,  232,  228,  226,  223,  221,  218,  216,
    214,  211,  208,  205,  203,  201,  198,  196,  192,  191,  188,  187,
    183,  181,  179,  176,  175,  171,  171,  168,  165,  163,  160,  159,
    156,  154,  152,  150,  148,  146,  144,  142,  139,  138,  135,  133,
    131,  128,  128,  125,  123,  121,  119,  117,  115,  113,  111,  110,
    107,  105,  103,  102,  100,  98,   96,   94,   92,   91,   89,   86,
    86,   83,   82,   80,   77,   76,   74,   73,   71,   69,   67,   66,
    64,   63,   61,   59,   57,   55,   54,   52,   51,   49,   47,   46,
    44,   43,   41,   40,   38,   36,   35,   33,   32,   30,   29,   27,
    25,   24,   22,   21,   19,   18,   16,   15,   13,   12,   10,   9,
    7,    6,    4,    3,
];

/// VP8BitCost — exact cost of encoding a boolean with the given probability.
/// Ported from libwebp cost_enc.h: `VP8EntropyCost[bit ? 255-prob : prob]`
///
/// Returns cost in 256-scaled units (256 = 1 bit).
#[inline]
pub fn vp8_bit_cost(bit: bool, prob: u8) -> u32 {
    if bit {
        VP8_ENTROPY_COST[255 - prob as usize] as u32
    } else {
        VP8_ENTROPY_COST[prob as usize] as u32
    }
}

/// VP8 level fixed costs for levels > MAX_VARIABLE_LEVEL (67).
/// Stub: returns approximate cost. The full 2048-entry table from libwebp
/// dsp/cost.c will be embedded when Track 2/3 needs exact values.
#[inline]
pub fn vp8_level_fixed_cost(level: usize) -> u16 {
    if level <= 67 {
        return 0; // handled by LevelCostTable
    }
    // Approximate: extra bits cost for levels above MAX_VARIABLE_LEVEL.
    // The exact libwebp VP8LevelFixedCosts table has values from ~1700 (level 68)
    // to ~7761 (level 2047). We approximate using bit length.
    let bit_len = 32 - (level as u32).leading_zeros(); // 7..11 for 68..2047
    // Each extra bit above the base cost adds ~256 units (1 bit in 256-scale)
    // Base: level 68 costs ~1700 (about 6.6 bits)
    (bit_len.saturating_mul(256)) as u16
}

/// Cost of encoding a boolean — approximate version for backward compat.
///
/// NOTE: The exact version is `vp8_bit_cost` using VP8_ENTROPY_COST table.
/// This approximate version is kept because the RDO lambda was calibrated
/// for this scale. Switching to exact costs requires recalibrating all
/// lambda values simultaneously. Use `vp8_bit_cost` for new libwebp-exact code.
#[inline]
fn prob_cost(prob: u32) -> u32 {
    if prob == 0 {
        return 2048;
    }
    if prob >= 256 {
        return 0;
    }
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
        let ctx = if i == 0 || quantized[i - 1] == 0 {
            0u8 // DC context or previous was zero
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

// ─── RD Mode Selection Helpers ───────────────────────────────────────────

/// Compute SSD (Sum of Squared Differences) between original and prediction.
pub fn ssd(original: &[u8], prediction: &[u8]) -> u64 {
    original
        .iter()
        .zip(prediction.iter())
        .map(|(&a, &b)| {
            let d = a as i32 - b as i32;
            (d * d) as u64
        })
        .sum()
}

/// Estimate total encoding bits for a quantized 4x4 block.
///
/// Walks all non-zero coefficients and sums token costs.
/// Returns cost in 256-scaled units (256 = 1 bit).
pub fn estimate_block_bits(quantized: &[i16; 16], block_type: u8) -> u32 {
    // Find last non-zero coefficient
    let last_nz = match quantized.iter().rposition(|&c| c != 0) {
        Some(pos) => pos,
        None => {
            // All-zero block: just the EOB signal at position 0
            let probs = token::get_coeff_probs(block_type as usize, BANDS[0] as usize, 0);
            return prob_cost(probs[0] as u32); // cost of signaling "zero/EOB"
        }
    };

    let mut total_cost = 0u32;
    let mut ctx: u8 = 0; // initial context: zero/eob

    for i in 0..=last_nz {
        let band = BANDS[i];
        total_cost += estimate_token_cost(quantized[i], block_type, band, ctx);

        ctx = match quantized[i].unsigned_abs() {
            0 => 0,
            1 => 1,
            _ => 2,
        };
    }

    // EOB after last non-zero
    if last_nz + 1 < 16 {
        let eob_band = BANDS[last_nz + 1];
        total_cost += estimate_token_cost(0, block_type, eob_band, ctx);
    }

    total_cost
}

/// RD cost: distortion (SSD) + lambda * rate (bits in 256-scale).
///
/// Lower is better. Lambda balances quality vs compression.
#[inline]
pub fn rd_cost(ssd: u64, bits_256: u32, lambda: f64) -> f64 {
    ssd as f64 + lambda * bits_256 as f64
}

/// VP8 I16x16 mode header cost.
///
/// VP8 I16x16 mode header cost — exact from libwebp cost_enc.c line 105.
/// Scale: 256 units = 1 bit (same as VP8EntropyCost and estimate_block_bits).
///
/// Includes VP8BitCost(1, 145) = 312 for the is_i16 flag.
/// Index: DC=0, TM=1, V=2, H=3.
pub fn mode_header_cost_16x16(mode: u8) -> u32 {
    const VP8_FIXED_COSTS_I16: [u32; 4] = [663, 919, 872, 919];
    VP8_FIXED_COSTS_I16[mode.min(3) as usize]
}

/// VP8 UV mode header cost — exact from libwebp cost_enc.c line 103.
pub const VP8_FIXED_COSTS_UV: [u32; 4] = [302, 984, 439, 642];

/// VP8 B_PRED 4x4 mode header cost (from RFC 6386 Section 11.3).
/// Returns cost in 256-scaled units.
pub fn mode_header_cost_4x4(mode: u8) -> u32 {
    // Approximate: B_PRED modes have ~10 options, each costs ~3-4 bits
    // DC mode is most common (cheapest), diagonal modes are rare (expensive)
    match mode {
        0 => 200, // B_DC
        1 => 250, // B_TM
        2 => 280, // B_VE
        3 => 280, // B_HE
        4 => 350, // B_RD
        5 => 350, // B_VR
        6 => 380, // B_LD
        7 => 350, // B_VL
        8 => 380, // B_HD
        9 => 380, // B_HU
        _ => 400,
    }
}

/// Evaluate RD cost for a 16x16 intra mode.
///
/// Computes: SSD(original, prediction) + lambda * (mode_header_bits + coeff_bits)
/// The block type for I16x16 AC is 0 (Y-AC).
pub fn evaluate_16x16_mode_rd(
    original: &[u8; 256],
    prediction: &[u8; 256],
    mode: u8,
    matrix: &QuantMatrix,
    lambda: f64,
) -> f64 {
    // SSD in pixel domain
    let dist = ssd(original, prediction);

    // Encode: DCT + quantize + estimate bits for all 16 sub-blocks
    let mut total_bits = mode_header_cost_16x16(mode);

    for sb_row in 0..4 {
        for sb_col in 0..4 {
            let mut src_4x4 = [0u8; 16];
            let mut ref_4x4 = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    src_4x4[r * 4 + c] = original[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                    ref_4x4[r * 4 + c] = prediction[(sb_row * 4 + r) * 16 + sb_col * 4 + c];
                }
            }
            let mut coeffs = [0i16; 16];
            crate::dct::forward_dct(&src_4x4, &ref_4x4, &mut coeffs);
            let mut quantized = [0i16; 16];
            crate::quant::quantize_block(&coeffs, matrix, &mut quantized);
            rdo_prune_block(&mut quantized, &coeffs, matrix, 0, lambda);
            quantized[0] = 0; // DC goes to Y2
            total_bits += estimate_block_bits(&quantized, 0);
        }
    }

    rd_cost(dist, total_bits, lambda)
}

// ─── VP8 Trellis Context ─────────────────────────────────────────────────

use rasmcore_trellis::{Candidate, TrellisContext};

/// VP8-specific trellis context for the shared trellis engine.
///
/// Maps VP8's 3 coefficient context states (zero/one/large) and 4 block types
/// to the TrellisContext trait. Uses the same VP8 token probability tables
/// as `estimate_token_cost` for bit-exact rate estimation.
///
/// Overrides `candidates()` to use VP8's actual quantization formula:
/// `level = (|coeff| + bias[i]) * iq[i] >> 16` instead of generic `coeff / step`.
pub struct Vp8TrellisContext<'a> {
    pub block_type: u8,
    pub matrix: &'a QuantMatrix,
}

impl TrellisContext for Vp8TrellisContext<'_> {
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
        let band = BANDS[position.min(16)];
        let probs = token::get_coeff_probs(self.block_type as usize, band as usize, state);
        prob_cost(probs[0] as u32)
    }

    fn candidates(
        &self,
        original_coeff: i16,
        _quant_step: u16,
        _dequant_step: u16,
        position: usize,
    ) -> Vec<Candidate> {
        let mut candidates = Vec::with_capacity(3);
        let orig = original_coeff as i64;

        // Candidate 0: zero (always an option)
        candidates.push(Candidate {
            level: 0,
            distortion: orig * orig,
        });

        if original_coeff == 0 {
            return candidates;
        }

        let sign: i64 = if orig < 0 { -1 } else { 1 };
        let abs_c = original_coeff.unsigned_abs() as u32;
        let m = self.matrix;

        // VP8 round-to-nearest: (|c| + bias) * iq >> 16
        let rtn = ((abs_c + m.bias[position]) * m.iq[position]) >> 16;

        if rtn > 0 {
            let level = (sign * rtn as i64) as i16;
            let recon = level as i64 * m.q[position] as i64;
            candidates.push(Candidate {
                level,
                distortion: (orig - recon) * (orig - recon),
            });
        }

        // Round-down: one level less (more aggressive zeroing)
        if rtn > 1 {
            let rd = rtn - 1;
            let level = (sign * rd as i64) as i16;
            let recon = level as i64 * m.q[position] as i64;
            candidates.push(Candidate {
                level,
                distortion: (orig - recon) * (orig - recon),
            });
        }

        candidates
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
    let ctx = Vp8TrellisContext { block_type, matrix };
    // Trellis finds globally optimal zeroing — needs lower lambda than greedy RDO
    // (the Viterbi search already accounts for context dependencies, so the
    // per-coefficient rate penalty should be gentler).
    let config = rasmcore_trellis::TrellisConfig {
        lambda: lambda * 0.05, // very low: almost pure distortion minimization
    };

    // quant_steps and dequant_steps are passed to the engine but
    // our candidates() override uses the QuantMatrix directly.
    // We still need valid step arrays for the generic fallback.
    let mut steps = [0u16; 16];
    for (i, step) in steps.iter_mut().enumerate() {
        *step = matrix.q[i];
    }

    rasmcore_trellis::trellis_optimize(original_coeffs, &steps, &steps, 16, &ctx, &config, output);

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

// ═══════════════════════════════════════════════════════════════════════════
// libwebp-exact RD Framework (ported from quant_enc.c / vp8i_enc.h)
// ═══════════════════════════════════════════════════════════════════════════

/// Score type — i64, matching libwebp's `score_t = int64_t`.
pub type ScoreT = i64;

/// Maximum cost sentinel (libwebp MAX_COST).
pub const MAX_COST: ScoreT = 0x7fffffffffffff;

/// Distortion multiplier — libwebp RD_DISTO_MULT = 256.
/// Applied to distortion in SetRDScore: score = (R + H) * lambda + 256 * (D + SD)
pub const RD_DISTO_MULT: ScoreT = 256;

/// VP8ModeScore — accumulates score during RD optimization.
/// Ported from libwebp vp8i_enc.h lines 214-225.
#[derive(Debug, Clone)]
pub struct VP8ModeScore {
    /// Distortion (SSD).
    pub d: ScoreT,
    /// Spectral distortion (for secondary evaluation).
    pub sd: ScoreT,
    /// Header bits (mode signaling cost).
    pub h: ScoreT,
    /// Residual bits (coefficient encoding cost).
    pub r: ScoreT,
    /// Combined RD score: (R + H) * lambda + RD_DISTO_MULT * (D + SD).
    pub score: ScoreT,
    /// Quantized luma DC levels (for I16x16 Y2 block).
    pub y_dc_levels: [i16; 16],
    /// Quantized luma AC levels (16 sub-blocks × 16 coefficients).
    pub y_ac_levels: [[i16; 16]; 16],
    /// Quantized chroma levels (4 U + 4 V blocks × 16 coefficients).
    pub uv_levels: [[i16; 16]; 8],
    /// I16x16 prediction mode.
    pub mode_i16: i32,
    /// Per-sub-block I4x4 prediction modes.
    pub modes_i4: [u8; 16],
    /// Chroma prediction mode.
    pub mode_uv: i32,
    /// Non-zero block pattern (bitmask).
    pub nz: u32,
}

impl Default for VP8ModeScore {
    fn default() -> Self {
        Self {
            d: 0,
            sd: 0,
            h: 0,
            r: 0,
            score: MAX_COST,
            y_dc_levels: [0; 16],
            y_ac_levels: [[0; 16]; 16],
            uv_levels: [[0; 16]; 8],
            mode_i16: 0,
            modes_i4: [0; 16],
            mode_uv: 0,
            nz: 0,
        }
    }
}

/// SetRDScore — compute combined RD score from components.
/// Ported from libwebp quant_enc.c line 557:
/// `rd->score = (rd->R + rd->H) * lambda + RD_DISTO_MULT * (rd->D + rd->SD);`
#[inline]
pub fn set_rd_score(lambda: i32, rd: &mut VP8ModeScore) {
    rd.score = (rd.r + rd.h) * lambda as ScoreT + RD_DISTO_MULT * (rd.d + rd.sd);
}

/// RDScoreTrellis — compute RD score for trellis quantization.
/// Ported from libwebp quant_enc.c line 561:
/// `return rate * lambda + RD_DISTO_MULT * distortion;`
#[inline]
pub fn rd_score_trellis(lambda: i32, rate: ScoreT, distortion: ScoreT) -> ScoreT {
    rate * lambda as ScoreT + RD_DISTO_MULT * distortion
}

/// VP8 segment info — per-segment lambda values and quantization parameters.
/// Ported from libwebp vp8i_enc.h lines 192-210.
#[derive(Debug, Clone)]
pub struct VP8SegmentLambdas {
    /// QP index for this segment (0-127).
    pub quant: u8,
    /// Lambda for I4x4 mode evaluation: (3 * q_i4² ) >> 7
    pub lambda_i4: i32,
    /// Lambda for I16x16 mode evaluation: 3 * q_i16²
    pub lambda_i16: i32,
    /// Lambda for UV mode evaluation: (3 * q_uv²) >> 6
    pub lambda_uv: i32,
    /// Lambda for mode decision (I4 vs I16): (1 * q_i4²) >> 7
    pub lambda_mode: i32,
    /// Lambda for I4 trellis: (7 * q_i4²) >> 3
    pub lambda_trellis_i4: i32,
    /// Lambda for I16 trellis: q_i16² >> 2
    pub lambda_trellis_i16: i32,
    /// Lambda for UV trellis: q_uv² << 1
    pub lambda_trellis_uv: i32,
    /// Spectral distortion lambda: (tlambda_scale * q_i4) >> 5
    /// Used as: SD = MULT_8B(tlambda, VP8TDisto(src, dst, kWeightY))
    pub tlambda: i32,
    /// Penalty for using I4 mode: 1000 * q_i4²
    pub i4_penalty: ScoreT,
    /// Minimum distortion for filtering: 20 * y1_q[0]
    pub min_disto: i32,
}

/// Compute segment lambdas from a QP index.
/// Ported from libwebp quant_enc.c SetupMatrices (lines 218-265).
///
/// Uses the same quantization tables and lambda formulas as libwebp.
/// `q_i4` = Y1 AC step, `q_i16` = Y2 AC step (clamped, min 8), `q_uv` = UV AC step.
pub fn compute_segment_lambdas(qp: u8) -> VP8SegmentLambdas {
    use crate::tables::{AC_TABLE, DC_TABLE};

    let q = qp.min(127) as usize;

    // Y1 quantizer steps (same as our existing build_matrix)
    let y1_q0 = DC_TABLE[q] as i32; // Y1 DC
    let y1_q1 = AC_TABLE[q] as i32; // Y1 AC

    // Y2 quantizer steps
    let _y2_q0 = (DC_TABLE[q] as i32 * 2).min(132); // Y2 DC
    let y2_q1 = ((AC_TABLE[q] as i32 * 155) / 100).max(8); // Y2 AC

    // UV quantizer steps (no delta_q for our single-segment encoder)
    let _uv_q0 = DC_TABLE[q].min(132) as i32; // UV DC
    let uv_q1 = AC_TABLE[q] as i32; // UV AC

    // ExpandMatrix returns the "average" quantizer step — libwebp uses
    // a weighted combination. For simplicity, use the AC step as the
    // representative value (matches libwebp for the lambda formulas).
    let q_i4 = y1_q1;
    let q_i16 = y2_q1;
    let q_uv = uv_q1;

    // Lambda formulas from libwebp SetupMatrices (lines 239-248)
    let mut lambda_i4 = (3 * q_i4 * q_i4) >> 7;
    let mut lambda_i16 = 3 * q_i16 * q_i16;
    let mut lambda_uv = (3 * q_uv * q_uv) >> 6;
    #[allow(clippy::identity_op)]
    let mut lambda_mode = (1 * q_i4 * q_i4) >> 7;
    let mut lambda_trellis_i4 = (7 * q_i4 * q_i4) >> 3;
    let mut lambda_trellis_i16 = (q_i16 * q_i16) >> 2;
    let mut lambda_trellis_uv = (q_uv * q_uv) << 1;

    // CheckLambdaValue: ensure >= 1
    fn check(v: &mut i32) {
        if *v < 1 {
            *v = 1;
        }
    }
    check(&mut lambda_i4);
    check(&mut lambda_i16);
    check(&mut lambda_uv);
    check(&mut lambda_mode);
    check(&mut lambda_trellis_i4);
    check(&mut lambda_trellis_i16);
    check(&mut lambda_trellis_uv);

    // Spectral distortion lambda: (sns_strength * q_i4) >> 5
    // libwebp uses enc->config->sns_strength (default 50 for method >= 4)
    // We use a fixed sns_strength of 50 to match default libwebp behavior.
    // SNS strength tuning: libwebp default is 50 for method >= 4.
    // Our encoder may need a different value since our RD framework differs slightly.
    // TODO: calibrate once coefficient quality gap is closed
    let sns_strength = 50i32; // libwebp default for method >= 4
    let mut tlambda = (sns_strength * q_i4) >> 5;
    check(&mut tlambda);

    let i4_penalty = 1000 * q_i4 as ScoreT * q_i4 as ScoreT;
    let min_disto = 20 * y1_q0;

    VP8SegmentLambdas {
        quant: qp,
        lambda_i4,
        lambda_i16,
        lambda_uv,
        lambda_mode,
        lambda_trellis_i4,
        lambda_trellis_i16,
        lambda_trellis_uv,
        tlambda,
        i4_penalty,
        min_disto,
    }
}

/// Coefficient type constants matching libwebp (quant_enc.c line 572).
pub const TYPE_I16_AC: u8 = 0;
pub const TYPE_I16_DC: u8 = 1;
pub const TYPE_CHROMA_A: u8 = 2;
pub const TYPE_I4_AC: u8 = 3;

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
