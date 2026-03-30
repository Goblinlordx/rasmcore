//! VP8 Cost Engine — libwebp-exact rate estimation.
//!
//! Ported from libwebp src/enc/cost_enc.c and src/dsp/cost.c.
//! Provides exact token-tree bit costs for VP8 coefficient encoding,
//! matching libwebp's precomputed level cost tables.
//!
//! This module is used by the libwebp-exact encode pipeline (Tracks 2-3).
//! The existing `rdo.rs` functions remain for backward compatibility.

use crate::token;

/// Reshape the flat DEFAULT_COEFF_PROBS[1056] into [4][8][3][11].
#[allow(clippy::needless_range_loop)]
pub fn reshape_probs() -> [[[[u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES] {
    let flat = token::get_default_coeff_probs_flat();
    let mut out = [[[[0u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];
    let mut idx = 0;
    for t in 0..NUM_TYPES {
        for b in 0..NUM_BANDS {
            for c in 0..NUM_CTX {
                for p in 0..NUM_PROBAS {
                    out[t][b][c][p] = flat[idx];
                    idx += 1;
                }
            }
        }
    }
    out
}

// ─── Constants (from libwebp common_dec.h) ────────────────────────────────

/// Number of coefficient types: i16-AC, i16-DC, chroma-AC, i4-AC.
pub const NUM_TYPES: usize = 4;
/// Number of frequency bands for probability context.
pub const NUM_BANDS: usize = 8;
/// Number of context states (0=zero/eob, 1=one, 2=large).
pub const NUM_CTX: usize = 3;
/// Number of probability values per context.
pub const NUM_PROBAS: usize = 11;
/// Last level with variable cost (levels > 67 have constant variable part).
pub const MAX_VARIABLE_LEVEL: usize = 67;

/// VP8 band index for each coefficient position (from libwebp).
pub const VP8_ENC_BANDS: [u8; 16 + 1] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

// ─── VP8LevelCodes (from cost_enc.c) ─────────────────────────────────────

/// Level encoding patterns: [pattern, bits] for levels 1-67.
/// Pattern bits indicate which probability nodes are visited;
/// bits indicate the value at each node.
#[rustfmt::skip]
/// Public alias for use by proba.rs VP8RecordCoeffs.
pub const VP8_LEVEL_CODES_PUB: &[[u16; 2]; MAX_VARIABLE_LEVEL] = &VP8_LEVEL_CODES;

const VP8_LEVEL_CODES: [[u16; 2]; MAX_VARIABLE_LEVEL] = [
    [0x001, 0x000],
    [0x007, 0x001],
    [0x00f, 0x005],
    [0x00f, 0x00d],
    [0x033, 0x003],
    [0x033, 0x003],
    [0x033, 0x023],
    [0x033, 0x023],
    [0x033, 0x023],
    [0x033, 0x023],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x013],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x0d3, 0x093],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x053],
    [0x153, 0x153],
];

// ─── VariableLevelCost (from cost_enc.c) ──────────────────────────────────

/// Compute the variable part of the cost for encoding a given level.
/// Uses VP8LevelCodes pattern + VP8BitCost for each visited probability node.
///
/// Ported from libwebp cost_enc.c VariableLevelCost().
pub fn variable_level_cost(level: usize, probas: &[u8]) -> u32 {
    if level == 0 {
        return 0;
    }
    let idx = (level - 1).min(MAX_VARIABLE_LEVEL - 1);
    let mut pattern = VP8_LEVEL_CODES[idx][0] as u32;
    let mut bits = VP8_LEVEL_CODES[idx][1] as u32;
    let mut cost: u32 = 0;
    let mut i: usize = 2;
    while pattern != 0 {
        if (pattern & 1) != 0 {
            cost += crate::rdo::vp8_bit_cost((bits & 1) != 0, probas[i]);
        }
        bits >>= 1;
        pattern >>= 1;
        i += 1;
    }
    cost
}

// ─── VP8CalculateLevelCosts (from cost_enc.c) ────────────────────────────

/// Precomputed level cost table: level_cost[type][band][ctx][level].
/// For each coefficient type, band, and context, stores the cost of encoding
/// each level from 0 to MAX_VARIABLE_LEVEL.
pub struct LevelCostTable {
    /// level_cost[type][band][ctx][level] — cost in 256-scaled units.
    pub level_cost: [[[[u16; MAX_VARIABLE_LEVEL + 1]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
    /// remapped_costs[type][position][ctx] points into level_cost for the band
    /// corresponding to that coefficient position.
    /// Stored as (type_idx, band_idx) pairs for indexing into level_cost.
    pub remap: [[[u8; NUM_CTX]; 16]; NUM_TYPES], // band index per position
}

impl LevelCostTable {
    /// Compute level cost tables from probability tables.
    /// Ported from libwebp cost_enc.c VP8CalculateLevelCosts().
    #[allow(clippy::needless_range_loop)]
    pub fn compute(coeffs: &[[[[u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES]) -> Self {
        let mut table = Self {
            level_cost: [[[[0u16; MAX_VARIABLE_LEVEL + 1]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            remap: [[[0u8; NUM_CTX]; 16]; NUM_TYPES],
        };

        for ctype in 0..NUM_TYPES {
            for band in 0..NUM_BANDS {
                for ctx in 0..NUM_CTX {
                    let p = &coeffs[ctype][band][ctx];

                    // cost0: cost of signaling "more coefficients" when ctx > 0
                    let cost0 = if ctx > 0 {
                        crate::rdo::vp8_bit_cost(true, p[0]) as u16
                    } else {
                        0
                    };
                    // cost_base: cost of "nonzero" + cost0
                    let cost_base = crate::rdo::vp8_bit_cost(true, p[1]) as u16 + cost0;

                    // level 0: cost of signaling "zero" at the is_nonzero node
                    table.level_cost[ctype][band][ctx][0] =
                        crate::rdo::vp8_bit_cost(false, p[1]) as u16 + cost0;

                    // levels 1..MAX_VARIABLE_LEVEL
                    for v in 1..=MAX_VARIABLE_LEVEL {
                        table.level_cost[ctype][band][ctx][v] =
                            cost_base + variable_level_cost(v, p) as u16;
                    }
                }
            }

            // Build remap: for each position, store the band index
            for n in 0..16 {
                let band = VP8_ENC_BANDS[n];
                for ctx in 0..NUM_CTX {
                    table.remap[ctype][n][ctx] = band;
                }
            }
        }

        table
    }

    /// Get the cost of encoding a given level at a specific position.
    #[inline]
    pub fn get_cost(&self, ctype: usize, pos: usize, ctx: usize, level: usize) -> u16 {
        let band = self.remap[ctype][pos.min(15)][ctx] as usize;
        let v = level.min(MAX_VARIABLE_LEVEL);
        self.level_cost[ctype][band][ctx][v]
    }
}

// ─── VP8GetResidualCost (from cost_enc.c) ─────────────────────────────────

/// Compute the rate (in 256-scaled units) of encoding a coefficient sequence.
///
/// Ported from libwebp cost_enc.c VP8GetResidualCost().
/// Walks the quantized coefficients, summing costs from the precomputed
/// level cost table.
#[allow(clippy::needless_range_loop)]
pub fn get_residual_cost(
    coeffs: &[i16; 16],
    first: usize,
    coeff_type: usize,
    initial_ctx: usize,
    table: &LevelCostTable,
) -> u32 {
    let mut cost: u32 = 0;
    let mut ctx = initial_ctx;

    // Find last nonzero (from position `first` onwards)
    let last = match coeffs[first..].iter().rposition(|&c| c != 0) {
        Some(pos) => (first + pos) as i32,
        None => {
            // All zero: just the EOB cost at the first position
            return table.get_cost(coeff_type, first, ctx, 0) as u32;
        }
    };

    for n in first..=last as usize {
        let v = coeffs[n].unsigned_abs() as usize;
        let level = v.min(MAX_VARIABLE_LEVEL);
        cost += table.get_cost(coeff_type, n, ctx, level) as u32;

        // For levels > MAX_VARIABLE_LEVEL, add the fixed cost for the extra bits
        if v > MAX_VARIABLE_LEVEL {
            cost += crate::rdo::vp8_level_fixed_cost(v) as u32;
        }

        // Update context: 0=zero, 1=one, 2=large
        ctx = match v {
            0 => 0,
            1 => 1,
            _ => 2,
        };
    }

    // EOB after last nonzero (if not at position 15)
    if (last as usize) < 15 {
        let eob_pos = last as usize + 1;
        cost += table.get_cost(coeff_type, eob_pos, ctx, 0) as u32;
    }

    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_codes_table_size() {
        assert_eq!(VP8_LEVEL_CODES.len(), MAX_VARIABLE_LEVEL);
        assert_eq!(VP8_LEVEL_CODES.len(), 67);
    }

    #[test]
    fn variable_level_cost_level_1() {
        // Level 1: pattern=0x001, bits=0x000
        // Only visits i=2: pattern bit 0 is set, bits bit 0 is 0
        // cost = VP8BitCost(0, probas[2])
        let probas = [128u8; NUM_PROBAS];
        let cost = variable_level_cost(1, &probas);
        // VP8BitCost(false, 128) = VP8EntropyCost[128] = 256
        assert_eq!(cost, 256, "level 1 with prob=128 should cost 256 (1 bit)");
    }

    #[test]
    fn variable_level_cost_level_0() {
        let probas = [128u8; NUM_PROBAS];
        assert_eq!(variable_level_cost(0, &probas), 0);
    }

    #[test]
    fn level_cost_table_default_probabilities() {
        // Use the default VP8 coefficient probabilities
        let default_probs = reshape_probs();
        let table = LevelCostTable::compute(&default_probs);

        // Verify some basic properties:
        // Level 0 should be cheaper than level 1 (zero is common)
        let cost_zero = table.get_cost(0, 1, 0, 0);
        let cost_one = table.get_cost(0, 1, 0, 1);
        assert!(
            cost_zero < cost_one,
            "zero should be cheaper than one: {cost_zero} vs {cost_one}"
        );

        // Higher levels should cost more
        let cost_5 = table.get_cost(0, 1, 0, 5);
        assert!(
            cost_5 > cost_one,
            "level 5 should cost more than 1: {cost_5} vs {cost_one}"
        );
    }

    #[test]
    fn get_residual_cost_all_zero() {
        let default_probs = reshape_probs();
        let table = LevelCostTable::compute(&default_probs);

        let coeffs = [0i16; 16];
        let cost = get_residual_cost(&coeffs, 0, 0, 0, &table);
        // Should just be the EOB cost at position 0
        assert!(cost > 0, "all-zero block should have nonzero EOB cost");
        assert!(cost < 500, "all-zero block should be cheap: {cost}");
    }

    #[test]
    fn get_residual_cost_single_dc() {
        let default_probs = reshape_probs();
        let table = LevelCostTable::compute(&default_probs);

        let mut coeffs = [0i16; 16];
        coeffs[0] = 5;
        let cost = get_residual_cost(&coeffs, 0, 0, 0, &table);
        // Should include: cost of level 5 at pos 0 + EOB at pos 1
        assert!(cost > 0);
    }

    #[test]
    fn enc_bands_matches_spec() {
        // VP8 band assignments for positions 0-15
        assert_eq!(VP8_ENC_BANDS[0], 0); // DC
        assert_eq!(VP8_ENC_BANDS[1], 1); // AC band 1
        assert_eq!(VP8_ENC_BANDS[2], 2);
        assert_eq!(VP8_ENC_BANDS[3], 3);
        assert_eq!(VP8_ENC_BANDS[15], 7); // last AC
    }
}
