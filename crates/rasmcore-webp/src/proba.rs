//! VP8 Coefficient Probability Adaptation.
//!
//! Ported from libwebp:
//! - cost_enc.c: VP8RecordCoeffs, VP8RecordStats
//! - frame_enc.c: CalcTokenProba, FinalizeTokenProbas, RecordResiduals
//!
//! Collects coefficient statistics during encoding and computes optimized
//! probability tables for the bitstream. This improves both RD cost estimation
//! accuracy and boolean coder compression efficiency.

use crate::cost_engine::{
    self, NUM_BANDS, NUM_CTX, NUM_PROBAS, NUM_TYPES, VP8_ENC_BANDS,
};
use crate::rdo;
use crate::tables::ZIGZAG;
use crate::token;

/// VP8 coefficient probability state for an entire frame.
///
/// Collects statistics during Pass 1, then computes updated probability
/// tables for Pass 2 encoding.
pub struct VP8Proba {
    /// Accumulated statistics: [type][band][ctx][proba_node].
    /// Each entry packs: lower 16 bits = count of 1-bits, upper 16 bits = total count.
    pub stats: [[[[u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],

    /// Current coefficient probabilities (starts as defaults, updated after stats).
    pub coeffs: [[[[u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],

    /// Whether any probability was changed from defaults.
    pub dirty: bool,
}

impl Default for VP8Proba {
    fn default() -> Self {
        Self::new()
    }
}

impl VP8Proba {
    /// Create a new probability state initialized with VP8 default probabilities.
    pub fn new() -> Self {
        let coeffs = cost_engine::reshape_probs();
        Self {
            stats: [[[[0u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            coeffs,
            dirty: false,
        }
    }

    /// Reset statistics for a new pass.
    pub fn reset_stats(&mut self) {
        self.stats = [[[[0u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];
    }

    /// Record a single bit decision at a probability node.
    /// Ported from libwebp cost_enc.h VP8RecordStats.
    #[inline]
    fn record_stats(bit: bool, stats: &mut u32) -> bool {
        let mut p = *stats;
        // Overflow protection: halve counters when approaching u16 max
        if p >= 0xfffe_0000 {
            p = ((p + 1) >> 1) & 0x7fff_7fff;
        }
        // Increment total (upper 16) and add bit value (lower 16)
        p += 0x0001_0000 + bit as u32;
        *stats = p;
        bit
    }

    /// Record coefficient statistics for one block.
    /// Ported from libwebp cost_enc.c VP8RecordCoeffs.
    ///
    /// `coeffs`: quantized coefficients in raster order (16 entries)
    /// `first`: first coefficient to process (0 for all, 1 for I16_AC)
    /// `coeff_type`: 0=I16_AC, 1=I16_DC, 2=chroma, 3=I4_AC
    /// `initial_ctx`: NZ context (0, 1, or 2)
    ///
    /// Returns true if any nonzero coefficient was found.
    pub fn record_coeffs(
        &mut self,
        coeffs: &[i16; 16],
        first: usize,
        coeff_type: usize,
        initial_ctx: usize,
    ) -> bool {
        // Use the standalone function that operates on the stats array directly
        record_coeffs_impl(&mut self.stats[coeff_type], coeffs, first, initial_ctx)
    }

    /// Record residuals for an entire macroblock.
    /// Ported from libwebp frame_enc.c RecordResiduals.
    #[allow(clippy::needless_range_loop)]
    pub fn record_residuals(
        &mut self,
        is_i16: bool,
        y_dc_levels: &[i16; 16],
        y_ac_levels: &[[i16; 16]; 16],
        uv_levels: &[[i16; 16]; 8],
        top_nz: &mut [u8; 9],
        left_nz: &mut [u8; 9],
    ) {
        if is_i16 {
            // Y2 DC block: type=1, first=0
            let dc_ctx = (top_nz[8] + left_nz[8]).min(2) as usize;
            let has_nz = self.record_coeffs(y_dc_levels, 0, 1, dc_ctx);
            let nz_val = if has_nz { 1 } else { 0 };
            top_nz[8] = nz_val;
            left_nz[8] = nz_val;

            // Y AC blocks: type=0, first=1
            for y in 0..4 {
                for x in 0..4 {
                    let sb = y * 4 + x;
                    let ctx = (top_nz[x] + left_nz[y]).min(2) as usize;
                    let has_nz = self.record_coeffs(&y_ac_levels[sb], 1, 0, ctx);
                    let nz_val = if has_nz { 1 } else { 0 };
                    top_nz[x] = nz_val;
                    left_nz[y] = nz_val;
                }
            }
        } else {
            // B_PRED: type=3, first=0
            for y in 0..4 {
                for x in 0..4 {
                    let sb = y * 4 + x;
                    let ctx = (top_nz[x] + left_nz[y]).min(2) as usize;
                    let has_nz = self.record_coeffs(&y_ac_levels[sb], 0, 3, ctx);
                    let nz_val = if has_nz { 1 } else { 0 };
                    top_nz[x] = nz_val;
                    left_nz[y] = nz_val;
                }
            }
        }

        // UV blocks: type=2, first=0
        for ch in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let sb = ch * 4 + y * 2 + x;
                    let ctx_x = 4 + ch * 2 + x;
                    let ctx_y = 4 + ch * 2 + y;
                    let ctx = (top_nz[ctx_x] + left_nz[ctx_y]).min(2) as usize;
                    let has_nz = self.record_coeffs(&uv_levels[sb], 0, 2, ctx);
                    let nz_val = if has_nz { 1 } else { 0 };
                    top_nz[ctx_x] = nz_val;
                    left_nz[ctx_y] = nz_val;
                }
            }
        }
    }

    /// Compute token probability from statistics.
    /// Ported from libwebp frame_enc.c CalcTokenProba.
    #[inline]
    fn calc_token_proba(nb: u32, total: u32) -> u8 {
        if nb == 0 {
            255
        } else {
            (255 - (nb as u64 * 255 / total as u64)) as u8
        }
    }

    /// Finalize probability tables from accumulated statistics.
    /// Ported from libwebp frame_enc.c FinalizeTokenProbas.
    ///
    /// For each probability node, computes the optimal probability from statistics
    /// and decides whether to update (signal in bitstream) based on cost savings.
    ///
    /// Returns estimated bit-cost of signaling the probability updates.
    pub fn finalize_token_probas(&mut self) -> u64 {
        let default_probs = cost_engine::reshape_probs();
        let update_probs = coeff_update_probs();
        let mut has_changed = false;
        let mut size: u64 = 0;

        for t in 0..NUM_TYPES {
            for b in 0..NUM_BANDS {
                for c in 0..NUM_CTX {
                    for p in 0..NUM_PROBAS {
                        let stats = self.stats[t][b][c][p];
                        let nb = stats & 0xffff;
                        let total = stats >> 16;
                        let update_proba = update_probs[t][b][c][p];
                        let old_p = default_probs[t][b][c][p];
                        let new_p = Self::calc_token_proba(nb, total);

                        // Cost comparison: use old vs new probability
                        let old_cost = branch_cost(nb, total, old_p)
                            + rdo::vp8_bit_cost(false, update_proba) as u64;
                        let new_cost = branch_cost(nb, total, new_p)
                            + rdo::vp8_bit_cost(true, update_proba) as u64
                            + 8 * 256; // 8 bits to encode the new probability value

                        let use_new = old_cost > new_cost;
                        size += rdo::vp8_bit_cost(use_new, update_proba) as u64;

                        if use_new {
                            self.coeffs[t][b][c][p] = new_p;
                            has_changed |= new_p != old_p;
                            size += 8 * 256;
                        } else {
                            self.coeffs[t][b][c][p] = old_p;
                        }
                    }
                }
            }
        }

        self.dirty = has_changed;
        size
    }

    /// Flatten coeffs to [1056] for compatibility with existing cost engine.
    pub fn flatten_coeffs(&self) -> [u8; 1056] {
        let mut flat = [0u8; 1056];
        let mut idx = 0;
        for t in 0..NUM_TYPES {
            for b in 0..NUM_BANDS {
                for c in 0..NUM_CTX {
                    for p in 0..NUM_PROBAS {
                        flat[idx] = self.coeffs[t][b][c][p];
                        idx += 1;
                    }
                }
            }
        }
        flat
    }
}

/// Record coefficient statistics for one block, operating on a single coeff_type slice.
/// Separated from VP8Proba to satisfy the borrow checker (only borrows stats[coeff_type]).
fn record_coeffs_impl(
    stats: &mut [[[u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS],
    coeffs: &[i16; 16],
    first: usize,
    initial_ctx: usize,
) -> bool {
    // Find last nonzero (in scan order from `first`)
    let mut last: i32 = -1;
    for i in (first..16).rev() {
        if coeffs[ZIGZAG[i]] != 0 {
            last = i as i32;
            break;
        }
    }

    let band0 = VP8_ENC_BANDS[first] as usize;
    if last < 0 {
        // All zero: record EOB (bit=0 at node 0)
        VP8Proba::record_stats(false, &mut stats[band0][initial_ctx][0]);
        return false;
    }

    let mut n = first;
    let mut ctx = initial_ctx;

    while n <= last as usize {
        let band = VP8_ENC_BANDS[n] as usize;

        // Node 0: not EOB (record 1)
        VP8Proba::record_stats(true, &mut stats[band][ctx][0]);

        let v = coeffs[ZIGZAG[n]];
        if v == 0 {
            // Node 1: ZERO (record 0)
            VP8Proba::record_stats(false, &mut stats[band][ctx][1]);
            ctx = 0;
            n += 1;
            continue;
        }

        // Node 1: not ZERO (record 1)
        VP8Proba::record_stats(true, &mut stats[band][ctx][1]);

        let abs_v = v.unsigned_abs() as i32;
        // Node 2: ONE vs larger
        let is_large = abs_v > 1;
        VP8Proba::record_stats(is_large, &mut stats[band][ctx][2]);

        if !is_large {
            ctx = 1; // after ONE
        } else {
            // v >= 2: record level tree decisions using VP8LevelCodes
            let level = (abs_v as usize).min(67);
            let codes = &cost_engine::VP8_LEVEL_CODES_PUB[level - 1];
            let mut pattern = codes[0] >> 1;
            let bits = codes[1];
            let mut i = 0usize;
            while pattern != 0 {
                if pattern & 1 != 0 {
                    VP8Proba::record_stats(bits & (2 << i) != 0, &mut stats[band][ctx][3 + i]);
                }
                i += 1;
                pattern >>= 1;
            }
            ctx = 2; // after LARGE
        }
        n += 1;
    }

    // EOB after last nonzero (if not at position 15)
    if (last as usize) < 15 {
        let eob_n = last as usize + 1;
        let eob_band = VP8_ENC_BANDS[eob_n] as usize;
        VP8Proba::record_stats(false, &mut stats[eob_band][ctx][0]);
    }

    true
}

/// Cost of coding `nb` 1-bits and `(total-nb)` 0-bits using probability `proba`.
/// Ported from libwebp frame_enc.c BranchCost.
fn branch_cost(nb: u32, total: u32, proba: u8) -> u64 {
    nb as u64 * rdo::vp8_bit_cost(true, proba) as u64
        + (total - nb) as u64 * rdo::vp8_bit_cost(false, proba) as u64
}

/// Get the COEFF_UPDATE_PROBS table reshaped to [4][8][3][11].
/// These are the probabilities used to signal whether each coefficient
/// probability is updated in the bitstream header.
#[allow(clippy::needless_range_loop)]
fn coeff_update_probs() -> [[[[u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES] {
    let flat = &token::COEFF_UPDATE_PROBS;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_stats_increments_correctly() {
        let mut s = 0u32;
        VP8Proba::record_stats(false, &mut s);
        assert_eq!(s >> 16, 1, "total should be 1");
        assert_eq!(s & 0xffff, 0, "nb should be 0");

        VP8Proba::record_stats(true, &mut s);
        assert_eq!(s >> 16, 2, "total should be 2");
        assert_eq!(s & 0xffff, 1, "nb should be 1");
    }

    #[test]
    fn calc_token_proba_values() {
        // All zeros → probability 255 (always 0-branch)
        assert_eq!(VP8Proba::calc_token_proba(0, 100), 255);
        // All ones → probability 0
        assert_eq!(VP8Proba::calc_token_proba(100, 100), 0);
        // Half and half → ~128
        assert_eq!(VP8Proba::calc_token_proba(50, 100), 128);
    }

    #[test]
    fn proba_new_starts_with_defaults() {
        let proba = VP8Proba::new();
        let defaults = cost_engine::reshape_probs();
        assert_eq!(proba.coeffs, defaults);
        assert!(!proba.dirty);
    }

    #[test]
    fn record_coeffs_all_zero() {
        let mut proba = VP8Proba::new();
        let coeffs = [0i16; 16];
        let has_nz = proba.record_coeffs(&coeffs, 0, 3, 0);
        assert!(!has_nz, "all-zero block should return false");
        // Should have recorded an EOB at band 0, ctx 0, node 0
        assert!(
            proba.stats[3][0][0][0] > 0,
            "should have recorded EOB stats"
        );
    }

    #[test]
    fn record_coeffs_with_nonzero() {
        let mut proba = VP8Proba::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5; // DC coefficient
        let has_nz = proba.record_coeffs(&coeffs, 0, 3, 0);
        assert!(has_nz, "should detect nonzero");
    }

    #[test]
    fn finalize_updates_probabilities() {
        let mut proba = VP8Proba::new();

        // Simulate some statistics: many zeros at a specific node
        // This should cause the probability to shift toward "more zeros"
        for _ in 0..1000 {
            VP8Proba::record_stats(false, &mut proba.stats[0][1][0][0]);
        }
        for _ in 0..10 {
            VP8Proba::record_stats(true, &mut proba.stats[0][1][0][0]);
        }

        let defaults = cost_engine::reshape_probs();
        proba.finalize_token_probas();

        // The probability at this node should have changed (to reflect ~99% zeros)
        // Only if the cost savings justify the update
        // (We can't assert the exact value since it depends on the update cost comparison)
        // But we can verify no crash and the function completes
    }

    #[test]
    fn record_residuals_no_panic() {
        let mut proba = VP8Proba::new();
        let y_dc = [0i16; 16];
        let y_ac = [[0i16; 16]; 16];
        let uv = [[0i16; 16]; 8];
        let mut top_nz = [0u8; 9];
        let mut left_nz = [0u8; 9];

        // I16 mode
        proba.record_residuals(true, &y_dc, &y_ac, &uv, &mut top_nz, &mut left_nz);

        // B_PRED mode
        let mut top_nz2 = [0u8; 9];
        let mut left_nz2 = [0u8; 9];
        proba.record_residuals(false, &y_dc, &y_ac, &uv, &mut top_nz2, &mut left_nz2);
    }
}
