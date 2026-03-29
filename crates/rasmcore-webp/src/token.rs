//! VP8 coefficient token tree and probability context (RFC 6386 Section 13).
//!
//! Encodes quantized DCT coefficients using a token tree with
//! band-based probability context. Probabilities are indexed by
//! `[block_type][band][prev_coef_context][tree_node]`.

use crate::boolcoder::BoolWriter;
use crate::tables::ZIGZAG;

/// Total number of coefficient probability update flags in a VP8 frame.
/// 4 block types x 8 coefficient bands x 3 complexity contexts x 11 tree nodes
pub const COEFF_PROB_UPDATE_COUNT: usize = 4 * 8 * 3 * 11;

/// Band index for each coefficient position (0-16) in scan order.
/// Position 16 is a sentinel for the "next coefficient" context of position 15.
/// Source: libvpx vp8/common/entropy.c — vp8_coef_bands[]
const BANDS: [u8; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

/// Default coefficient probabilities for VP8 token decoding.
///
/// Source: libvpx `vp8/common/default_coef_probs.h` (kDefaultCoefProbs).
/// BSD-3-Clause licensed. Original copyright: The WebM Project Authors.
///
/// Layout: `[4 block types][8 coeff bands][3 prev coef contexts][11 entropy nodes]`
/// Flattened to 1056 entries. Access: `probs[type*264 + band*33 + ctx*11 + node]`
///
/// Block types: 0=Y-AC (after Y2), 1=Y2 (DC transform), 2=UV, 3=Y-with-DC
#[rustfmt::skip]
/// Get the default coefficient probability table (for self-decode testing).
pub fn default_coeff_probs() -> &'static [u8; 1056] {
    &DEFAULT_COEFF_PROBS
}

const DEFAULT_COEFF_PROBS: [u8; 1056] = [
    // Type 0 — Y after Y2 (AC coefficients of luma when I16x16)
    // Band 0
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, // Band 1
    253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128, 189, 129, 242, 255, 227, 213, 255, 219,
    128, 128, 128, 106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128, // Band 2
    1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128, 181, 133, 238, 254, 221, 234, 255, 154,
    128, 128, 128, 78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128, // Band 3
    1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128, 184, 150, 247, 255, 236, 224, 128, 128,
    128, 128, 128, 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128, // Band 4
    1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128, 170, 139, 241, 252, 236, 209, 255, 255,
    128, 128, 128, 37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128, // Band 5
    1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128, 207, 160, 250, 255, 238, 128, 128, 128,
    128, 128, 128, 102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128, // Band 6
    1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128, 177, 135, 243, 255, 234, 225, 128, 128,
    128, 128, 128, 80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128, // Band 7
    1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128, 246, 1, 255, 128, 128, 128, 128, 128, 128,
    128, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    // Type 1 — Y2 (DC transform of luma)
    // Band 0
    198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62, 131, 45, 198, 221, 172, 176, 220, 157, 252,
    221, 1, 68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128, // Band 1
    1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128, 184, 141, 234, 253, 222, 220, 255, 199,
    128, 128, 128, 81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128, // Band 2
    1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128, 99, 121, 210, 250, 201, 198, 255, 202,
    128, 128, 128, 23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128, // Band 3
    1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128, 109, 178, 241, 255, 231, 245, 255, 255,
    128, 128, 128, 44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128, // Band 4
    1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128, 94, 136, 225, 251, 218, 190, 255, 255,
    128, 128, 128, 22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128, // Band 5
    1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128, 124, 143, 241, 255, 227, 234, 128, 128,
    128, 128, 128, 35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128, // Band 6
    1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128, 121, 141, 235, 255, 225, 227, 255, 255,
    128, 128, 128, 45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128, // Band 7
    1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128, 203, 1, 248, 255, 255, 128, 128, 128, 128,
    128, 128, 137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128,
    // Type 2 — UV (chroma)
    // Band 0
    253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128, 175, 13, 224, 243, 193, 185, 249, 198, 255,
    255, 128, 73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128, // Band 1
    1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128, 239, 90, 244, 250, 211, 209, 255, 255, 128,
    128, 128, 155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128, // Band 2
    1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128, 201, 51, 219, 255, 196, 186, 128, 128, 128,
    128, 128, 69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128, // Band 3
    1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128, 223, 165, 249, 255, 213, 255, 128, 128,
    128, 128, 128, 141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128, // Band 4
    1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128, 190, 36, 230, 255, 236, 255, 128, 128, 128,
    128, 128, 149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128, // Band 5
    1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128, 247, 192, 255, 128, 128, 128, 128, 128,
    128, 128, 128, 240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128, // Band 6
    1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128, 213, 62, 250, 255, 255, 128, 128, 128,
    128, 128, 128, 55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128, // Band 7
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    // Type 3 — Y with DC (luma for non-I16x16 modes)
    // Band 0
    202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255, 126, 38, 182, 232, 169, 184, 228, 174,
    255, 187, 128, 61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128, // Band 1
    1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128, 166, 109, 228, 252, 211, 215, 255, 174,
    128, 128, 128, 39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128, // Band 2
    1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128, 124, 74, 191, 243, 183, 193, 250, 221, 255,
    255, 128, 24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128, // Band 3
    1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128, 149, 150, 226, 252, 216, 205, 255, 171,
    128, 128, 128, 28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128, // Band 4
    1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128, 123, 102, 209, 247, 188, 196, 255, 233,
    128, 128, 128, 20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128, // Band 5
    1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128, 168, 175, 246, 252, 235, 205, 255, 255,
    128, 128, 128, 47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128, // Band 6
    1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128, 141, 84, 213, 252, 201, 202, 255, 219,
    128, 128, 128, 42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128, // Band 7
    1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128, 244, 1, 255, 128, 128, 128, 128, 128, 128,
    128, 128, 238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
];

/// Coefficient update probabilities for VP8 boolean decoder.
///
/// Source: libvpx `vp8/common/coefupdateprobs.h` (vp8_coef_update_probs).
/// BSD-3-Clause licensed. Original copyright: The WebM Project Authors.
///
/// Same layout as DEFAULT_COEFF_PROBS: `[4][8][3][11]` flattened.
#[rustfmt::skip]
pub const COEFF_UPDATE_PROBS: [u8; COEFF_PROB_UPDATE_COUNT] = [
    // Type 0
    255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    176,246,255,255,255,255,255,255,255,255,255, 223,241,252,255,255,255,255,255,255,255,255, 249,253,253,255,255,255,255,255,255,255,255,
    255,244,252,255,255,255,255,255,255,255,255, 234,254,254,255,255,255,255,255,255,255,255, 253,255,255,255,255,255,255,255,255,255,255,
    255,246,254,255,255,255,255,255,255,255,255, 239,253,254,255,255,255,255,255,255,255,255, 254,255,254,255,255,255,255,255,255,255,255,
    255,248,254,255,255,255,255,255,255,255,255, 251,255,254,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,253,254,255,255,255,255,255,255,255,255, 251,254,254,255,255,255,255,255,255,255,255, 254,255,254,255,255,255,255,255,255,255,255,
    255,254,253,255,254,255,255,255,255,255,255, 250,255,254,255,254,255,255,255,255,255,255, 254,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    // Type 1
    217,255,255,255,255,255,255,255,255,255,255, 225,252,241,253,255,255,254,255,255,255,255, 234,250,241,250,253,255,253,254,255,255,255,
    255,254,255,255,255,255,255,255,255,255,255, 223,254,254,255,255,255,255,255,255,255,255, 238,253,254,254,255,255,255,255,255,255,255,
    255,248,254,255,255,255,255,255,255,255,255, 249,254,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,253,255,255,255,255,255,255,255,255,255, 247,254,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,253,254,255,255,255,255,255,255,255,255, 252,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,254,254,255,255,255,255,255,255,255,255, 253,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,254,253,255,255,255,255,255,255,255,255, 250,255,255,255,255,255,255,255,255,255,255, 254,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    // Type 2
    186,251,250,255,255,255,255,255,255,255,255, 234,251,244,254,255,255,255,255,255,255,255, 251,251,243,253,254,255,254,255,255,255,255,
    255,253,254,255,255,255,255,255,255,255,255, 236,253,254,255,255,255,255,255,255,255,255, 251,253,253,254,254,255,255,255,255,255,255,
    255,254,254,255,255,255,255,255,255,255,255, 254,254,254,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,254,255,255,255,255,255,255,255,255,255, 254,254,255,255,255,255,255,255,255,255,255, 254,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 254,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    // Type 3
    248,255,255,255,255,255,255,255,255,255,255, 250,254,252,254,255,255,255,255,255,255,255, 248,254,249,253,255,255,255,255,255,255,255,
    255,253,253,255,255,255,255,255,255,255,255, 246,253,253,255,255,255,255,255,255,255,255, 252,254,251,254,254,255,255,255,255,255,255,
    255,254,252,255,255,255,255,255,255,255,255, 248,254,253,255,255,255,255,255,255,255,255, 253,255,254,254,255,255,255,255,255,255,255,
    255,251,254,255,255,255,255,255,255,255,255, 245,251,254,255,255,255,255,255,255,255,255, 253,253,254,255,255,255,255,255,255,255,255,
    255,251,253,255,255,255,255,255,255,255,255, 252,253,254,255,255,255,255,255,255,255,255, 255,254,255,255,255,255,255,255,255,255,255,
    255,252,255,255,255,255,255,255,255,255,255, 249,255,254,255,255,255,255,255,255,255,255, 255,255,254,255,255,255,255,255,255,255,255,
    255,255,253,255,255,255,255,255,255,255,255, 250,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 254,255,255,255,255,255,255,255,255,255,255, 255,255,255,255,255,255,255,255,255,255,255,
];

/// Look up the 11-element probability slice for a given block type, band, and context.
///
/// Public for use by the RDO module's bitrate estimation.
#[inline]
pub fn get_coeff_probs(block_type: usize, band: usize, ctx: usize) -> &'static [u8] {
    let offset = block_type * 264 + band * 33 + ctx * 11;
    &DEFAULT_COEFF_PROBS[offset..offset + 11]
}

/// Get the flat default coefficient probability table (1056 bytes).
/// Layout: [4 types][8 bands][3 ctx][11 probas] flattened.
pub fn get_default_coeff_probs_flat() -> &'static [u8; 1056] {
    &DEFAULT_COEFF_PROBS
}

/// Encode a block of 16 quantized coefficients into the token partition.
///
/// Uses band-indexed, context-aware probabilities matching the VP8 decoder.
///
/// `plane_type`: 0=Y (AC only, after Y2), 1=Y2, 2=UV
/// Encode a block of 16 quantized coefficients into the token partition.
///
/// Uses the decoder's exact skip optimization: after a ZERO token, the next
/// token starts at tree node 1 (skipping EOB). EOB can only follow a non-zero
/// token or be the first token in the block.
///
/// `plane_type`: 0=Y (AC only, after Y2), 1=Y2, 2=UV
/// `complexity`: inter-MB context (0, 1, or 2) from neighboring blocks
///
/// Returns `true` if any non-zero coefficient was encoded (for context tracking).
pub fn encode_block(
    bw: &mut BoolWriter,
    coeffs: &[i16; 16],
    plane_type: u8,
    complexity: usize,
) -> bool {
    let block_type = plane_type as usize;
    let start = if plane_type == 0 { 1 } else { 0 };

    // Find last non-zero coefficient
    let mut last_nz = -1i32;
    for i in (start..16).rev() {
        if coeffs[ZIGZAG[i]] != 0 {
            last_nz = i as i32;
            break;
        }
    }

    let mut ctx: usize = complexity;
    let mut skip = false; // true after encoding a ZERO token
    let mut has_nonzero = false;

    for i in start..16 {
        let coeff = coeffs[ZIGZAG[i]];
        let band = BANDS[i] as usize;
        let probs = get_coeff_probs(block_type, band, ctx);
        let past_last = i as i32 > last_nz;

        if past_last && coeff == 0 {
            if skip {
                // After ZERO, decoder starts at node 1 — cannot encode EOB.
                // Encode ZERO for remaining positions instead.
                bw.put_bit(probs[1], false); // ZERO at node 1
                ctx = 0; // matches decoder: complexity = 0 after ZERO
                // skip stays true
                continue;
            } else {
                // Can encode EOB (at node 0)
                bw.put_bit(probs[0], false); // EOB
                return has_nonzero;
            }
        }

        if coeff == 0 {
            if skip {
                // Already at node 1, encode ZERO directly
                bw.put_bit(probs[1], false); // ZERO at node 1
            } else {
                // At node 0, skip EOB then encode ZERO
                bw.put_bit(probs[0], true); // not EOB (node 0 → node 1)
                bw.put_bit(probs[1], false); // ZERO at node 1
            }
            skip = true;
            ctx = 0; // matches decoder: complexity = 0 after ZERO
        } else {
            let abs_val = coeff.unsigned_abs() as u32;
            let sign = coeff < 0;
            has_nonzero = true;

            // When skip=false, we start at node 0 (encode not-EOB first)
            // When skip=true, we start at node 1 (skip EOB check)
            if !skip {
                bw.put_bit(probs[0], true); // not EOB (node 0 → node 1)
            }

            if abs_val == 1 {
                bw.put_bit(probs[1], true); // not ZERO (node 1 → node 2)
                bw.put_bit(probs[2], false); // ONE (node 2 left)
            } else if abs_val == 2 {
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], false); // node 3 left → node 4
                bw.put_bit(probs[4], false); // TWO
            } else if abs_val == 3 {
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], false);
                bw.put_bit(probs[4], true);
                bw.put_bit(probs[5], false);
            } else if abs_val == 4 {
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], false);
                bw.put_bit(probs[4], true);
                bw.put_bit(probs[5], true);
            } else if abs_val <= 6 {
                // CAT1: node3→right(→n6), n6→left(→n7), n7→left(=CAT1)
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], true); // → node 6
                bw.put_bit(probs[6], false); // → node 7
                bw.put_bit(probs[7], false); // → CAT1
                bw.put_bit(159, (abs_val - 5) != 0);
            } else if abs_val <= 10 {
                // CAT2: node3→right(→n6), n6→left(→n7), n7→right(=CAT2)
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], true); // → node 6
                bw.put_bit(probs[6], false); // → node 7
                bw.put_bit(probs[7], true); // → CAT2
                let extra = abs_val - 7;
                bw.put_bit(165, (extra >> 1) & 1 != 0);
                bw.put_bit(145, extra & 1 != 0);
            } else if abs_val <= 18 {
                // CAT3: node3→right(→n6), n6→right(→n8), n8→left(→n9), n9→left(=CAT3)
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], true); // → node 6
                bw.put_bit(probs[6], true); // → node 8
                bw.put_bit(probs[8], false); // → node 9
                bw.put_bit(probs[9], false); // → CAT3
                let extra = abs_val - 11;
                bw.put_bit(173, (extra >> 2) & 1 != 0);
                bw.put_bit(148, (extra >> 1) & 1 != 0);
                bw.put_bit(140, extra & 1 != 0);
            } else if abs_val <= 34 {
                // CAT4: ...n8→left(→n9), n9→right(=CAT4)
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], true); // → node 6
                bw.put_bit(probs[6], true); // → node 8
                bw.put_bit(probs[8], false); // → node 9
                bw.put_bit(probs[9], true); // → CAT4
                let extra = abs_val - 19;
                bw.put_bit(176, (extra >> 3) & 1 != 0);
                bw.put_bit(155, (extra >> 2) & 1 != 0);
                bw.put_bit(140, (extra >> 1) & 1 != 0);
                bw.put_bit(135, extra & 1 != 0);
            } else if abs_val <= 66 {
                // CAT5: ...n8→right(→n10), n10→left(=CAT5)
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], true); // → node 6
                bw.put_bit(probs[6], true); // → node 8
                bw.put_bit(probs[8], true); // → node 10
                bw.put_bit(probs[10], false); // → CAT5
                let extra = abs_val - 35;
                bw.put_bit(180, (extra >> 4) & 1 != 0);
                bw.put_bit(157, (extra >> 3) & 1 != 0);
                bw.put_bit(141, (extra >> 2) & 1 != 0);
                bw.put_bit(134, (extra >> 1) & 1 != 0);
                bw.put_bit(130, extra & 1 != 0);
            } else {
                // CAT6: ...n8→right(→n10), n10→right(=CAT6)
                bw.put_bit(probs[1], true);
                bw.put_bit(probs[2], true);
                bw.put_bit(probs[3], true); // → node 6
                bw.put_bit(probs[6], true); // → node 8
                bw.put_bit(probs[8], true); // → node 10
                bw.put_bit(probs[10], true); // → CAT6
                let extra = abs_val - 67;
                let cat6_probs = [254u8, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];
                for (j, &p) in cat6_probs.iter().enumerate() {
                    bw.put_bit(p, (extra >> (10 - j)) & 1 != 0);
                }
            }

            bw.put_bit(128, sign);
            skip = false;
            // Match decoder: complexity = if abs==1 {1} else {2}
            ctx = if abs_val == 1 { 1 } else { 2 };
        }
    }
    has_nonzero
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_all_zero_block_is_eob() {
        let mut bw = BoolWriter::new();
        let coeffs = [0i16; 16];
        encode_block(&mut bw, &coeffs, 1, 0);
        let bytes = bw.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_single_dc_coefficient() {
        let mut bw = BoolWriter::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5;
        encode_block(&mut bw, &coeffs, 1, 0);
        let bytes = bw.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_block_with_multiple_nonzero() {
        let mut bw = BoolWriter::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 10;
        coeffs[1] = -3;
        coeffs[4] = 1;
        encode_block(&mut bw, &coeffs, 2, 0);
        let bytes = bw.finish();
        assert!(bytes.len() > 1);
    }

    #[test]
    fn probs_table_has_correct_size() {
        assert_eq!(DEFAULT_COEFF_PROBS.len(), 1056);
        assert_eq!(COEFF_UPDATE_PROBS.len(), 1056);
    }

    #[test]
    fn probs_lookup_valid_ranges() {
        // Verify get_probs doesn't panic for all valid indices
        for bt in 0..4 {
            for band in 0..8 {
                for ctx in 0..3 {
                    let probs = get_coeff_probs(bt, band, ctx);
                    assert_eq!(probs.len(), 11);
                    // All probabilities should be in [1, 255]
                    // (128 is valid as a "don't care" sentinel)
                }
            }
        }
    }

    #[test]
    fn band_table_correct() {
        assert_eq!(BANDS[0], 0); // DC
        assert_eq!(BANDS[1], 1);
        assert_eq!(BANDS[4], 6); // position 4 maps to band 6 (not 4!)
        assert_eq!(BANDS[15], 7);
        assert_eq!(BANDS[16], 0); // sentinel
    }
}
