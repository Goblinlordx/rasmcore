//! VP8 coefficient token tree and probability context (RFC 6386 Section 13).
//!
//! Encodes quantized DCT coefficients using a token tree with
//! band-based probability context. Uses default probability tables.

use crate::boolcoder::BoolWriter;
use crate::tables::ZIGZAG;

/// Total number of coefficient probability update flags in a VP8 frame.
/// 4 block types × 8 coefficient bands × 3 complexity contexts × 11 tree nodes
pub const COEFF_PROB_UPDATE_COUNT: usize = 4 * 8 * 3 * 11;

/// Default coefficient probabilities (RFC 6386 Section 13.5).
/// Indexed by [plane_type][band][complexity][node].
/// Plane types: 0=Y (after Y2), 1=Y2, 2=UV, 3=Y (DC included for non-I16x16)
///
/// Simplified: we use a flat default table that works for initial encoding.
/// A production encoder would optimize these per-frame.
const DEFAULT_COEFF_PROBS: [u8; 11] = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128];

/// Band index for each coefficient position (0-15) in scan order.
/// RFC 6386 Section 13.3 — reserved for future probability context refinement.
#[allow(dead_code)]
const BANDS: [u8; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

/// Encode a block of 16 quantized coefficients into the token partition.
///
/// Uses the VP8 token tree (RFC 6386 Section 13.2):
/// - First, signal whether the block has any non-zero coefficients
/// - For each coefficient in zigzag order: encode token + extra bits
///
/// `plane_type`: 0=Y (AC only), 1=Y2, 2=UV
pub fn encode_block(bw: &mut BoolWriter, coeffs: &[i16; 16], plane_type: u8) {
    let start = if plane_type == 0 { 1 } else { 0 }; // Y AC starts at index 1

    // Find last non-zero coefficient
    let mut last_nz = -1i32;
    for i in (start..16).rev() {
        if coeffs[ZIGZAG[i]] != 0 {
            last_nz = i as i32;
            break;
        }
    }

    // Encode coefficients in scan order
    for i in start..16 {
        let coeff = coeffs[ZIGZAG[i]];
        let is_last = i as i32 > last_nz;

        if is_last && coeff == 0 {
            // End of block (EOB)
            encode_token(bw, Token::Eob);
            return;
        }

        if coeff == 0 {
            encode_token(bw, Token::Zero);
        } else {
            let abs_val = coeff.unsigned_abs() as u32;
            let sign = coeff < 0;

            if abs_val == 1 {
                encode_token(bw, Token::One);
            } else if abs_val <= 4 {
                encode_token(bw, Token::Lit(abs_val));
            } else {
                encode_token(bw, Token::Cat(abs_val));
            }
            // Sign bit
            bw.put_bit(128, sign);
        }
    }

    // If we encoded all 16 coefficients without EOB, that's fine —
    // the decoder knows to stop at 16.
}

/// Token categories for the coefficient tree.
enum Token {
    Eob,
    Zero,
    One,
    Lit(u32), // 2, 3, or 4
    Cat(u32), // 5+ (category encoded)
}

/// Encode a single token using the VP8 token tree.
///
/// Tree structure (RFC 6386 Section 13.2):
/// ```text
///           [0]
///          /   \
///        EOB   [1]
///             /   \
///           ZERO  [2]
///                /   \
///              ONE   [3]
///                    /   \
///                 [4]     CAT_3+
///                /   \
///             TWO   [5]
///                  /   \
///               THREE  FOUR
/// ```
fn encode_token(bw: &mut BoolWriter, token: Token) {
    // Use default probabilities for each tree node
    let probs = &DEFAULT_COEFF_PROBS;

    match token {
        Token::Eob => {
            bw.put_bit(probs[0], false); // branch left at node 0 → EOB
        }
        Token::Zero => {
            bw.put_bit(probs[0], true); // branch right at node 0
            bw.put_bit(probs[1], false); // branch left at node 1 → ZERO
        }
        Token::One => {
            bw.put_bit(probs[0], true);
            bw.put_bit(probs[1], true);
            bw.put_bit(probs[2], false); // → ONE
        }
        Token::Lit(v) => {
            bw.put_bit(probs[0], true);
            bw.put_bit(probs[1], true);
            bw.put_bit(probs[2], true);
            bw.put_bit(probs[3], false); // branch left at node 3

            match v {
                2 => {
                    bw.put_bit(probs[4], false); // → TWO
                }
                3 => {
                    bw.put_bit(probs[4], true);
                    bw.put_bit(probs[5], false); // → THREE
                }
                4 => {
                    bw.put_bit(probs[4], true);
                    bw.put_bit(probs[5], true); // → FOUR
                }
                _ => unreachable!(),
            }
        }
        Token::Cat(v) => {
            bw.put_bit(probs[0], true);
            bw.put_bit(probs[1], true);
            bw.put_bit(probs[2], true);
            bw.put_bit(probs[3], true); // branch right at node 3 → categories

            // Category encoding (RFC 6386 Section 13.2)
            if v <= 6 {
                // CAT1 (5-6): 1 extra bit
                bw.put_bit(probs[6], false);
                bw.put_bit(probs[7], false); // CAT1
                bw.put_bit(159, (v - 5) != 0); // extra bit: 0→5, 1→6
            } else if v <= 10 {
                // CAT2 (7-10): 2 extra bits
                bw.put_bit(probs[6], false);
                bw.put_bit(probs[7], true); // CAT2
                let extra = v - 7;
                bw.put_bit(165, (extra >> 1) & 1 != 0);
                bw.put_bit(145, extra & 1 != 0);
            } else if v <= 18 {
                // CAT3 (11-18): 3 extra bits
                bw.put_bit(probs[6], true);
                bw.put_bit(probs[8], false); // CAT3
                let extra = v - 11;
                bw.put_bit(173, (extra >> 2) & 1 != 0);
                bw.put_bit(148, (extra >> 1) & 1 != 0);
                bw.put_bit(140, extra & 1 != 0);
            } else if v <= 34 {
                // CAT4 (19-34): 4 extra bits
                bw.put_bit(probs[6], true);
                bw.put_bit(probs[8], true);
                bw.put_bit(probs[9], false); // CAT4
                let extra = v - 19;
                bw.put_bit(176, (extra >> 3) & 1 != 0);
                bw.put_bit(155, (extra >> 2) & 1 != 0);
                bw.put_bit(140, (extra >> 1) & 1 != 0);
                bw.put_bit(135, extra & 1 != 0);
            } else if v <= 66 {
                // CAT5 (35-66): 5 extra bits
                bw.put_bit(probs[6], true);
                bw.put_bit(probs[8], true);
                bw.put_bit(probs[9], true);
                bw.put_bit(probs[10], false); // CAT5
                let extra = v - 35;
                bw.put_bit(180, (extra >> 4) & 1 != 0);
                bw.put_bit(157, (extra >> 3) & 1 != 0);
                bw.put_bit(141, (extra >> 2) & 1 != 0);
                bw.put_bit(134, (extra >> 1) & 1 != 0);
                bw.put_bit(130, extra & 1 != 0);
            } else {
                // CAT6 (67+): 11 extra bits
                bw.put_bit(probs[6], true);
                bw.put_bit(probs[8], true);
                bw.put_bit(probs[9], true);
                bw.put_bit(probs[10], true); // CAT6
                let extra = v - 67;
                // 11 extra bits, MSB first with category-specific probs
                let cat6_probs = [254u8, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];
                for (j, &p) in cat6_probs.iter().enumerate() {
                    bw.put_bit(p, (extra >> (10 - j)) & 1 != 0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_all_zero_block_is_eob() {
        let mut bw = BoolWriter::new();
        let coeffs = [0i16; 16];
        encode_block(&mut bw, &coeffs, 1);
        let bytes = bw.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_single_dc_coefficient() {
        let mut bw = BoolWriter::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5; // DC coefficient
        encode_block(&mut bw, &coeffs, 1); // Y2 block (starts at index 0)
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
        encode_block(&mut bw, &coeffs, 2); // UV block
        let bytes = bw.finish();
        assert!(bytes.len() > 1, "should produce multiple bytes");
    }
}
