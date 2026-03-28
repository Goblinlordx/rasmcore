//! HEVC Dequantization (ITU-T H.265 Section 8.6.3).

use super::scaling_list::ScalingList;

/// Inverse quantization scale factors for qP % 6 (Table 8-16).
const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// Dequantize a block of quantized coefficients in-place (no scaling list).
///
/// Uses the simplified formula when scaling lists are disabled:
/// `d = clip_i16((coeff * scale + offset) >> shift)`
///
/// where `scale = LEVEL_SCALE[qp%6] << (qp/6)` and
/// `shift = bit_depth - 9 + log2_size`.
pub fn dequantize_block(coeffs: &mut [i16], log2_size: u8, qp: i32, bit_depth: u8) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let shift = bit_depth as i32 - 9 + log2_size as i32;

    // scale incorporates the qP/6 left shift
    let scale = (LEVEL_SCALE[qp_rem] as i64) << qp_per;

    if shift > 0 {
        let add = 1i64 << (shift - 1);
        for c in coeffs.iter_mut() {
            if *c != 0 {
                let d = (*c as i64 * scale + add) >> shift;
                *c = d.clamp(-32768, 32767) as i16;
            }
        }
    } else {
        // shift == 0 or negative (shouldn't happen for valid inputs, but handle safely)
        let left = (-shift) as u32;
        for c in coeffs.iter_mut() {
            if *c != 0 {
                let d = (*c as i64 * scale) << left;
                *c = d.clamp(-32768, 32767) as i16;
            }
        }
    }
}

/// Dequantize a block of quantized coefficients in-place using a scaling list.
///
/// `d = clip_i16((coeff * level_scale * sl_factor + offset) >> right_shift << left_shift)`
///
/// The scaling list factor adds 4 extra bits (log2(16)) to the shift baseline.
pub fn dequantize_block_with_scaling_list(
    coeffs: &mut [i16],
    log2_size: u8,
    qp: i32,
    bit_depth: u8,
    scaling_list: &ScalingList,
    matrix_id: u8,
) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let scale = LEVEL_SCALE[qp_rem] as i64;
    let size = 1usize << log2_size;

    // Shift baseline includes the scaling list precision (4 bits for values centered at 16)
    let bd_shift = bit_depth as i32 + log2_size as i32 - 5;

    let right_shift = (bd_shift - qp_per).max(0);
    let left_shift = (qp_per - bd_shift).max(0) as u32;
    let add = if right_shift > 0 {
        1i64 << (right_shift - 1)
    } else {
        0
    };

    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            let c = coeffs[idx];
            if c != 0 {
                let sl_factor = scaling_list.get_factor(log2_size, matrix_id, x, y) as i64;
                let d = ((c as i64 * scale * sl_factor + add) >> right_shift) << left_shift;
                coeffs[idx] = d.clamp(-32768, 32767) as i16;
            }
        }
    }
}

/// Get the level scale factor for a given QP remainder.
pub fn level_scale(qp_rem: usize) -> i32 {
    LEVEL_SCALE[qp_rem]
}
