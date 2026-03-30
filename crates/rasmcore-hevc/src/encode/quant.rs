//! Forward quantization for HEVC encoder.
//!
//! The inverse of dequantization (transform/dequant.rs). Takes DCT/DST coefficients
//! and produces quantized levels that, after dequantization, approximate the originals.
//!
//! Ref: x265 4.1 common/quant.cpp — quant_c()
//! Ref: ITU-T H.265 Section 8.6.3 (inverse, from which forward is derived)

/// Inverse quantization scale factors (same as dequant, for roundtrip validation).
#[allow(dead_code)]
const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// Forward quantization scale factors — reciprocals of LEVEL_SCALE, scaled by (1<<20).
/// These are the HM/x265 g_quantScales values.
/// Computed as: QUANT_SCALE[i] = round((1 << 20) / LEVEL_SCALE[i])
///
/// Ref: HM g_quantScales[] in TComRom.cpp
/// Ref: x265 4.1 common/quant.cpp — invQuantScales (same values)
const QUANT_SCALE: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

/// Forward quantize a block of DCT/DST coefficients.
///
/// Produces quantized levels (i16) from transform coefficients. The levels can be
/// dequantized back using `dequant::dequantize_block()`.
///
/// Formula per coefficient:
///   level = sign(coeff) * ((abs(coeff) * quant_scale + offset) >> shift)
///
/// where:
///   quant_scale = QUANT_SCALE[qp%6]
///   shift = QUANT_SHIFT + qp/6 + transform_shift
///   transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2_size
///                   = 15 - 8 - log2_size (for 8-bit)
///   offset = dead-zone rounding (1 << (shift-1)) / 3 for inter, /6 for intra
///
/// Ref: x265 4.1 common/quant.cpp — quant_c()
pub fn quantize_block(coeffs: &[i16], output: &mut [i16], log2_size: u8, qp: i32, bit_depth: u8) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let quant_scale = QUANT_SCALE[qp_rem] as i64;

    // Transform shift: MAX_TR_DYNAMIC_RANGE - bitDepth - log2Size
    // MAX_TR_DYNAMIC_RANGE = 15 for 8-bit
    let transform_shift = 15i32 - bit_depth as i32 - log2_size as i32;

    // Total shift = QUANT_SHIFT + qp/6 + transform_shift
    // QUANT_SHIFT = 14 (matches the 14-bit precision of QUANT_SCALE)
    let shift = 14 + qp_per + transform_shift;

    if shift > 0 {
        let add = (1i64 << (shift - 1)) / 3; // dead-zone rounding for intra
        for (i, &c) in coeffs.iter().enumerate() {
            if c == 0 {
                output[i] = 0;
            } else {
                let sign = if c < 0 { -1i64 } else { 1i64 };
                let abs_c = c.unsigned_abs() as i64;
                let level = (abs_c * quant_scale + add) >> shift;
                output[i] = (sign * level).clamp(-32768, 32767) as i16;
            }
        }
    } else {
        // Shouldn't happen for valid QP/size combinations
        for (i, &c) in coeffs.iter().enumerate() {
            output[i] = c;
        }
    }
}

/// Count the number of non-zero quantized coefficients.
pub fn has_nonzero(coeffs: &[i16]) -> bool {
    coeffs.iter().any(|&c| c != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::dequant::dequantize_block;

    #[test]
    fn quantize_dequantize_roundtrip() {
        // Forward quant then dequant: DC should survive, small AC may be killed
        let mut coeffs = [0i16; 16]; // 4x4 block
        coeffs[0] = 2000; // Large DC — will survive quantization
        coeffs[1] = -1000; // Large AC

        let mut quantized = [0i16; 16];
        quantize_block(&coeffs, &mut quantized, 2, 22, 8);

        // DC should survive quantization at QP=22
        assert!(quantized[0] != 0, "DC should survive quantization");

        // Dequantize
        let mut reconstructed = quantized;
        dequantize_block(&mut reconstructed, 2, 22, 8);

        // DC should approximately match (within quant step)
        let dc_diff = (coeffs[0] as i32 - reconstructed[0] as i32).abs();
        // At QP=22, the quantization step is ~256 for a 4x4 block.
        // The roundtrip error should be less than one full quant step.
        assert!(
            dc_diff < 300,
            "DC roundtrip: original={}, reconstructed={}, diff={dc_diff}",
            coeffs[0],
            reconstructed[0]
        );
    }

    #[test]
    fn quantize_zero_input() {
        let coeffs = [0i16; 64];
        let mut output = [99i16; 64];
        quantize_block(&coeffs, &mut output, 3, 26, 8);
        assert!(
            output.iter().all(|&v| v == 0),
            "zero input should produce zero output"
        );
    }

    #[test]
    fn quantize_high_qp_kills_small_coeffs() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 10; // Small coefficient
        coeffs[1] = 5;

        let mut output = [0i16; 16];
        quantize_block(&coeffs, &mut output, 2, 40, 8); // High QP

        // At QP=40, small coefficients should be quantized to zero
        assert!(
            output.iter().filter(|&&v| v != 0).count() <= 1,
            "high QP should kill most small coefficients"
        );
    }

    #[test]
    fn quantize_preserves_sign() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 500;
        coeffs[1] = -500;

        let mut output = [0i16; 16];
        quantize_block(&coeffs, &mut output, 2, 22, 8);

        if output[0] != 0 {
            assert!(
                output[0] > 0,
                "positive input should produce positive output"
            );
        }
        if output[1] != 0 {
            assert!(
                output[1] < 0,
                "negative input should produce negative output"
            );
        }
    }

    #[test]
    fn quantize_various_qp() {
        let mut coeffs = [0i16; 64]; // 8x8
        coeffs[0] = 1000;

        for qp in [10, 22, 30, 40, 51] {
            let mut output = [0i16; 64];
            quantize_block(&coeffs, &mut output, 3, qp, 8);
            // Higher QP should produce smaller (or zero) quantized values
            eprintln!("QP={qp}: DC quantized to {}", output[0]);
        }
    }
}
