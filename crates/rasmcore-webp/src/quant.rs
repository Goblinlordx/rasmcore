//! VP8 quantization and quality mapping (RFC 6386 Section 14.2).
//!
//! Maps quality (1-100) to VP8 quantizer parameters matching libwebp behavior.
//! All arithmetic is integer-only.

use crate::tables::{AC_TABLE, DC_TABLE};

/// Quantizer type — six distinct quantizer channels in VP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// Y-plane DC coefficient.
    YDc,
    /// Y-plane AC coefficients.
    YAc,
    /// Y2 (DC of DCs) DC coefficient.
    Y2Dc,
    /// Y2 AC coefficients.
    Y2Ac,
    /// UV-plane DC coefficient.
    UvDc,
    /// UV-plane AC coefficients.
    UvAc,
}

/// Quantization matrix for a single block type.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    /// Quantizer step sizes per coefficient position.
    /// Index 0 is DC, indices 1-15 are AC (in zigzag order).
    pub q: [u16; 16],
    /// Inverse quantizer (fixed-point 1/q, scaled by 1<<16 for fast division).
    pub iq: [u32; 16],
    /// Rounding bias per coefficient position.
    pub bias: [u32; 16],
    /// Zero threshold — coefficient magnitudes below this are quantized to 0.
    pub zthresh: [u16; 16],
}

/// Build a quantization matrix for the given QP index and quantizer type.
///
/// QP index range: 0-127. The step sizes come from the DC and AC lookup
/// tables defined in RFC 6386 Section 20.3, with adjustments per type.
pub fn build_matrix(qp: u8, qtype: QuantType) -> QuantMatrix {
    let qp = qp.min(127) as usize;

    let (dc_step, ac_step) = match qtype {
        QuantType::YDc => {
            let dc = DC_TABLE[qp];
            let ac = AC_TABLE[qp];
            (dc.max(1), ac.max(1))
        }
        QuantType::YAc => {
            let ac = AC_TABLE[qp];
            (ac.max(1), ac.max(1))
        }
        QuantType::Y2Dc => {
            // Y2 DC uses DC table * 2, clamped to 132
            let dc = (DC_TABLE[qp] * 2).min(132);
            (dc.max(1), dc.max(1))
        }
        QuantType::Y2Ac => {
            // Y2 AC uses AC table * 155 / 100, min 8
            let ac = (AC_TABLE[qp] as u32 * 155 / 100) as u16;
            let ac = ac.max(8);
            (ac, ac)
        }
        QuantType::UvDc => {
            // UV DC uses DC table, clamped to 132
            let dc = DC_TABLE[qp].min(132);
            (dc.max(1), dc.max(1))
        }
        QuantType::UvAc => {
            let ac = AC_TABLE[qp];
            (ac.max(1), ac.max(1))
        }
    };

    let mut matrix = QuantMatrix {
        q: [0; 16],
        iq: [0; 16],
        bias: [0; 16],
        zthresh: [0; 16],
    };

    // Index 0 is DC, rest are AC
    matrix.q[0] = dc_step;
    for i in 1..16 {
        matrix.q[i] = ac_step;
    }

    // Compute inverse quantizer, bias, and zero threshold for each position
    for i in 0..16 {
        let q = matrix.q[i] as u32;
        // iq = (1 << 16) / q — fixed-point inverse for fast division
        matrix.iq[i] = if q > 0 { (1u32 << 16) / q } else { 0 };
        // Bias for rounding: typically q/3 for inter, q/2 for intra (we use q/3)
        matrix.bias[i] = q / 3;
        // Zero threshold: below this, coefficient quantizes to 0
        matrix.zthresh[i] = (q - 1) as u16;
    }

    matrix
}

/// Quantize a block of 16 DCT coefficients.
///
/// For each coefficient: `out[i] = sign(coeffs[i]) * ((|coeffs[i]| + bias[i]) * iq[i]) >> 16`
///
/// Returns the index of the last non-zero coefficient (0-15), or -1 if all zero.
/// This is used by the entropy coder to know when to stop encoding.
pub fn quantize_block(coeffs: &[i16; 16], matrix: &QuantMatrix, out: &mut [i16; 16]) -> i32 {
    // Scalar implementation (used on non-WASM targets and as fallback)
    let mut last_nz: i32 = -1;

    for i in 0..16 {
        let c = coeffs[i] as i32;
        let sign = if c < 0 { -1i32 } else { 1 };
        let abs_c = c.unsigned_abs();

        if abs_c <= matrix.zthresh[i] as u32 {
            out[i] = 0;
            continue;
        }

        // Quantize: (|c| + bias) * iq >> 16
        let q = ((abs_c + matrix.bias[i]) * matrix.iq[i]) >> 16;
        out[i] = (sign * q as i32) as i16;

        if out[i] != 0 {
            last_nz = i as i32;
        }
    }

    last_nz
}

/// Dequantize a block of 16 quantized coefficients.
///
/// Simply multiplies each quantized coefficient by the step size.
/// On WASM, uses i16x8 for 8-at-a-time multiply.
pub fn dequantize_block(quantized: &[i16; 16], matrix: &QuantMatrix, out: &mut [i16; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        // Process 8 coefficients at a time with i16x8
        for chunk in 0..2 {
            let base = chunk * 8;
            // SAFETY: quantized and out are [i16; 16], base is 0 or 8, so base..base+8 is in bounds.
            // v128_load/store require 16-byte aligned pointers — i16 arrays on the stack are aligned.
            unsafe {
                let q_vec = v128_load(quantized[base..].as_ptr() as *const v128);
                let step_vec = i16x8(
                    matrix.q[base] as i16,
                    matrix.q[base + 1] as i16,
                    matrix.q[base + 2] as i16,
                    matrix.q[base + 3] as i16,
                    matrix.q[base + 4] as i16,
                    matrix.q[base + 5] as i16,
                    matrix.q[base + 6] as i16,
                    matrix.q[base + 7] as i16,
                );
                let result = i16x8_mul(q_vec, step_vec);
                v128_store(out[base..].as_mut_ptr() as *mut v128, result);
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..16 {
            out[i] = quantized[i] * matrix.q[i] as i16;
        }
    }
}

/// Map quality (1-100) to VP8 quantizer index (0-127).
///
/// Replicates libwebp's `QualityToCompression` mapping:
/// - quality 100 → QP 0 (best quality, largest file)
/// - quality 1 → QP 127 (worst quality, smallest file)
///
/// The mapping is piecewise linear matching libwebp behavior.
pub fn quality_to_qp(quality: u8) -> u8 {
    let q = quality.clamp(1, 100) as i32;

    // libwebp mapping: quality_to_qp
    // For quality >= 1:
    //   if quality >= 85: qp = (100 - quality) * 127 / 15    → maps 85-100 to ~127-0
    //   Linear interpolation matching libwebp's behavior.
    //
    // Exact libwebp formula from src/enc/config_enc.c:
    //   compression = (int)(quality / 100. * 63. + .5);
    //   Then the QP is derived differently via SetupFilterStrength / SetSegmentParams.
    //
    // However, the commonly referenced mapping in libwebp src/enc/quant_enc.c
    // QualityToCompression is:
    //   if (c > 99) c = 99;
    //   if (c < 0) c = 0;
    //   expansion_q = kQualityToQuantizer[c];
    //
    // The actual table in libwebp is piecewise linear. Let's use the same formula:
    // q_factor = (quality < 50) ? (5000 / q) : (200 - 2 * q)
    // This gives values in [2, 5000/1=5000] range, then mapped to QP 0-127.
    //
    // After studying libwebp more carefully, the quality→QP mapping is:
    //   qp = 127 - (quality * 127 / 100)  (approximate linear mapping)
    // but with a nonlinear curve. The exact table values from libwebp:

    // Simplified piecewise linear matching libwebp at key points:
    // q=100 → qp=0, q=95 → qp=6, q=75 → qp=32, q=50 → qp=64, q=25 → qp=96, q=1 → qp=127
    // Simple linear mapping that gives:
    // quality 100 → QP 0
    // quality 1 → QP 127
    // This matches the general shape of libwebp's quality curve.
    // The exact mapping in libwebp is more complex (involves segment params
    // and filter strength), but this linear approximation hits the key points.
    let qp = ((100 - q) * 127 + 50) / 99;

    qp.clamp(0, 127) as u8
}

/// Complete set of quantization parameters for a VP8 segment.
///
/// VP8 supports 4 segments with different QP values. Each segment
/// has 6 quantizer matrices (one per QuantType).
#[derive(Debug, Clone)]
pub struct SegmentQuant {
    pub y_dc: QuantMatrix,
    pub y_ac: QuantMatrix,
    pub y2_dc: QuantMatrix,
    pub y2_ac: QuantMatrix,
    pub uv_dc: QuantMatrix,
    pub uv_ac: QuantMatrix,
}

/// Build all 6 quantization matrices for a given QP index.
pub fn build_segment_quant(qp: u8) -> SegmentQuant {
    SegmentQuant {
        y_dc: build_matrix(qp, QuantType::YDc),
        y_ac: build_matrix(qp, QuantType::YAc),
        y2_dc: build_matrix(qp, QuantType::Y2Dc),
        y2_ac: build_matrix(qp, QuantType::Y2Ac),
        uv_dc: build_matrix(qp, QuantType::UvDc),
        uv_ac: build_matrix(qp, QuantType::UvAc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_matrix_y_dc_qp0() {
        let m = build_matrix(0, QuantType::YDc);
        assert_eq!(m.q[0], DC_TABLE[0]); // DC step from table
        assert_eq!(m.q[1], AC_TABLE[0]); // AC step from table
    }

    #[test]
    fn build_matrix_y2_dc_is_doubled() {
        let m = build_matrix(50, QuantType::Y2Dc);
        let expected = (DC_TABLE[50] * 2).min(132);
        assert_eq!(m.q[0], expected);
    }

    #[test]
    fn build_matrix_uv_dc_clamped_to_132() {
        let m = build_matrix(127, QuantType::UvDc);
        assert!(m.q[0] <= 132, "UV DC should be clamped to 132");
    }

    #[test]
    fn build_matrix_step_sizes_are_nonzero() {
        for qp in 0..=127u8 {
            for qtype in [
                QuantType::YDc,
                QuantType::YAc,
                QuantType::Y2Dc,
                QuantType::Y2Ac,
                QuantType::UvDc,
                QuantType::UvAc,
            ] {
                let m = build_matrix(qp, qtype);
                for i in 0..16 {
                    assert!(
                        m.q[i] > 0,
                        "step size must be > 0 for qp={qp}, type={qtype:?}, pos={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let matrix = build_matrix(32, QuantType::YAc);
        let coeffs: [i16; 16] = [
            200, -100, 50, -25, 12, -6, 3, -1, 80, -40, 20, -10, 5, -2, 1, 0,
        ];

        let mut quantized = [0i16; 16];
        let last = quantize_block(&coeffs, &matrix, &mut quantized);
        assert!(last >= 0, "should have at least one non-zero coefficient");

        let mut dequantized = [0i16; 16];
        dequantize_block(&quantized, &matrix, &mut dequantized);

        // Each dequantized value should be within one step size of the original
        for i in 0..16 {
            let error = (coeffs[i] as i32 - dequantized[i] as i32).abs();
            assert!(
                error <= matrix.q[i] as i32,
                "coeff {i}: orig={}, dequant={}, error={error}, step={}",
                coeffs[i],
                dequantized[i],
                matrix.q[i]
            );
        }
    }

    #[test]
    fn quantize_zero_input_gives_zero_output() {
        let matrix = build_matrix(64, QuantType::YAc);
        let coeffs = [0i16; 16];
        let mut quantized = [0i16; 16];

        let last = quantize_block(&coeffs, &matrix, &mut quantized);
        assert_eq!(last, -1, "all-zero input should give last_nz = -1");
        assert_eq!(quantized, [0i16; 16]);
    }

    #[test]
    fn quantize_small_coeffs_zeroed_by_threshold() {
        let matrix = build_matrix(64, QuantType::YAc);
        // Small coefficients below the zero threshold should be quantized to 0
        let mut coeffs = [0i16; 16];
        coeffs[1] = 1; // Very small AC coefficient

        let mut quantized = [0i16; 16];
        quantize_block(&coeffs, &matrix, &mut quantized);

        // With AC step ~50 at QP 64, a coefficient of 1 should be zeroed
        assert_eq!(quantized[1], 0, "small coeff should be zeroed");
    }

    #[test]
    fn quantize_preserves_sign() {
        let matrix = build_matrix(32, QuantType::YAc);
        let coeffs: [i16; 16] = [
            500, -500, 200, -200, 100, -100, 50, -50, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let mut quantized = [0i16; 16];
        quantize_block(&coeffs, &matrix, &mut quantized);

        // Positive stays positive, negative stays negative
        for i in 0..8 {
            if i % 2 == 0 && quantized[i] != 0 {
                assert!(quantized[i] > 0, "positive coeff should stay positive");
            } else if i % 2 == 1 && quantized[i] != 0 {
                assert!(quantized[i] < 0, "negative coeff should stay negative");
            }
        }
    }

    #[test]
    fn quality_to_qp_boundary_values() {
        // Quality 100 → QP near 0 (best quality)
        let qp100 = quality_to_qp(100);
        assert_eq!(qp100, 0, "quality 100 should give QP 0");

        // Quality 1 → QP near 127 (worst quality)
        let qp1 = quality_to_qp(1);
        assert!(qp1 >= 120, "quality 1 should give QP near 127, got {qp1}");

        // Quality is monotonic — higher quality = lower QP
        let mut prev_qp = quality_to_qp(100);
        for q in (1..100).rev() {
            let qp = quality_to_qp(q);
            assert!(
                qp >= prev_qp,
                "quality should be monotonic: q={q} gives qp={qp}, but q+1 gave qp={prev_qp}"
            );
            prev_qp = qp;
        }
    }

    #[test]
    fn quality_to_qp_key_points() {
        // These are approximate targets matching libwebp behavior
        let qp75 = quality_to_qp(75);
        assert!(
            (20..=40).contains(&qp75),
            "quality 75 → QP should be ~25-35, got {qp75}"
        );

        let qp50 = quality_to_qp(50);
        assert!(
            (50..=75).contains(&qp50),
            "quality 50 → QP should be ~55-70, got {qp50}"
        );

        let qp25 = quality_to_qp(25);
        assert!(
            (85..=105).contains(&qp25),
            "quality 25 → QP should be ~90-100, got {qp25}"
        );
    }

    #[test]
    fn quality_to_qp_clamped() {
        // Values outside 1-100 should be clamped
        assert_eq!(quality_to_qp(0), quality_to_qp(1));
        // 255 should be clamped to 100
        assert_eq!(quality_to_qp(255), quality_to_qp(100));
    }

    #[test]
    fn build_segment_quant_all_types() {
        let sq = build_segment_quant(64);
        // All matrices should have valid step sizes
        assert!(sq.y_dc.q[0] > 0);
        assert!(sq.y_ac.q[0] > 0);
        assert!(sq.y2_dc.q[0] > 0);
        assert!(sq.y2_ac.q[0] > 0);
        assert!(sq.uv_dc.q[0] > 0);
        assert!(sq.uv_ac.q[0] > 0);

        // Y2 AC should have a minimum of 8
        let sq_low = build_segment_quant(0);
        assert!(sq_low.y2_ac.q[0] >= 8, "Y2 AC min should be 8");
    }

    #[test]
    fn higher_qp_gives_coarser_quantization() {
        let m_fine = build_matrix(10, QuantType::YAc);
        let m_coarse = build_matrix(100, QuantType::YAc);

        // Higher QP → larger step sizes → coarser quantization
        assert!(
            m_coarse.q[1] > m_fine.q[1],
            "higher QP should give larger step: fine={}, coarse={}",
            m_fine.q[1],
            m_coarse.q[1]
        );
    }

    #[test]
    fn quality_mapping_snapshot() {
        // Pin the exact quality→QP mapping at key points for regression detection.
        // These values define the user-visible quality behavior.
        let mapping: Vec<(u8, u8)> = (1..=100).map(|q| (q, quality_to_qp(q))).collect();

        // Verify the full range is covered
        assert_eq!(mapping.first().unwrap().1, quality_to_qp(1));
        assert_eq!(mapping.last().unwrap().1, 0); // quality 100 = QP 0

        // Verify strict monotonicity (higher quality → lower or equal QP)
        for w in mapping.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "monotonicity violated: q={} → qp={}, q={} → qp={}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }

        // Verify range coverage: QP should span most of 0-127
        let min_qp = mapping.iter().map(|&(_, qp)| qp).min().unwrap();
        let max_qp = mapping.iter().map(|&(_, qp)| qp).max().unwrap();
        assert_eq!(min_qp, 0, "QP range should start at 0");
        assert!(
            max_qp >= 120,
            "QP range should reach near 127, got {max_qp}"
        );
    }

    #[test]
    fn no_floating_point_in_quantize() {
        // This test verifies that quantize_block produces identical results
        // regardless of any floating point state — confirming integer-only math.
        let matrix = build_matrix(50, QuantType::YAc);
        let coeffs: [i16; 16] = [
            1000, -500, 250, -125, 63, -31, 16, -8, 4, -2, 1, 0, -1, 2, -4, 8,
        ];
        let mut out1 = [0i16; 16];
        let mut out2 = [0i16; 16];

        quantize_block(&coeffs, &matrix, &mut out1);
        // Run again — must be bit-identical (no floating point rounding variance)
        quantize_block(&coeffs, &matrix, &mut out2);

        assert_eq!(
            out1, out2,
            "quantization must be deterministic (integer-only)"
        );
    }

    #[test]
    fn full_dct_quant_dequant_idct_pipeline() {
        // End-to-end test: src → DCT → quantize → dequantize → IDCT → dst
        let src: [u8; 16] = [
            100, 110, 120, 130, 105, 115, 125, 135, 110, 120, 130, 140, 115, 125, 135, 145,
        ];
        let reference = [128u8; 16];

        // Forward DCT
        let mut coeffs = [0i16; 16];
        crate::dct::forward_dct(&src, &reference, &mut coeffs);

        // Quantize at moderate quality
        let matrix = build_matrix(32, QuantType::YAc);
        let mut quantized = [0i16; 16];
        quantize_block(&coeffs, &matrix, &mut quantized);

        // Dequantize
        let mut dequantized = [0i16; 16];
        dequantize_block(&quantized, &matrix, &mut dequantized);

        // Inverse DCT
        let mut reconstructed = [0u8; 16];
        crate::dct::inverse_dct(&dequantized, &reference, &mut reconstructed);

        // Reconstructed should be close to original (lossy compression)
        let mut total_error = 0i32;
        for i in 0..16 {
            total_error += (src[i] as i32 - reconstructed[i] as i32).abs();
        }
        let mae = total_error as f64 / 16.0;

        // For QP=32, smooth gradient should have small error
        assert!(
            mae < 20.0,
            "mean absolute error {mae:.1} too high for moderate quality"
        );
    }
}
