//! Unit tests for HEVC Transform + Dequantization.

use rasmcore_hevc::transform::{
    ScalingList, dequantize_block, dequantize_block_with_scaling_list, level_scale,
    reconstruct_residual,
};

// ============================================================================
// Level scale factor tests
// ============================================================================

#[test]
fn level_scale_values() {
    assert_eq!(level_scale(0), 40);
    assert_eq!(level_scale(1), 45);
    assert_eq!(level_scale(2), 51);
    assert_eq!(level_scale(3), 57);
    assert_eq!(level_scale(4), 64);
    assert_eq!(level_scale(5), 72);
}

// ============================================================================
// Dequantization tests (no scaling list)
// ============================================================================

#[test]
fn dequant_qp0_4x4_8bit() {
    // QP=0: qp_per=0, qp_rem=0, scale=40<<0=40
    // shift = 8 - 9 + 2 = 1, add = 1
    // d = (coeff * 40 + 1) >> 1
    let mut coeffs = [0i16; 16];
    coeffs[0] = 1;
    coeffs[1] = -1;
    coeffs[5] = 10;

    dequantize_block(&mut coeffs, 2, 0, 8);

    // (1 * 40 + 1) >> 1 = 20
    assert_eq!(coeffs[0], 20);
    // (-1 * 40 + 1) >> 1 = -39 >> 1 = -20 (arithmetic shift)
    assert_eq!(coeffs[1], -20);
    // (10 * 40 + 1) >> 1 = 200
    assert_eq!(coeffs[5], 200);
    // Zeros stay zero
    assert_eq!(coeffs[2], 0);
}

#[test]
fn dequant_qp22_4x4_8bit() {
    // QP=22: qp_per=3, qp_rem=4, scale=64<<3=512
    // shift = 8 - 9 + 2 = 1, add = 1
    // d = (coeff * 512 + 1) >> 1
    let mut coeffs = [0i16; 16];
    coeffs[0] = 1;

    dequantize_block(&mut coeffs, 2, 22, 8);

    assert_eq!(coeffs[0], 256); // (1 * 512 + 1) >> 1 = 256
}

#[test]
fn dequant_qp37_8x8_8bit() {
    // QP=37: qp_per=6, qp_rem=1, scale=45<<6=2880
    // shift = 8 - 9 + 3 = 2, add = 2
    // d = (coeff * 2880 + 2) >> 2
    let mut coeffs = [0i16; 64];
    coeffs[0] = 1;

    dequantize_block(&mut coeffs, 3, 37, 8);

    assert_eq!(coeffs[0], 720); // (1 * 2880 + 2) >> 2 = 720
}

#[test]
fn dequant_qp51_32x32_8bit() {
    // QP=51: qp_per=8, qp_rem=3, scale=57<<8=14592
    // shift = 8 - 9 + 5 = 4, add = 8
    // d = (coeff * 14592 + 8) >> 4
    let mut coeffs = [0i16; 1024];
    coeffs[0] = 1;

    dequantize_block(&mut coeffs, 5, 51, 8);

    assert_eq!(coeffs[0], 912); // (1 * 14592 + 8) >> 4 = 912
}

#[test]
fn dequant_clips_to_i16_range() {
    // Large coefficient * large scale should clip to i16 range
    let mut coeffs = [0i16; 16];
    coeffs[0] = 32767; // max i16

    dequantize_block(&mut coeffs, 2, 51, 8);

    // Should clip to 32767 (i16 max)
    assert_eq!(coeffs[0], 32767);
}

#[test]
fn dequant_negative_clips() {
    let mut coeffs = [0i16; 16];
    coeffs[0] = -32768; // min i16

    dequantize_block(&mut coeffs, 2, 51, 8);

    assert_eq!(coeffs[0], -32768);
}

#[test]
fn dequant_10bit() {
    // 10-bit: shift = 10 - 9 + 2 = 3
    // QP=0: scale = 40, add = 4
    // d = (1 * 40 + 4) >> 3 = 5
    let mut coeffs = [0i16; 16];
    coeffs[0] = 1;

    dequantize_block(&mut coeffs, 2, 0, 10);

    assert_eq!(coeffs[0], 5);
}

// ============================================================================
// Dequantization with scaling list
// ============================================================================

#[test]
fn dequant_with_default_scaling_list() {
    let sl = ScalingList::default();
    let mut coeffs = [0i16; 16];
    coeffs[0] = 1;

    // Default 4x4 scaling list: all values are 16
    // QP=0: scale=40, sl_factor=16
    // bd_shift = 8 + 2 - 5 = 5
    // right_shift = max(0, 5 - 0) = 5
    // add = 1 << 4 = 16
    // d = (1 * 40 * 16 + 16) >> 5 = (640 + 16) >> 5 = 656 >> 5 = 20
    dequantize_block_with_scaling_list(&mut coeffs, 2, 0, 8, &sl, 0);

    assert_eq!(coeffs[0], 20);
}

#[test]
fn dequant_with_flat_scaling_list_matches_no_list() {
    // With all-16 scaling list, the result should match the no-scaling-list path
    let sl = ScalingList::default();

    for qp in [0, 22, 26, 37, 51] {
        let mut with_sl = [0i16; 64];
        let mut without_sl = [0i16; 64];

        // Set some test coefficients
        with_sl[0] = 5;
        with_sl[1] = -3;
        with_sl[9] = 7;
        without_sl.copy_from_slice(&with_sl);

        dequantize_block_with_scaling_list(&mut with_sl, 3, qp, 8, &sl, 0);
        dequantize_block(&mut without_sl, 3, qp, 8);

        assert_eq!(
            with_sl, without_sl,
            "Mismatch at QP={qp}: with_sl != without_sl"
        );
    }
}

#[test]
fn dequant_scaling_list_non_flat() {
    let mut sl = ScalingList::default();
    // Set (0,0) to 32 instead of 16
    sl.list_4x4[0][0] = 32;

    let mut coeffs = [0i16; 16];
    coeffs[0] = 1;

    // QP=0, 4x4, 8-bit: scale=40, sl_factor=32
    // bd_shift=5, right_shift=5, add=16
    // d = (1 * 40 * 32 + 16) >> 5 = (1280 + 16) >> 5 = 1296 >> 5 = 40
    dequantize_block_with_scaling_list(&mut coeffs, 2, 0, 8, &sl, 0);

    assert_eq!(coeffs[0], 40);
}

// ============================================================================
// Scaling list tests
// ============================================================================

#[test]
fn scaling_list_default_4x4_is_flat() {
    let sl = ScalingList::default();
    for matrix_id in 0..6u8 {
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(sl.get_factor(2, matrix_id, x, y), 16);
            }
        }
    }
}

#[test]
fn scaling_list_default_8x8_intra_dc() {
    let sl = ScalingList::default();
    // DC (0,0) of 8x8 intra should be 16
    assert_eq!(sl.get_factor(3, 0, 0, 0), 16);
}

#[test]
fn scaling_list_default_8x8_has_variation() {
    let sl = ScalingList::default();
    // Intra 8x8: (7,7) should be 115 (from default table)
    assert_eq!(sl.get_factor(3, 0, 7, 7), 115);
    // Inter 8x8: (7,7) should be 91
    assert_eq!(sl.get_factor(3, 3, 7, 7), 91);
}

#[test]
fn scaling_list_16x16_upsamples_from_8x8() {
    let sl = ScalingList::default();
    // 16x16 at (14,14) maps to 8x8 at (7,7) = 115 for intra
    assert_eq!(sl.get_factor(4, 0, 14, 14), 115);
    // DC uses dc_16x16 value = 16
    assert_eq!(sl.get_factor(4, 0, 0, 0), 16);
}

#[test]
fn scaling_list_32x32_upsamples_from_8x8() {
    let sl = ScalingList::default();
    // 32x32 at (28,28) maps to 8x8 at (7,7) = 115 for intra
    assert_eq!(sl.get_factor(5, 0, 28, 28), 115);
    // 32x32 inter: matrixId 3 maps to index 1 → inter default
    assert_eq!(sl.get_factor(5, 3, 28, 28), 91);
}

// ============================================================================
// Transform dispatch tests
// ============================================================================

#[test]
fn transform_dispatch_dst_for_intra_4x4_luma() {
    // DC-only input: coefficient at (0,0) = 100, rest = 0
    let mut coeffs = [0i16; 16];
    coeffs[0] = 100;
    let mut output = [0i16; 16];

    // Intra luma 4x4 → should use DST
    reconstruct_residual(&coeffs, &mut output, 2, 26, 8, true, None, 0).unwrap();

    // DST: DC does NOT distribute evenly (unlike DCT)
    // Output should be non-uniform
    let all_same = output.iter().all(|&v| v == output[0]);
    assert!(!all_same, "DST output should not be uniform for DC input");
}

#[test]
fn transform_dispatch_dct_for_inter_4x4() {
    // DC-only input
    let mut coeffs = [0i16; 16];
    coeffs[0] = 100;
    let mut output = [0i16; 16];

    // Inter 4x4 → should use DCT
    reconstruct_residual(&coeffs, &mut output, 2, 26, 8, false, None, 3).unwrap();

    // DCT: DC distributes more evenly than DST
    // Just verify no error and output is reasonable
    assert!(output.iter().any(|&v| v != 0));
}

#[test]
fn transform_dispatch_dct_for_8x8() {
    let mut coeffs = [0i16; 64];
    coeffs[0] = 50;
    let mut output = [0i16; 64];

    reconstruct_residual(&coeffs, &mut output, 3, 26, 8, false, None, 0).unwrap();
    assert!(output.iter().any(|&v| v != 0));
}

#[test]
fn transform_dispatch_dct_for_16x16() {
    let mut coeffs = [0i16; 256];
    coeffs[0] = 50;
    let mut output = [0i16; 256];

    reconstruct_residual(&coeffs, &mut output, 4, 26, 8, false, None, 0).unwrap();
    assert!(output.iter().any(|&v| v != 0));
}

#[test]
fn transform_dispatch_dct_for_32x32() {
    let mut coeffs = [0i16; 1024];
    coeffs[0] = 50;
    let mut output = [0i16; 1024];

    reconstruct_residual(&coeffs, &mut output, 5, 26, 8, false, None, 0).unwrap();
    assert!(output.iter().any(|&v| v != 0));
}

// ============================================================================
// Full residual pipeline tests
// ============================================================================

#[test]
fn residual_pipeline_zero_coefficients() {
    let coeffs = [0i16; 16];
    let mut output = [99i16; 16]; // Fill with non-zero to verify it's zeroed

    reconstruct_residual(&coeffs, &mut output, 2, 26, 8, false, None, 0).unwrap();

    // All-zero coefficients should produce all-zero residuals
    for &v in &output {
        assert_eq!(v, 0);
    }
}

#[test]
fn residual_pipeline_clips_to_bit_depth() {
    // Use a very large coefficient to verify clipping
    let mut coeffs = [0i16; 16];
    coeffs[0] = 32767;
    let mut output = [0i16; 16];

    reconstruct_residual(&coeffs, &mut output, 2, 51, 8, false, None, 0).unwrap();

    // HEVC Section 8.6.2: residuals clip to [-(1<<(BitDepth+1)), (1<<(BitDepth+1))-1].
    // For 8-bit: [-512, 511], NOT [-128, 127].
    for &v in &output {
        assert!(v >= -512 && v <= 511, "residual {v} out of 8-bit range [-512,511]");
    }
}

#[test]
fn residual_pipeline_clips_10bit() {
    let mut coeffs = [0i16; 16];
    coeffs[0] = 32767;
    let mut output = [0i16; 16];

    reconstruct_residual(&coeffs, &mut output, 2, 51, 10, false, None, 0).unwrap();

    // All output values should be in [-512, 511] range for 10-bit
    for &v in &output {
        assert!(v >= -512 && v <= 511, "residual {v} out of 10-bit range");
    }
}

#[test]
fn residual_pipeline_with_scaling_list() {
    let sl = ScalingList::default();
    let mut coeffs = [0i16; 16];
    coeffs[0] = 10;
    let mut output_sl = [0i16; 16];
    let mut output_no_sl = [0i16; 16];

    // With flat default scaling list
    reconstruct_residual(&coeffs, &mut output_sl, 2, 26, 8, false, Some(&sl), 0).unwrap();

    // Without scaling list
    reconstruct_residual(&coeffs, &mut output_no_sl, 2, 26, 8, false, None, 0).unwrap();

    // Flat default should match no-scaling-list
    assert_eq!(output_sl, output_no_sl);
}

#[test]
fn residual_pipeline_buffer_too_small() {
    let coeffs = [0i16; 8]; // Too small for 4x4
    let mut output = [0i16; 16];

    let result = reconstruct_residual(&coeffs, &mut output, 2, 26, 8, false, None, 0);
    assert!(result.is_err());
}

#[test]
fn residual_pipeline_invalid_size() {
    let coeffs = [0i16; 16];
    let mut output = [0i16; 16];

    let result = reconstruct_residual(&coeffs, &mut output, 1, 26, 8, false, None, 0);
    assert!(result.is_err());
}

#[test]
fn residual_dc_only_dct_4x4_is_uniform() {
    // For DCT, a DC-only input should produce a nearly uniform spatial block
    let mut coeffs = [0i16; 16];
    coeffs[0] = 40; // Moderate DC value
    let mut output = [0i16; 16];

    reconstruct_residual(&coeffs, &mut output, 2, 26, 8, false, None, 0).unwrap();

    // DCT DC distributes evenly — all values should be the same
    let first = output[0];
    for &v in &output[1..] {
        assert_eq!(v, first, "DCT DC output should be uniform");
    }
}
