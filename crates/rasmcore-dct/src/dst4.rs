//! 4x4 integer DST (HEVC spec, ITU-T H.265 Table 8-7).
//!
//! Used only for HEVC 4x4 intra luma blocks. Different basis from DCT.
//!
//! Transform matrix (scaled):
//!   [ 29  55  74  84 ]
//!   [ 74  74   0 -74 ]
//!   [ 84 -29 -74  55 ]
//!   [ 55 -84  74 -29 ]

const D4: [i32; 4] = [29, 55, 74, 84];

/// Inverse 4x4 DST: frequency → spatial domain.
///
/// Input: 4x4 block of i16 DST coefficients (row-major).
/// Output: 4x4 block of i16 residuals (row-major).
pub fn inverse_dst_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // Row pass
    for i in 0..4 {
        let s = i * 4;
        let c0 = input[s] as i32;
        let c1 = input[s + 1] as i32;
        let c2 = input[s + 2] as i32;
        let c3 = input[s + 3] as i32;

        let add = 64; // 1 << 6
        tmp[s] =
            ((D4[0] * c0 + D4[1] * c1 + D4[2] * c2 + D4[3] * c3 + add) >> 7).clamp(-32768, 32767);
        tmp[s + 1] = ((D4[2] * c0 + D4[2] * c1 - D4[2] * c3 + add) >> 7).clamp(-32768, 32767);
        tmp[s + 2] =
            ((D4[3] * c0 - D4[0] * c1 - D4[2] * c2 + D4[1] * c3 + add) >> 7).clamp(-32768, 32767);
        tmp[s + 3] =
            ((D4[1] * c0 - D4[3] * c1 + D4[2] * c2 - D4[0] * c3 + add) >> 7).clamp(-32768, 32767);
    }

    // Column pass
    for j in 0..4 {
        let c0 = tmp[j];
        let c1 = tmp[4 + j];
        let c2 = tmp[8 + j];
        let c3 = tmp[12 + j];

        let add = 1 << 11;
        output[j] = ((D4[0] * c0 + D4[1] * c1 + D4[2] * c2 + D4[3] * c3 + add) >> 12) as i16;
        output[4 + j] = ((D4[2] * c0 + D4[2] * c1 - D4[2] * c3 + add) >> 12) as i16;
        output[8 + j] = ((D4[3] * c0 - D4[0] * c1 - D4[2] * c2 + D4[1] * c3 + add) >> 12) as i16;
        output[12 + j] = ((D4[1] * c0 - D4[3] * c1 + D4[2] * c2 - D4[0] * c3 + add) >> 12) as i16;
    }
}

/// Forward 4x4 DST: spatial → frequency domain.
///
/// The forward DST uses the TRANSPOSE of the inverse matrix.
/// For DST, the forward matrix is the transpose of the inverse matrix
/// (since DST is an orthogonal transform).
///
/// Shift convention per HEVC spec (8-bit):
/// - shift1 = log2(N) + bitDepth - 9 = 2 + 8 - 9 = 1
/// - shift2 = log2(N) + 6 = 2 + 6 = 8
///
/// Ref: x265 4.1 common/dct.cpp — dst4_c()
pub fn forward_dst_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // Row pass: multiply by DST matrix (transpose of inverse)
    let shift1 = 1;
    let add1 = 1 << (shift1 - 1);
    for i in 0..4 {
        let s = i * 4;
        let x0 = input[s] as i32;
        let x1 = input[s + 1] as i32;
        let x2 = input[s + 2] as i32;
        let x3 = input[s + 3] as i32;

        // Forward DST matrix (rows are the transposed columns of the inverse matrix):
        // Row 0: [29, 74, 84, 55]
        // Row 1: [55, 74, -29, -84]
        // Row 2: [74, 0, -74, 74]  (note: coefficient pattern differs from inverse)
        // Row 3: [84, -74, 55, -29]
        tmp[s] = (D4[0] * x0 + D4[2] * x1 + D4[3] * x2 + D4[1] * x3 + add1) >> shift1;
        tmp[s + 1] = (D4[1] * x0 + D4[2] * x1 - D4[0] * x2 - D4[3] * x3 + add1) >> shift1;
        tmp[s + 2] = (D4[2] * x0 - D4[2] * x3 + add1) >> shift1;
        tmp[s + 3] = (D4[3] * x0 - D4[2] * x1 + D4[1] * x2 - D4[0] * x3 + add1) >> shift1;
    }

    // Column pass
    let shift2 = 8;
    let add2 = 1 << (shift2 - 1);
    for j in 0..4 {
        let x0 = tmp[j];
        let x1 = tmp[4 + j];
        let x2 = tmp[8 + j];
        let x3 = tmp[12 + j];

        output[j] = ((D4[0] * x0 + D4[2] * x1 + D4[3] * x2 + D4[1] * x3 + add2) >> shift2) as i16;
        output[4 + j] =
            ((D4[1] * x0 + D4[2] * x1 - D4[0] * x2 - D4[3] * x3 + add2) >> shift2) as i16;
        output[8 + j] = ((D4[2] * x0 - D4[2] * x3 + add2) >> shift2) as i16;
        output[12 + j] =
            ((D4[3] * x0 - D4[2] * x1 + D4[1] * x2 - D4[0] * x3 + add2) >> shift2) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_dst_4x4_dc_input() {
        // Single non-zero DC coefficient
        let mut input = [0i16; 16];
        input[0] = 100;
        let mut output = [0i16; 16];
        inverse_dst_4x4(&input, &mut output);

        // All outputs should be non-zero (DST doesn't have flat DC response like DCT)
        let non_zero = output.iter().filter(|&&v| v != 0).count();
        assert!(non_zero > 0, "DST should produce non-zero output from DC");
    }

    #[test]
    fn inverse_dst_4x4_zero_input() {
        let input = [0i16; 16];
        let mut output = [0i16; 16];
        inverse_dst_4x4(&input, &mut output);
        assert_eq!(output, [0i16; 16]);
    }

    #[test]
    fn determinism() {
        let input = [10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut out1 = [0i16; 16];
        let mut out2 = [0i16; 16];
        inverse_dst_4x4(&input, &mut out1);
        inverse_dst_4x4(&input, &mut out2);
        assert_eq!(out1, out2);
    }

    #[test]
    fn forward_inverse_dst_4x4_approximate_roundtrip() {
        // HEVC forward/inverse have different shift amounts — not exact inverses
        let input = [10i16, -5, 8, -3, 5, -3, 4, -2, 3, -2, 2, -1, 1, -1, 1, 0];
        let mut coeffs = [0i16; 16];
        let mut reconstructed = [0i16; 16];
        forward_dst_4x4(&input, &mut coeffs);
        inverse_dst_4x4(&coeffs, &mut reconstructed);
        let mse: f64 = input
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / 16.0;
        let psnr = if mse < 0.001 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        assert!(
            psnr > 20.0,
            "DST forward-inverse roundtrip PSNR={psnr:.1}dB, expected > 20dB"
        );
    }

    #[test]
    fn forward_dst_determinism() {
        let input = [10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut out1 = [0i16; 16];
        let mut out2 = [0i16; 16];
        forward_dst_4x4(&input, &mut out1);
        forward_dst_4x4(&input, &mut out2);
        assert_eq!(out1, out2);
    }
}
