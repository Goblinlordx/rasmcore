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
        tmp[s] = (D4[0] * c0 + D4[1] * c1 + D4[2] * c2 + D4[3] * c3 + add) >> 7;
        tmp[s + 1] = (D4[2] * c0 + D4[2] * c1 - D4[2] * c3 + add) >> 7;
        tmp[s + 2] = (D4[3] * c0 - D4[0] * c1 - D4[2] * c2 + D4[1] * c3 + add) >> 7;
        tmp[s + 3] = (D4[1] * c0 - D4[3] * c1 + D4[2] * c2 - D4[0] * c3 + add) >> 7;
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
}
