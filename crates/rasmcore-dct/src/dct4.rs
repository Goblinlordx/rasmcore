//! 4x4 integer DCT (HEVC spec, ITU-T H.265 Table 8-6).
//!
//! Transform matrix (scaled by 64):
//!   [ 64  64  64  64 ]
//!   [ 83  36 -36 -83 ]
//!   [ 64 -64 -64  64 ]
//!   [ 36 -83  83 -36 ]

const C4: [i32; 4] = [64, 83, 64, 36];

/// Forward 4x4 DCT: spatial → frequency domain.
///
/// Input: 4x4 block of i16 residuals (row-major).
/// Output: 4x4 block of i16 DCT coefficients (row-major).
///
/// HEVC-spec forward 4x4 DCT.
///
/// Shift convention per HEVC spec (8-bit):
/// - shift1 = log2(N) + bitDepth - 9 = 2 + 8 - 9 = 1
/// - shift2 = log2(N) + 6 = 2 + 6 = 8
///
/// Note: forward and inverse are NOT exact inverses. The HEVC spec designs them
/// with quantization in between. For direct roundtrip testing, scale the
/// reconstructed values by `1 << (fwd_total - inv_total + 2*log2(N))`.
pub fn forward_dct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    let shift1 = 1; // HEVC spec: log2(4) + 8 - 9 = 1
    let add1 = 1 << (shift1 - 1);
    for i in 0..4 {
        let s = i * 4;
        let x0 = input[s] as i32;
        let x1 = input[s + 1] as i32;
        let x2 = input[s + 2] as i32;
        let x3 = input[s + 3] as i32;

        let e0 = x0 + x3;
        let e1 = x1 + x2;
        let o0 = x0 - x3;
        let o1 = x1 - x2;

        tmp[s] = (C4[0] * e0 + C4[0] * e1 + add1) >> shift1;
        tmp[s + 1] = (C4[1] * o0 + C4[3] * o1 + add1) >> shift1;
        tmp[s + 2] = (C4[0] * e0 - C4[0] * e1 + add1) >> shift1;
        tmp[s + 3] = (C4[3] * o0 - C4[1] * o1 + add1) >> shift1;
    }

    let shift2 = 8; // HEVC spec: log2(4) + 6 = 8
    let add2 = 1 << (shift2 - 1);
    for j in 0..4 {
        let x0 = tmp[j];
        let x1 = tmp[4 + j];
        let x2 = tmp[8 + j];
        let x3 = tmp[12 + j];

        let e0 = x0 + x3;
        let e1 = x1 + x2;
        let o0 = x0 - x3;
        let o1 = x1 - x2;

        output[j] = ((C4[0] * e0 + C4[0] * e1 + add2) >> shift2) as i16;
        output[4 + j] = ((C4[1] * o0 + C4[3] * o1 + add2) >> shift2) as i16;
        output[8 + j] = ((C4[0] * e0 - C4[0] * e1 + add2) >> shift2) as i16;
        output[12 + j] = ((C4[3] * o0 - C4[1] * o1 + add2) >> shift2) as i16;
    }
}

/// Inverse 4x4 DCT: frequency → spatial domain.
///
/// Input: 4x4 block of i16 DCT coefficients (row-major).
/// Output: 4x4 block of i16 residuals (row-major).
pub fn inverse_dct_4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];

    // Row pass (no shift — accumulate)
    for i in 0..4 {
        let s = i * 4;
        let c0 = input[s] as i32;
        let c1 = input[s + 1] as i32;
        let c2 = input[s + 2] as i32;
        let c3 = input[s + 3] as i32;

        let e0 = C4[0] * c0 + C4[0] * c2;
        let e1 = C4[0] * c0 - C4[0] * c2;
        let o0 = C4[1] * c1 + C4[3] * c3;
        let o1 = C4[3] * c1 - C4[1] * c3;

        // Shift by 7 after row pass (HEVC spec: shift1 = 7)
        let add = 64; // 1 << 6
        tmp[s] = (e0 + o0 + add) >> 7;
        tmp[s + 1] = (e1 + o1 + add) >> 7;
        tmp[s + 2] = (e1 - o1 + add) >> 7;
        tmp[s + 3] = (e0 - o0 + add) >> 7;
    }

    // Column pass
    for j in 0..4 {
        let c0 = tmp[j];
        let c1 = tmp[4 + j];
        let c2 = tmp[8 + j];
        let c3 = tmp[12 + j];

        let e0 = C4[0] * c0 + C4[0] * c2;
        let e1 = C4[0] * c0 - C4[0] * c2;
        let o0 = C4[1] * c1 + C4[3] * c3;
        let o1 = C4[3] * c1 - C4[1] * c3;

        // Shift by 12 after column pass (HEVC spec: shift2 = 20 - bit_depth = 12 for 8-bit)
        let add = 1 << 11;
        output[j] = ((e0 + o0 + add) >> 12) as i16;
        output[4 + j] = ((e1 + o1 + add) >> 12) as i16;
        output[8 + j] = ((e1 - o1 + add) >> 12) as i16;
        output[12 + j] = ((e0 - o0 + add) >> 12) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_4x4_identity() {
        let input = [100i16; 16];
        let mut freq = [0i16; 16];
        let mut recon = [0i16; 16];

        forward_dct_4x4(&input, &mut freq);
        inverse_dct_4x4(&freq, &mut recon);

        for i in 0..16 {
            assert!(
                (input[i] - recon[i]).abs() <= 1,
                "mismatch at {i}: input={}, recon={}",
                input[i],
                recon[i]
            );
        }
    }

    #[test]
    fn roundtrip_4x4_gradient() {
        let mut input = [0i16; 16];
        for i in 0..16 {
            input[i] = (i * 10) as i16;
        }
        let mut freq = [0i16; 16];
        let mut recon = [0i16; 16];

        forward_dct_4x4(&input, &mut freq);
        inverse_dct_4x4(&freq, &mut recon);

        for i in 0..16 {
            assert!(
                (input[i] - recon[i]).abs() <= 1,
                "mismatch at {i}: input={}, recon={}",
                input[i],
                recon[i]
            );
        }
    }

    #[test]
    fn dc_only_concentrates_energy() {
        let input = [50i16; 16];
        let mut freq = [0i16; 16];
        forward_dct_4x4(&input, &mut freq);

        // DC coefficient should dominate
        assert!(freq[0].abs() > 0, "DC should be non-zero");
        // AC coefficients should be near zero for flat input
        for i in 1..16 {
            assert!(
                freq[i].abs() <= 1,
                "AC[{i}] should be ~0 for flat input, got {}",
                freq[i]
            );
        }
    }

    #[test]
    fn determinism() {
        let input = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, -10, -20, -30, -40,
        ];
        let mut out1 = [0i16; 16];
        let mut out2 = [0i16; 16];
        forward_dct_4x4(&input, &mut out1);
        forward_dct_4x4(&input, &mut out2);
        assert_eq!(out1, out2);
    }
}
