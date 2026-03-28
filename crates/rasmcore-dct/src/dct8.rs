//! 8x8 integer DCT (HEVC spec, ITU-T H.265 Table 8-6).
//!
//! Uses butterfly decomposition with the HEVC 8-point kernel coefficients.
//! Also compatible with JPEG IDCT when used with appropriate scaling.

/// HEVC 8-point DCT even coefficients (same as 4-point).
const E8: [i32; 4] = [64, 83, 64, 36];

/// HEVC 8-point DCT odd coefficients.
const O8: [i32; 4] = [89, 75, 50, 18];

/// Forward 8x8 DCT: spatial → frequency domain.
///
/// HEVC-spec forward 8x8 DCT.
///
/// Shift convention per HEVC spec (8-bit):
/// - shift1 = log2(N) + bitDepth - 9 = 3 + 8 - 9 = 2
/// - shift2 = log2(N) + 6 = 3 + 6 = 9
pub fn forward_dct_8x8(input: &[i16; 64], output: &mut [i16; 64]) {
    let mut tmp = [0i32; 64];
    let shift1 = 2; // HEVC spec
    let add1 = 1i32 << (shift1 - 1);

    // Row pass
    for i in 0..8 {
        let s = i * 8;
        let x: [i32; 8] = core::array::from_fn(|j| input[s + j] as i32);

        let ee0 = x[0] + x[7];
        let ee1 = x[1] + x[6];
        let ee2 = x[2] + x[5];
        let ee3 = x[3] + x[4];
        let eo0 = x[0] - x[7];
        let eo1 = x[1] - x[6];
        let eo2 = x[2] - x[5];
        let eo3 = x[3] - x[4];

        let eee0 = ee0 + ee3;
        let eee1 = ee1 + ee2;
        let eeo0 = ee0 - ee3;
        let eeo1 = ee1 - ee2;

        tmp[s] = (E8[0] * eee0 + E8[0] * eee1 + add1) >> shift1;
        tmp[s + 4] = (E8[0] * eee0 - E8[0] * eee1 + add1) >> shift1;
        tmp[s + 2] = (E8[1] * eeo0 + E8[3] * eeo1 + add1) >> shift1;
        tmp[s + 6] = (E8[3] * eeo0 - E8[1] * eeo1 + add1) >> shift1;

        tmp[s + 1] = (O8[0] * eo0 + O8[1] * eo1 + O8[2] * eo2 + O8[3] * eo3 + add1) >> shift1;
        tmp[s + 3] = (O8[1] * eo0 - O8[3] * eo1 - O8[0] * eo2 - O8[2] * eo3 + add1) >> shift1;
        tmp[s + 5] = (O8[2] * eo0 - O8[0] * eo1 + O8[3] * eo2 + O8[1] * eo3 + add1) >> shift1;
        tmp[s + 7] = (O8[3] * eo0 - O8[2] * eo1 + O8[1] * eo2 - O8[0] * eo3 + add1) >> shift1;
    }

    let shift2 = 9; // HEVC spec: log2(8) + 6 = 9
    let add2 = 1i32 << (shift2 - 1);

    // Column pass
    for j in 0..8 {
        let x: [i32; 8] = core::array::from_fn(|i| tmp[i * 8 + j]);

        let ee0 = x[0] + x[7];
        let ee1 = x[1] + x[6];
        let ee2 = x[2] + x[5];
        let ee3 = x[3] + x[4];
        let eo0 = x[0] - x[7];
        let eo1 = x[1] - x[6];
        let eo2 = x[2] - x[5];
        let eo3 = x[3] - x[4];

        let eee0 = ee0 + ee3;
        let eee1 = ee1 + ee2;
        let eeo0 = ee0 - ee3;
        let eeo1 = ee1 - ee2;

        output[j] = ((E8[0] * eee0 + E8[0] * eee1 + add2) >> shift2) as i16;
        output[32 + j] = ((E8[0] * eee0 - E8[0] * eee1 + add2) >> shift2) as i16;
        output[16 + j] = ((E8[1] * eeo0 + E8[3] * eeo1 + add2) >> shift2) as i16;
        output[48 + j] = ((E8[3] * eeo0 - E8[1] * eeo1 + add2) >> shift2) as i16;

        output[8 + j] =
            ((O8[0] * eo0 + O8[1] * eo1 + O8[2] * eo2 + O8[3] * eo3 + add2) >> shift2) as i16;
        output[24 + j] =
            ((O8[1] * eo0 - O8[3] * eo1 - O8[0] * eo2 - O8[2] * eo3 + add2) >> shift2) as i16;
        output[40 + j] =
            ((O8[2] * eo0 - O8[0] * eo1 + O8[3] * eo2 + O8[1] * eo3 + add2) >> shift2) as i16;
        output[56 + j] =
            ((O8[3] * eo0 - O8[2] * eo1 + O8[1] * eo2 - O8[0] * eo3 + add2) >> shift2) as i16;
    }
}

/// Inverse 8x8 DCT: frequency → spatial domain.
pub fn inverse_dct_8x8(input: &[i16; 64], output: &mut [i16; 64]) {
    let mut tmp = [0i32; 64];

    // Row pass
    for i in 0..8 {
        let s = i * 8;
        let c: [i32; 8] = core::array::from_fn(|j| input[s + j] as i32);

        // Even part
        let eeo0 = E8[1] * c[2] + E8[3] * c[6];
        let eeo1 = E8[3] * c[2] - E8[1] * c[6];
        let eee0 = E8[0] * c[0] + E8[0] * c[4];
        let eee1 = E8[0] * c[0] - E8[0] * c[4];

        let e0 = eee0 + eeo0;
        let e3 = eee0 - eeo0;
        let e1 = eee1 + eeo1;
        let e2 = eee1 - eeo1;

        // Odd part
        let o0 = O8[0] * c[1] + O8[1] * c[3] + O8[2] * c[5] + O8[3] * c[7];
        let o1 = O8[1] * c[1] - O8[3] * c[3] - O8[0] * c[5] - O8[2] * c[7];
        let o2 = O8[2] * c[1] - O8[0] * c[3] + O8[3] * c[5] + O8[1] * c[7];
        let o3 = O8[3] * c[1] - O8[2] * c[3] + O8[1] * c[5] - O8[0] * c[7];

        let add = 64; // 1 << 6
        tmp[s] = (e0 + o0 + add) >> 7;
        tmp[s + 1] = (e1 + o1 + add) >> 7;
        tmp[s + 2] = (e2 + o2 + add) >> 7;
        tmp[s + 3] = (e3 + o3 + add) >> 7;
        tmp[s + 4] = (e3 - o3 + add) >> 7;
        tmp[s + 5] = (e2 - o2 + add) >> 7;
        tmp[s + 6] = (e1 - o1 + add) >> 7;
        tmp[s + 7] = (e0 - o0 + add) >> 7;
    }

    // Column pass
    for j in 0..8 {
        let c: [i32; 8] = core::array::from_fn(|i| tmp[i * 8 + j]);

        let eeo0 = E8[1] * c[2] + E8[3] * c[6];
        let eeo1 = E8[3] * c[2] - E8[1] * c[6];
        let eee0 = E8[0] * c[0] + E8[0] * c[4];
        let eee1 = E8[0] * c[0] - E8[0] * c[4];

        let e0 = eee0 + eeo0;
        let e3 = eee0 - eeo0;
        let e1 = eee1 + eeo1;
        let e2 = eee1 - eeo1;

        let o0 = O8[0] * c[1] + O8[1] * c[3] + O8[2] * c[5] + O8[3] * c[7];
        let o1 = O8[1] * c[1] - O8[3] * c[3] - O8[0] * c[5] - O8[2] * c[7];
        let o2 = O8[2] * c[1] - O8[0] * c[3] + O8[3] * c[5] + O8[1] * c[7];
        let o3 = O8[3] * c[1] - O8[2] * c[3] + O8[1] * c[5] - O8[0] * c[7];

        let add = 1 << 11;
        output[j] = ((e0 + o0 + add) >> 12) as i16;
        output[8 + j] = ((e1 + o1 + add) >> 12) as i16;
        output[16 + j] = ((e2 + o2 + add) >> 12) as i16;
        output[24 + j] = ((e3 + o3 + add) >> 12) as i16;
        output[32 + j] = ((e3 - o3 + add) >> 12) as i16;
        output[40 + j] = ((e2 - o2 + add) >> 12) as i16;
        output[48 + j] = ((e1 - o1 + add) >> 12) as i16;
        output[56 + j] = ((e0 - o0 + add) >> 12) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_8x8_flat() {
        let input = [50i16; 64];
        let mut freq = [0i16; 64];
        let mut recon = [0i16; 64];
        forward_dct_8x8(&input, &mut freq);
        inverse_dct_8x8(&freq, &mut recon);
        for i in 0..64 {
            assert!(
                (input[i] - recon[i]).abs() <= 1,
                "at {i}: {} vs {}",
                input[i],
                recon[i]
            );
        }
    }

    #[test]
    fn roundtrip_8x8_gradient() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = (i * 3) as i16;
        }
        let mut freq = [0i16; 64];
        let mut recon = [0i16; 64];
        forward_dct_8x8(&input, &mut freq);
        inverse_dct_8x8(&freq, &mut recon);
        for i in 0..64 {
            assert!(
                (input[i] - recon[i]).abs() <= 1,
                "at {i}: {} vs {}",
                input[i],
                recon[i]
            );
        }
    }

    #[test]
    fn dc_only_8x8() {
        let input = [25i16; 64];
        let mut freq = [0i16; 64];
        forward_dct_8x8(&input, &mut freq);
        assert!(freq[0].abs() > 0);
        for i in 1..64 {
            assert!(freq[i].abs() <= 1, "AC[{i}] = {}", freq[i]);
        }
    }
}
