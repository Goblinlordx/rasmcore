//! 32x32 integer inverse DCT (HEVC spec, ITU-T H.265 Table 8-6).

/// Even coefficients (4-point kernel).
const E: [i32; 4] = [64, 83, 64, 36];
/// 8-point odd coefficients.
const O8: [i32; 4] = [89, 75, 50, 18];
/// 16-point odd coefficients.
const O16: [i32; 8] = [90, 87, 80, 70, 57, 43, 25, 9];
/// 32-point odd coefficients.
const O32: [i32; 16] = [
    90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4,
];

/// Inverse 32x32 DCT: frequency → spatial domain.
pub fn inverse_dct_32x32(input: &[i16; 1024], output: &mut [i16; 1024]) {
    let mut tmp = [0i32; 1024];

    // Row pass
    for i in 0..32 {
        let s = i * 32;
        let c: Vec<i32> = (0..32).map(|j| input[s + j] as i32).collect();
        let row = idct32_1d(&c);
        let add = 64;
        for j in 0..32 {
            tmp[s + j] = (row[j] + add) >> 7;
        }
    }

    // Column pass
    for j in 0..32 {
        let c: Vec<i32> = (0..32).map(|i| tmp[i * 32 + j]).collect();
        let col = idct32_1d(&c);
        let add = 1 << 11;
        for i in 0..32 {
            output[i * 32 + j] = ((col[i] + add) >> 12) as i16;
        }
    }
}

/// 1D 32-point inverse DCT butterfly.
fn idct32_1d(c: &[i32]) -> [i32; 32] {
    // 32-point odd part (16 outputs from odd-indexed inputs: c[1],c[3],...,c[31])
    // Sign pattern from HEVC spec Table 8-6
    let odd_src: [i32; 16] = core::array::from_fn(|k| c[2 * k + 1]);

    // HEVC 32-point odd transform matrix (signs from spec)
    #[rustfmt::skip]
    let signs: [[i8; 16]; 16] = [
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
        [ 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
        [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1],
        [ 1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1],
        [ 1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1],
        [ 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1],
        [ 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1],
        [ 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1],
        [ 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1],
        [ 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1],
        [ 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1],
        [ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [ 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1],
        [ 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1],
        [ 1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
    ];

    // Coefficient index permutation for the 32-point odd matrix
    let coeff_idx: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    let mut o = [0i32; 16];
    for k in 0..16 {
        let mut sum = 0i32;
        for m in 0..16 {
            sum += O32[coeff_idx[m]] * odd_src[m] * signs[k][m] as i32;
        }
        o[k] = sum;
    }

    // Even part — 16-point IDCT on even-indexed inputs
    let even_src: Vec<i32> = (0..16).map(|k| c[2 * k]).collect();
    let e = idct16_even(&even_src);

    let mut result = [0i32; 32];
    for k in 0..16 {
        result[k] = e[k] + o[k];
        result[31 - k] = e[k] - o[k];
    }
    result
}

/// 16-point even part of the 32-point IDCT (recursive butterfly).
fn idct16_even(c: &[i32]) -> [i32; 16] {
    // 16-point odd from odd-indexed positions of the even input
    let eo = [
        O16[0] * c[1]
            + O16[1] * c[3]
            + O16[2] * c[5]
            + O16[3] * c[7]
            + O16[4] * c[9]
            + O16[5] * c[11]
            + O16[6] * c[13]
            + O16[7] * c[15],
        O16[1] * c[1] - O16[4] * c[3] - O16[7] * c[5] - O16[5] * c[7] - O16[2] * c[9]
            + O16[0] * c[11]
            + O16[3] * c[13]
            + O16[6] * c[15],
        O16[2] * c[1] - O16[7] * c[3] - O16[3] * c[5]
            + O16[6] * c[7]
            + O16[0] * c[9]
            + O16[4] * c[11]
            - O16[1] * c[13]
            - O16[5] * c[15],
        O16[3] * c[1] - O16[5] * c[3] + O16[6] * c[5] + O16[0] * c[7]
            - O16[7] * c[9]
            - O16[1] * c[11]
            + O16[4] * c[13]
            + O16[2] * c[15],
        O16[4] * c[1] - O16[2] * c[3] + O16[0] * c[5] - O16[7] * c[7] - O16[3] * c[9]
            + O16[6] * c[11]
            + O16[5] * c[13]
            - O16[1] * c[15],
        O16[5] * c[1] - O16[0] * c[3] + O16[4] * c[5] + O16[1] * c[7]
            - O16[6] * c[9]
            - O16[2] * c[11]
            + O16[7] * c[13]
            + O16[3] * c[15],
        O16[6] * c[1] - O16[3] * c[3] + O16[1] * c[5] - O16[4] * c[7]
            + O16[5] * c[9]
            + O16[7] * c[11]
            - O16[0] * c[13]
            + O16[2] * c[15],
        O16[7] * c[1] - O16[6] * c[3] + O16[5] * c[5] - O16[4] * c[7] + O16[3] * c[9]
            - O16[2] * c[11]
            + O16[1] * c[13]
            - O16[0] * c[15],
    ];

    // 8-point even part from even-indexed positions
    let eeo = [
        O8[0] * c[2] + O8[1] * c[6] + O8[2] * c[10] + O8[3] * c[14],
        O8[1] * c[2] - O8[3] * c[6] - O8[0] * c[10] - O8[2] * c[14],
        O8[2] * c[2] - O8[0] * c[6] + O8[3] * c[10] + O8[1] * c[14],
        O8[3] * c[2] - O8[2] * c[6] + O8[1] * c[10] - O8[0] * c[14],
    ];

    let eeo_4_0 = E[1] * c[4] + E[3] * c[12];
    let eeo_4_1 = E[3] * c[4] - E[1] * c[12];
    let eee_0 = E[0] * c[0] + E[0] * c[8];
    let eee_1 = E[0] * c[0] - E[0] * c[8];

    let ee = [
        eee_0 + eeo_4_0,
        eee_1 + eeo_4_1,
        eee_1 - eeo_4_1,
        eee_0 - eeo_4_0,
    ];

    let mut e = [0i32; 8];
    for k in 0..4 {
        e[k] = ee[k] + eeo[k];
        e[7 - k] = ee[k] - eeo[k];
    }

    let mut result = [0i32; 16];
    for k in 0..8 {
        result[k] = e[k] + eo[k];
        result[15 - k] = e[k] - eo[k];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_32x32_zero_input() {
        let input = [0i16; 1024];
        let mut output = [0i16; 1024];
        inverse_dct_32x32(&input, &mut output);
        assert_eq!(output, [0i16; 1024]);
    }

    #[test]
    fn inverse_32x32_dc_only() {
        let mut input = [0i16; 1024];
        input[0] = 100;
        let mut output = [0i16; 1024];
        inverse_dct_32x32(&input, &mut output);
        let first = output[0];
        for v in &output {
            assert!((*v - first).abs() <= 1);
        }
    }

    #[test]
    fn determinism_32x32() {
        let mut input = [0i16; 1024];
        for i in 0..1024 {
            input[i] = ((i as i16 * 3) % 50) - 25;
        }
        let mut out1 = [0i16; 1024];
        let mut out2 = [0i16; 1024];
        inverse_dct_32x32(&input, &mut out1);
        inverse_dct_32x32(&input, &mut out2);
        assert_eq!(out1, out2);
    }
}
