//! 16x16 integer inverse DCT (HEVC spec, ITU-T H.265 Table 8-6).

/// Even coefficients (same as 8-point even).
const E: [i32; 4] = [64, 83, 64, 36];
/// 8-point odd coefficients.
const O8: [i32; 4] = [89, 75, 50, 18];
/// 16-point odd coefficients.
const O16: [i32; 8] = [90, 87, 80, 70, 57, 43, 25, 9];

/// Inverse 16x16 DCT: frequency → spatial domain.
pub fn inverse_dct_16x16(input: &[i16; 256], output: &mut [i16; 256]) {
    let mut tmp = [0i32; 256];

    // Row pass
    for i in 0..16 {
        let s = i * 16;
        let c: Vec<i32> = (0..16).map(|j| input[s + j] as i32).collect();
        let row = idct16_1d(&c);
        let add = 64;
        for j in 0..16 {
            tmp[s + j] = (row[j] + add) >> 7;
        }
    }

    // Column pass
    for j in 0..16 {
        let c: Vec<i32> = (0..16).map(|i| tmp[i * 16 + j]).collect();
        let col = idct16_1d(&c);
        let add = 1 << 11;
        for i in 0..16 {
            output[i * 16 + j] = ((col[i] + add) >> 12) as i16;
        }
    }
}

/// 1D 16-point inverse DCT butterfly.
fn idct16_1d(c: &[i32]) -> [i32; 16] {
    // Odd part (8 outputs from odd-indexed inputs)
    let o = [
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

    // Even part — 8-point IDCT on even-indexed inputs
    let eo = [
        O8[0] * c[2] + O8[1] * c[6] + O8[2] * c[10] + O8[3] * c[14],
        O8[1] * c[2] - O8[3] * c[6] - O8[0] * c[10] - O8[2] * c[14],
        O8[2] * c[2] - O8[0] * c[6] + O8[3] * c[10] + O8[1] * c[14],
        O8[3] * c[2] - O8[2] * c[6] + O8[1] * c[10] - O8[0] * c[14],
    ];

    let eeo0 = E[1] * c[4] + E[3] * c[12];
    let eeo1 = E[3] * c[4] - E[1] * c[12];
    let eee0 = E[0] * c[0] + E[0] * c[8];
    let eee1 = E[0] * c[0] - E[0] * c[8];

    let ee = [eee0 + eeo0, eee1 + eeo1, eee1 - eeo1, eee0 - eeo0];

    let mut e = [0i32; 8];
    for k in 0..4 {
        e[k] = ee[k] + eo[k];
        e[7 - k] = ee[k] - eo[k];
    }

    let mut result = [0i32; 16];
    for k in 0..8 {
        result[k] = e[k] + o[k];
        result[15 - k] = e[k] - o[k];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_16x16_zero_input() {
        let input = [0i16; 256];
        let mut output = [0i16; 256];
        inverse_dct_16x16(&input, &mut output);
        assert_eq!(output, [0i16; 256]);
    }

    #[test]
    fn inverse_16x16_dc_only() {
        let mut input = [0i16; 256];
        input[0] = 100;
        let mut output = [0i16; 256];
        inverse_dct_16x16(&input, &mut output);
        // All values should be similar (DC distributes evenly)
        let first = output[0];
        for v in &output {
            assert!((*v - first).abs() <= 1, "DC should distribute evenly");
        }
    }

    #[test]
    fn determinism_16x16() {
        let mut input = [0i16; 256];
        for i in 0..256 {
            input[i] = ((i as i16 * 7) % 100) - 50;
        }
        let mut out1 = [0i16; 256];
        let mut out2 = [0i16; 256];
        inverse_dct_16x16(&input, &mut out1);
        inverse_dct_16x16(&input, &mut out2);
        assert_eq!(out1, out2);
    }
}
