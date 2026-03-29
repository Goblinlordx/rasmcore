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

/// Forward 16x16 DCT: spatial → frequency domain.
///
/// HEVC-spec forward 16x16 DCT using butterfly decomposition.
///
/// Shift convention per HEVC spec (8-bit):
/// - shift1 = log2(N) + bitDepth - 9 = 4 + 8 - 9 = 3
/// - shift2 = log2(N) + 6 = 4 + 6 = 10
///
/// Ref: x265 4.1 common/dct.cpp — dct16_c()
/// Ref: ITU-T H.265 Table 8-6 (16-point DCT kernel)
pub fn forward_dct_16x16(input: &[i16; 256], output: &mut [i16; 256]) {
    let mut tmp = [0i32; 256];

    // Row pass
    let shift1 = 3; // log2(16) + 8 - 9
    let add1 = 1 << (shift1 - 1);
    for i in 0..16 {
        let s = i * 16;
        let x: Vec<i32> = (0..16).map(|j| input[s + j] as i32).collect();
        let row = fdct16_1d(&x);
        for j in 0..16 {
            tmp[s + j] = (row[j] + add1) >> shift1;
        }
    }

    // Column pass
    let shift2 = 10; // log2(16) + 6
    let add2 = 1 << (shift2 - 1);
    for j in 0..16 {
        let x: Vec<i32> = (0..16).map(|i| tmp[i * 16 + j]).collect();
        let col = fdct16_1d(&x);
        for i in 0..16 {
            output[i * 16 + j] = ((col[i] + add2) >> shift2) as i16;
        }
    }
}

/// 1D 16-point forward DCT butterfly.
///
/// The forward DCT butterfly computes:
///   Y[k] = sum(x[n] * T[k][n]) for n = 0..15
/// where T is the HEVC 16-point DCT matrix.
///
/// Uses the same even/odd decomposition as the inverse, but
/// with input samples decomposed into even (e) and odd (o) parts.
fn fdct16_1d(x: &[i32]) -> [i32; 16] {
    // Stage 1: even/odd decomposition of input
    let mut e = [0i32; 8];
    let mut o = [0i32; 8];
    for k in 0..8 {
        e[k] = x[k] + x[15 - k];
        o[k] = x[k] - x[15 - k];
    }

    // Stage 2: even part → 8-point decomposition
    let mut ee = [0i32; 4];
    let mut eo = [0i32; 4];
    for k in 0..4 {
        ee[k] = e[k] + e[7 - k];
        eo[k] = e[k] - e[7 - k];
    }

    // Stage 3: even-even part → 4-point decomposition
    let eee0 = ee[0] + ee[3];
    let eee1 = ee[1] + ee[2];
    let eeo0 = ee[0] - ee[3];
    let eeo1 = ee[1] - ee[2];

    // Output even-even: indices 0, 8 (DC and half-Nyquist)
    let mut result = [0i32; 16];
    result[0] = E[0] * eee0 + E[0] * eee1;
    result[8] = E[0] * eee0 - E[0] * eee1;
    // Output even-even-odd: indices 4, 12
    result[4] = E[1] * eeo0 + E[3] * eeo1;
    result[12] = E[3] * eeo0 - E[1] * eeo1;

    // Output even-odd: indices 2, 6, 10, 14
    result[2] = O8[0] * eo[0] + O8[1] * eo[1] + O8[2] * eo[2] + O8[3] * eo[3];
    result[6] = O8[1] * eo[0] - O8[3] * eo[1] - O8[0] * eo[2] - O8[2] * eo[3];
    result[10] = O8[2] * eo[0] - O8[0] * eo[1] + O8[3] * eo[2] + O8[1] * eo[3];
    result[14] = O8[3] * eo[0] - O8[2] * eo[1] + O8[1] * eo[2] - O8[0] * eo[3];

    // Output odd: indices 1, 3, 5, 7, 9, 11, 13, 15
    result[1] = O16[0] * o[0]
        + O16[1] * o[1]
        + O16[2] * o[2]
        + O16[3] * o[3]
        + O16[4] * o[4]
        + O16[5] * o[5]
        + O16[6] * o[6]
        + O16[7] * o[7];
    result[3] = O16[1] * o[0] - O16[4] * o[1] - O16[7] * o[2] - O16[5] * o[3]
        - O16[2] * o[4]
        + O16[0] * o[5]
        + O16[3] * o[6]
        + O16[6] * o[7];
    result[5] = O16[2] * o[0] - O16[7] * o[1] - O16[3] * o[2]
        + O16[6] * o[3]
        + O16[0] * o[4]
        + O16[4] * o[5]
        - O16[1] * o[6]
        - O16[5] * o[7];
    result[7] = O16[3] * o[0] - O16[5] * o[1] + O16[6] * o[2] + O16[0] * o[3]
        - O16[7] * o[4]
        - O16[1] * o[5]
        + O16[4] * o[6]
        + O16[2] * o[7];
    result[9] = O16[4] * o[0] - O16[2] * o[1] + O16[0] * o[2] - O16[7] * o[3]
        - O16[3] * o[4]
        + O16[6] * o[5]
        + O16[5] * o[6]
        - O16[1] * o[7];
    result[11] = O16[5] * o[0] - O16[0] * o[1] + O16[4] * o[2] + O16[1] * o[3]
        - O16[6] * o[4]
        - O16[2] * o[5]
        + O16[7] * o[6]
        + O16[3] * o[7];
    result[13] = O16[6] * o[0] - O16[3] * o[1] + O16[1] * o[2] - O16[4] * o[3]
        + O16[5] * o[4]
        + O16[7] * o[5]
        - O16[0] * o[6]
        + O16[2] * o[7];
    result[15] = O16[7] * o[0] - O16[6] * o[1] + O16[5] * o[2] - O16[4] * o[3]
        + O16[3] * o[4]
        - O16[2] * o[5]
        + O16[1] * o[6]
        - O16[0] * o[7];

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

    #[test]
    fn forward_16x16_dc_concentrates_energy() {
        // Flat input should produce large DC and near-zero AC
        let input = [50i16; 256];
        let mut output = [0i16; 256];
        forward_dct_16x16(&input, &mut output);
        // DC coefficient should be non-zero
        assert!(output[0].abs() > 0, "DC should be non-zero for flat input");
        // AC coefficients should be zero for perfectly flat input
        let ac_energy: i64 = output[1..].iter().map(|&v| (v as i64) * (v as i64)).sum();
        assert_eq!(ac_energy, 0, "AC should be zero for flat input");
    }

    #[test]
    fn forward_inverse_16x16_approximate_roundtrip() {
        // HEVC forward/inverse are NOT exact inverses — different shift amounts
        // mean quantization is needed in between for exact reconstruction.
        // This test verifies the transforms are structurally correct by checking
        // that roundtrip error is bounded (not that it's zero).
        let mut input = [0i16; 256];
        for i in 0..256 {
            input[i] = ((i % 16) as i16 - 8) * 4; // smaller range
        }
        let mut coeffs = [0i16; 256];
        let mut reconstructed = [0i16; 256];
        forward_dct_16x16(&input, &mut coeffs);
        inverse_dct_16x16(&coeffs, &mut reconstructed);
        // Verify correlation — same sign and approximate magnitude
        let mse: f64 = input
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / 256.0;
        let psnr = if mse < 0.001 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        assert!(
            psnr > 20.0,
            "forward-inverse roundtrip PSNR={psnr:.1}dB, expected > 20dB"
        );
    }
}
