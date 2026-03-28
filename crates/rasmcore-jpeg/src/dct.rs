//! 8x8 Discrete Cosine Transform for JPEG (ITU-T T.81 Section A.3.3).
//!
//! Uses separable row-column approach with fixed-point arithmetic.
//! Initial implementation uses direct computation for correctness;
//! fast algorithms (AAN/Loeffler) will be added in SIMD optimization track.

use std::f64::consts::PI;

/// Precomputed cosine table for 8x8 DCT: C[u][x] = cos((2x+1)*u*pi/16) * 2^14
/// Scaled by 16384 (14-bit precision) for fixed-point multiply.
const DCT_COS: [[i32; 8]; 8] = {
    // cos((2x+1)*u*pi/16) * 16384, rounded
    [
        [16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384], // u=0: all 1.0
        [16069, 13623, 9102, 3196, -3196, -9102, -13623, -16069], // u=1
        [15137, 6270, -6270, -15137, -15137, -6270, 6270, 15137], // u=2
        [13623, -3196, -16069, -9102, 9102, 16069, 3196, -13623], // u=3
        [11585, -11585, -11585, 11585, 11585, -11585, -11585, 11585], // u=4
        [9102, -16069, 3196, 13623, -13623, -3196, 16069, -9102], // u=5
        [6270, -15137, 15137, -6270, -6270, 15137, -15137, 6270], // u=6
        [3196, -9102, 13623, -16069, 16069, -13623, 9102, -3196], // u=7
    ]
};

/// Forward 8x8 DCT: pixel block → frequency coefficients.
///
/// Uses precomputed cosine table with i32 fixed-point arithmetic.
/// Replaces the naive f64+cos() implementation for ~10x speedup.
/// Output scaling matches the f64 reference (JPEG standard normalization).
///
/// Input: 8x8 block of level-shifted samples (subtract 128 for 8-bit).
/// Output: 8x8 DCT coefficients, scaled for JPEG quantization.
pub fn forward_dct(input: &[i16; 64], output: &mut [i32; 64]) {
    // Precision: 14-bit cosine table, divide by 2^14 per pass.
    // Two passes (row+col), so total scale = / 2^28.
    // The f64 version scales by C(u)*C(v)/4 ≈ 1/4 for DC (u=v=0).
    // With 14-bit table: DC term = sum * 16384 * 16384 / 2^28 = sum * 1 ≈ sum.
    // Need to match f64 output: f64_DC = sum * cos(0)^2 / 4 = sum / 4 * (1/sqrt2)^2 = sum/8
    // Actually f64: for u=0, cu=1/sqrt2, sum is /2.0 per pass → total /(2*2)=sum/4 * (1/sqrt2)^2
    // This is getting complicated. Let me use the EXACT f64 scaling.

    // Scaling: the f64 DCT computes: output[v*8+u] = (cu*cv/4) * sum(input[y*8+x] * cos(...))
    // For u=0,v=0: C0 = 1/sqrt(2), so DC = input_sum * (1/sqrt2)^2 / 4 = input_sum / 8
    //
    // With i32 cosine table scaled by S=16384:
    //   row_pass: temp[u] = sum(input[x] * COS[u][x]) (result scaled by S)
    //   col_pass: out[v*8+u] = sum(temp[u] * COS[v][y]) (result scaled by S^2)
    //   Descale: out >> 28 (since S^2 = 2^28)
    //   But COS[0][x] = 16384 = S (not S/sqrt(2)), so DC = input_sum * S^2 >> 28 = input_sum
    //   f64 DC = input_sum / 8. So we need to divide by 8 additionally, or adjust the table.
    //
    // Simpler: use COS table where COS[0][x] = 16384/sqrt(2) ≈ 11585 (includes C(u) scaling).
    // That way the output directly matches the f64 version.

    // Use the COSINE table as-is (includes the 1/sqrt(2) for u=0/v=0).
    // The table has COS[0][x] = 16384 which is cos(0) * 2^14 = 1.0 * 16384.
    // But the f64 DCT applies C(u)/2 per pass where C(0) = 1/sqrt(2).
    // So the table should be: for u=0, values = 16384 / sqrt(2) = 11585.

    // Let me redefine: use table with the /sqrt(2) factor baked in.
    const S: i32 = 16384; // 2^14
    const C0: i32 = 11585; // S / sqrt(2), rounded

    let mut temp = [0i32; 64];

    // Row pass
    for row in 0..8 {
        for u in 0..8 {
            let mut sum: i64 = 0;
            for x in 0..8 {
                sum += input[row * 8 + x] as i64 * DCT_COS[u][x] as i64;
            }
            // Apply C(u) normalization: divide by sqrt(2) for u=0
            let scaled = if u == 0 {
                // C(0) = 1/sqrt(2): multiply by C0/S = 11585/16384
                (sum * C0 as i64) >> 14 // divide by S
            } else {
                sum
            };
            // Descale by S and divide by 2 (the /2 from the DCT formula per pass)
            temp[row * 8 + u] = ((scaled + (1 << 14)) >> 15) as i32; // >> 15 = >> 14 (S) + >> 1 (/2)
        }
    }

    // Column pass
    for col in 0..8 {
        for v in 0..8 {
            let mut sum: i64 = 0;
            for y in 0..8 {
                sum += temp[y * 8 + col] as i64 * DCT_COS[v][y] as i64;
            }
            let scaled = if v == 0 { (sum * C0 as i64) >> 14 } else { sum };
            output[v * 8 + col] = ((scaled + (1 << 14)) >> 15) as i32;
        }
    }
}

/// Inverse 8x8 DCT: frequency coefficients → pixel block.
///
/// Integer fixed-point algorithm matching zune-jpeg (used by the `image` crate).
/// Ported from zune-jpeg's scalar IDCT for bit-exact parity.
///
/// Input: 8x8 DCT coefficients (after dequantization).
/// Output: 8x8 block of pixel values (add 128 to get final 8-bit values).
pub fn inverse_dct(input: &[i32; 64], output: &mut [i16; 64]) {
    // Horizontal pass rounding constant: 512 + 65536 + (128 << 17)
    const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

    #[inline(always)]
    fn fsh(x: i32) -> i32 {
        x << 12
    }

    // Work in-place on a copy
    let mut v = *input;

    // All-zero shortcut
    if v[1..] == [0i32; 63] {
        let coeff = ((v[0].wrapping_add(4).wrapping_add(1024)) >> 3).clamp(0, 255) as i16;
        output.fill(coeff);
        return;
    }

    // Vertical pass (columns)
    for ptr in 0..8 {
        let p2 = v[ptr + 16];
        let p3 = v[ptr + 48];
        let p1 = (p2.wrapping_add(p3)).wrapping_mul(2217);
        let t2 = p1.wrapping_add(p3.wrapping_mul(-7567));
        let t3 = p1.wrapping_add(p2.wrapping_mul(3135));

        let p2 = v[ptr];
        let p3 = v[32 + ptr];
        let t0 = fsh(p2.wrapping_add(p3));
        let t1 = fsh(p2.wrapping_sub(p3));

        let x0 = t0.wrapping_add(t3).wrapping_add(512);
        let x3 = t0.wrapping_sub(t3).wrapping_add(512);
        let x1 = t1.wrapping_add(t2).wrapping_add(512);
        let x2 = t1.wrapping_sub(t2).wrapping_add(512);

        let mut t0 = v[ptr + 56];
        let mut t1 = v[ptr + 40];
        let mut t2 = v[ptr + 24];
        let mut t3 = v[ptr + 8];

        let p3 = t0.wrapping_add(t2);
        let p4 = t1.wrapping_add(t3);
        let p1 = t0.wrapping_add(t3);
        let p2 = t1.wrapping_add(t2);
        let p5 = (p3.wrapping_add(p4)).wrapping_mul(4816);

        t0 = t0.wrapping_mul(1223);
        t1 = t1.wrapping_mul(8410);
        t2 = t2.wrapping_mul(12586);
        t3 = t3.wrapping_mul(6149);

        let p1 = p5.wrapping_add(p1.wrapping_mul(-3685));
        let p2 = p5.wrapping_add(p2.wrapping_mul(-10497));
        let p3 = p3.wrapping_mul(-8034);
        let p4 = p4.wrapping_mul(-1597);

        t3 = t3.wrapping_add(p1.wrapping_add(p4));
        t2 = t2.wrapping_add(p2.wrapping_add(p3));
        t1 = t1.wrapping_add(p2.wrapping_add(p4));
        t0 = t0.wrapping_add(p1.wrapping_add(p3));

        v[ptr] = x0.wrapping_add(t3) >> 10;
        v[ptr + 8] = x1.wrapping_add(t2) >> 10;
        v[ptr + 16] = x2.wrapping_add(t1) >> 10;
        v[ptr + 24] = x3.wrapping_add(t0) >> 10;
        v[ptr + 32] = x3.wrapping_sub(t0) >> 10;
        v[ptr + 40] = x2.wrapping_sub(t1) >> 10;
        v[ptr + 48] = x1.wrapping_sub(t2) >> 10;
        v[ptr + 56] = x0.wrapping_sub(t3) >> 10;
    }

    // Horizontal pass (rows)
    for row in 0..8 {
        let i = row * 8;

        let p2 = v[i + 2];
        let p3 = v[i + 6];
        let p1 = (p2.wrapping_add(p3)).wrapping_mul(2217);
        let t2 = p1.wrapping_add(p3.wrapping_mul(-7567));
        let t3 = p1.wrapping_add(p2.wrapping_mul(3135));

        let p2 = v[i];
        let p3 = v[i + 4];
        let t0 = fsh(p2.wrapping_add(p3));
        let t1 = fsh(p2.wrapping_sub(p3));

        let x0 = t0.wrapping_add(t3).wrapping_add(SCALE_BITS);
        let x3 = t0.wrapping_sub(t3).wrapping_add(SCALE_BITS);
        let x1 = t1.wrapping_add(t2).wrapping_add(SCALE_BITS);
        let x2 = t1.wrapping_sub(t2).wrapping_add(SCALE_BITS);

        let mut t0 = v[i + 7];
        let mut t1 = v[i + 5];
        let mut t2 = v[i + 3];
        let mut t3 = v[i + 1];

        let p3 = t0.wrapping_add(t2);
        let p4 = t1.wrapping_add(t3);
        let p1 = t0.wrapping_add(t3);
        let p2 = t1.wrapping_add(t2);
        let p5 = (p3.wrapping_add(p4)).wrapping_mul(4816); // f2f(1.175875602)

        t0 = t0.wrapping_mul(1223);
        t1 = t1.wrapping_mul(8410);
        t2 = t2.wrapping_mul(12586);
        t3 = t3.wrapping_mul(6149);

        let p1 = p5.wrapping_add(p1.wrapping_mul(-3685));
        let p2 = p5.wrapping_add(p2.wrapping_mul(-10497));
        let p3 = p3.wrapping_mul(-8034);
        let p4 = p4.wrapping_mul(-1597);

        t3 = t3.wrapping_add(p1.wrapping_add(p4));
        t2 = t2.wrapping_add(p2.wrapping_add(p3));
        t1 = t1.wrapping_add(p2.wrapping_add(p4));
        t0 = t0.wrapping_add(p1.wrapping_add(p3));

        output[i] = (x0.wrapping_add(t3) >> 17).clamp(0, 255) as i16;
        output[i + 1] = (x1.wrapping_add(t2) >> 17).clamp(0, 255) as i16;
        output[i + 2] = (x2.wrapping_add(t1) >> 17).clamp(0, 255) as i16;
        output[i + 3] = (x3.wrapping_add(t0) >> 17).clamp(0, 255) as i16;
        output[i + 4] = (x3.wrapping_sub(t0) >> 17).clamp(0, 255) as i16;
        output[i + 5] = (x2.wrapping_sub(t1) >> 17).clamp(0, 255) as i16;
        output[i + 6] = (x1.wrapping_sub(t2) >> 17).clamp(0, 255) as i16;
        output[i + 7] = (x0.wrapping_sub(t3) >> 17).clamp(0, 255) as i16;
    }
}

/// Reference f64 IDCT for encoder validation (forward_dct roundtrip testing).
pub fn inverse_dct_reference(input: &[i32; 64], output: &mut [i16; 64]) {
    let mut temp = [0.0f64; 64];

    for col in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0;
            for v in 0..8 {
                let cv = if v == 0 {
                    1.0 / std::f64::consts::SQRT_2
                } else {
                    1.0
                };
                sum += cv
                    * input[v * 8 + col] as f64
                    * ((2.0 * y as f64 + 1.0) * v as f64 * PI / 16.0).cos();
            }
            temp[y * 8 + col] = sum / 2.0;
        }
    }

    for row in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0;
            for u in 0..8 {
                let cu = if u == 0 {
                    1.0 / std::f64::consts::SQRT_2
                } else {
                    1.0
                };
                sum +=
                    cu * temp[row * 8 + u] * ((2.0 * x as f64 + 1.0) * u as f64 * PI / 16.0).cos();
            }
            output[row * 8 + x] = (sum / 2.0).round() as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_inverse_roundtrip() {
        // Use the f64 reference IDCT for encoder roundtrip validation,
        // since the integer IDCT includes level-shift clamping to [0,255].
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = ((i as i16 * 3 + 7) % 200) - 100;
        }

        let mut dct_out = [0i32; 64];
        forward_dct(&input, &mut dct_out);

        let mut reconstructed = [0i16; 64];
        inverse_dct_reference(&dct_out, &mut reconstructed);

        for i in 0..64 {
            let diff = (input[i] - reconstructed[i]).abs();
            assert!(
                diff <= 1,
                "position {i}: input={}, reconstructed={}, diff={diff}",
                input[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn dc_only_block() {
        let input = [42i16; 64];
        let mut dct_out = [0i32; 64];
        forward_dct(&input, &mut dct_out);

        // DC coefficient should be significant
        assert!(
            dct_out[0].abs() > 100,
            "DC should be significant, got {}",
            dct_out[0]
        );
        // AC coefficients should be near zero
        for i in 1..64 {
            assert!(
                dct_out[i].abs() < 2,
                "AC[{i}] = {} (should be ~0)",
                dct_out[i]
            );
        }
    }

    #[test]
    fn zero_block() {
        let input = [0i16; 64];
        let mut dct_out = [0i32; 64];
        forward_dct(&input, &mut dct_out);
        for v in &dct_out {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn gradient_has_low_freq_energy() {
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = (col as i16 * 16) - 56;
            }
        }
        let mut dct_out = [0i32; 64];
        forward_dct(&input, &mut dct_out);
        assert!(dct_out[1].abs() > dct_out[7].abs());
    }

    #[test]
    fn roundtrip_random_patterns() {
        for seed in 0..50 {
            let mut input = [0i16; 64];
            for i in 0..64 {
                input[i] = (((seed * 17 + i * 31 + 5) % 511) as i16) - 255;
            }
            let mut dct = [0i32; 64];
            forward_dct(&input, &mut dct);
            let mut recon = [0i16; 64];
            inverse_dct_reference(&dct, &mut recon);

            for i in 0..64 {
                let diff = (input[i] - recon[i]).abs();
                assert!(
                    diff <= 1,
                    "seed={seed} pos={i}: {}->{} diff={diff}",
                    input[i],
                    recon[i]
                );
            }
        }
    }

    #[test]
    fn dc_value_is_mean_times_8() {
        // For a flat block of value V, DC = V * 8 (standard normalization)
        let input = [10i16; 64];
        let mut dct_out = [0i32; 64];
        forward_dct(&input, &mut dct_out);
        // DC = 10 * 8 * C(0,0) where C(0,0) = 1/(2*sqrt(2)) * 2 * sum = complex
        // Just verify it's close to 8*10 = 80
        let expected_dc = 80; // 10 * 8 (the 1/4 normalization gives N * value / sqrt(N) * sqrt(N) / N = value * sqrt(N))
        assert!(
            (dct_out[0] - expected_dc).abs() < 5,
            "DC should be ~{expected_dc}, got {}",
            dct_out[0]
        );
    }
}
