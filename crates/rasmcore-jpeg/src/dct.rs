//! 8x8 Discrete Cosine Transform for JPEG (ITU-T T.81 Section A.3.3).
//!
//! Uses separable row-column approach with fixed-point arithmetic.
//! Initial implementation uses direct computation for correctness;
//! fast algorithms (AAN/Loeffler) will be added in SIMD optimization track.

use std::f64::consts::PI;

/// Forward 8x8 DCT: pixel block → frequency coefficients.
///
/// Input: 8x8 block of level-shifted samples (subtract 128 for 8-bit).
/// Output: 8x8 DCT coefficients, scaled for JPEG quantization.
///
/// The output includes the standard JPEG 8x scaling factor
/// (absorbed by quantization tables in normal JPEG encoding).
pub fn forward_dct(input: &[i16; 64], output: &mut [i32; 64]) {
    // Compute using f64 for precision, convert to i32 at the end.
    // This is the reference implementation — fast integer version comes later.
    let mut temp = [0.0f64; 64];

    // Row pass
    for row in 0..8 {
        for u in 0..8 {
            let cu = if u == 0 {
                1.0 / std::f64::consts::SQRT_2
            } else {
                1.0
            };
            let mut sum = 0.0;
            for x in 0..8 {
                sum += input[row * 8 + x] as f64
                    * ((2.0 * x as f64 + 1.0) * u as f64 * PI / 16.0).cos();
            }
            temp[row * 8 + u] = cu * sum / 2.0;
        }
    }

    // Column pass
    for col in 0..8 {
        for v in 0..8 {
            let cv = if v == 0 {
                1.0 / std::f64::consts::SQRT_2
            } else {
                1.0
            };
            let mut sum = 0.0;
            for y in 0..8 {
                sum += temp[y * 8 + col] * ((2.0 * y as f64 + 1.0) * v as f64 * PI / 16.0).cos();
            }
            output[v * 8 + col] = (cv * sum / 2.0).round() as i32;
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
