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
/// Input: 8x8 DCT coefficients (after dequantization).
/// Output: 8x8 block of pixel values (add 128 to get final 8-bit values).
pub fn inverse_dct(input: &[i32; 64], output: &mut [i16; 64]) {
    let mut temp = [0.0f64; 64];

    // Column pass
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

    // Row pass
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
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = ((i as i16 * 3 + 7) % 200) - 100;
        }

        let mut dct_out = [0i32; 64];
        forward_dct(&input, &mut dct_out);

        let mut reconstructed = [0i16; 64];
        inverse_dct(&dct_out, &mut reconstructed);

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
            inverse_dct(&dct, &mut recon);

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
