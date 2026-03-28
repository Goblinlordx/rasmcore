//! Quantization and dequantization for JPEG (ITU-T T.81 Section A.3.4).
//!
//! Provides:
//! - Standard luminance/chrominance quantization tables (Annex K)
//! - Quality (1-100) to table scaling (libjpeg formula)
//! - Quantize/dequantize functions
//! - Zigzag scan order

/// Zigzag scan order for 8x8 block.
/// Maps sequential coefficient index to zigzag position.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag: maps zigzag position to sequential index.
pub const UNZIGZAG: [usize; 64] = {
    let mut table = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        table[ZIGZAG[i]] = i;
        i += 1;
    }
    table
};

/// Standard luminance quantization table (ITU-T T.81 Annex K, Table K.1).
pub const STD_LUMA_QUANT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard chrominance quantization table (ITU-T T.81 Annex K, Table K.2).
pub const STD_CHROMA_QUANT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Scale a base quantization table by quality factor.
///
/// Uses the libjpeg formula:
/// - quality < 50: scale = 5000 / quality
/// - quality >= 50: scale = 200 - 2 * quality
/// - table[i] = clamp((base[i] * scale + 50) / 100, 1, max)
///
/// For 8-bit precision, max = 255. For 12-bit, max = 4095.
pub fn scale_quant_table(base: &[u16; 64], quality: u8, twelve_bit: bool) -> [u16; 64] {
    let q = (quality.clamp(1, 100)) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let max_val: u16 = if twelve_bit { 4095 } else { 255 };

    let mut table = [0u16; 64];
    for i in 0..64 {
        let val = ((base[i] as u32 * scale + 50) / 100)
            .max(1)
            .min(max_val as u32);
        table[i] = val as u16;
    }
    table
}

/// Build luminance quantization table for the given quality.
pub fn luma_quant_table(quality: u8, twelve_bit: bool) -> [u16; 64] {
    scale_quant_table(&STD_LUMA_QUANT, quality, twelve_bit)
}

/// Build chrominance quantization table for the given quality.
pub fn chroma_quant_table(quality: u8, twelve_bit: bool) -> [u16; 64] {
    scale_quant_table(&STD_CHROMA_QUANT, quality, twelve_bit)
}

/// Quantize a block of 64 DCT coefficients.
///
/// output[i] = round(input[i] / quant_table[i])
///
/// Uses integer division with rounding toward nearest.
#[inline]
pub fn quantize(coeffs: &[i32; 64], quant_table: &[u16; 64], output: &mut [i16; 64]) {
    for i in 0..64 {
        let q = quant_table[i] as i32;
        // Round to nearest: (coeff + q/2) / q for positive, (coeff - q/2) / q for negative
        let c = coeffs[i];
        output[i] = if c >= 0 {
            ((c + q / 2) / q) as i16
        } else {
            ((c - q / 2) / q) as i16
        };
    }
}

/// Dequantize a block of 64 quantized coefficients.
///
/// output[i] = input[i] * quant_table[i]
#[inline]
pub fn dequantize(coeffs: &[i16; 64], quant_table: &[u16; 64], output: &mut [i32; 64]) {
    for i in 0..64 {
        output[i] = coeffs[i] as i32 * quant_table[i] as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_covers_all_positions() {
        let mut seen = [false; 64];
        for &pos in &ZIGZAG {
            assert!(!seen[pos], "duplicate zigzag position {pos}");
            seen[pos] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "zigzag must cover all 64 positions"
        );
    }

    #[test]
    fn unzigzag_is_inverse_of_zigzag() {
        for i in 0..64 {
            assert_eq!(UNZIGZAG[ZIGZAG[i]], i);
        }
    }

    #[test]
    fn quality_50_preserves_standard_tables() {
        // At quality 50, scale = 200 - 100 = 100, so table is unchanged
        let table = luma_quant_table(50, false);
        for i in 0..64 {
            assert_eq!(
                table[i],
                STD_LUMA_QUANT[i].min(255),
                "mismatch at position {i}"
            );
        }
    }

    #[test]
    fn quality_100_gives_all_ones() {
        // At quality 100, scale = 0, so all values clamp to 1
        let table = luma_quant_table(100, false);
        for i in 0..64 {
            assert_eq!(
                table[i], 1,
                "q100 should give all 1s, got {} at {i}",
                table[i]
            );
        }
    }

    #[test]
    fn quality_1_gives_large_values() {
        // At quality 1, scale = 5000, so values are very large (clamped to 255)
        let table = luma_quant_table(1, false);
        for i in 0..64 {
            assert!(table[i] >= 1);
            assert!(table[i] <= 255);
        }
        // Most values should be at max
        let max_count = table.iter().filter(|&&v| v == 255).count();
        assert!(max_count > 50, "most values should be 255 at q1");
    }

    #[test]
    fn quality_25_matches_libjpeg() {
        // At q25: scale = 5000/25 = 200
        // Position 0: (16 * 200 + 50) / 100 = 32.5 → 32
        let table = luma_quant_table(25, false);
        assert_eq!(table[0], 32);
    }

    #[test]
    fn quality_75_matches_libjpeg() {
        // At q75: scale = 200 - 150 = 50
        // Position 0: (16 * 50 + 50) / 100 = 8.5 → 8
        let table = luma_quant_table(75, false);
        assert_eq!(table[0], 8);
    }

    #[test]
    fn quality_95_matches_libjpeg() {
        // At q95: scale = 200 - 190 = 10
        // Position 0: (16 * 10 + 50) / 100 = 2.1 → 2
        let table = luma_quant_table(95, false);
        assert_eq!(table[0], 2);
    }

    #[test]
    fn twelve_bit_allows_larger_values() {
        let table_8 = luma_quant_table(1, false);
        let table_12 = luma_quant_table(1, true);
        // 12-bit table can have values up to 4095
        assert!(table_12.iter().any(|&v| v > 255));
        // 8-bit table capped at 255
        assert!(table_8.iter().all(|&v| v <= 255));
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let table = luma_quant_table(50, false);
        let mut coeffs = [0i32; 64];
        for i in 0..64 {
            coeffs[i] = (i as i32 * 7 - 200).clamp(-1024, 1023);
        }

        let mut quantized = [0i16; 64];
        quantize(&coeffs, &table, &mut quantized);

        let mut dequantized = [0i32; 64];
        dequantize(&quantized, &table, &mut dequantized);

        // Dequantized should be within quant_table[i] of original
        for i in 0..64 {
            let diff = (coeffs[i] - dequantized[i]).abs();
            assert!(
                diff <= table[i] as i32,
                "position {i}: diff {diff} > quant {}",
                table[i]
            );
        }
    }

    #[test]
    fn quantize_zero_block() {
        let table = luma_quant_table(50, false);
        let coeffs = [0i32; 64];
        let mut quantized = [0i16; 64];
        quantize(&coeffs, &table, &mut quantized);

        for v in &quantized {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn quantize_preserves_sign() {
        let table = [10u16; 64];
        let mut coeffs = [0i32; 64];
        coeffs[0] = 50;
        coeffs[1] = -50;

        let mut quantized = [0i16; 64];
        quantize(&coeffs, &table, &mut quantized);

        assert_eq!(quantized[0], 5);
        assert_eq!(quantized[1], -5);
    }
}
