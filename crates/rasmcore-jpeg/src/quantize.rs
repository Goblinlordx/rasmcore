//! Quantization for JPEG (ITU-T T.81 Section A.3.4).
//!
//! 9 table presets from mozjpeg research. Default: Robidoux (mozjpeg/ImageMagick).
//! Sources documented per-table with paper references.

/// Zigzag scan order for 8x8 block (ITU-T T.81 Figure A.6).
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag: maps zigzag position to sequential index.
pub const UNZIGZAG: [usize; 64] = {
    let mut t = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        t[ZIGZAG[i]] = i;
        i += 1;
    }
    t
};

/// Quantization table preset. Default: Robidoux (mozjpeg/ImageMagick default).
///
/// Sources: mozjpeg jcparam.c, via zenjpeg encode/tables/presets.rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantPreset {
    /// Nicolas Robidoux — mozjpeg/ImageMagick default. Psychovisually optimized.
    #[default]
    Robidoux,
    /// ITU-T T.81 Annex K Tables K.1/K.2. Standard libjpeg default.
    AnnexK,
    /// Uniform quantization (all 16). Testing/debug.
    Flat,
    /// MSSIM-optimized on Kodak image set. (mozjpeg research)
    MssimTuned,
    /// PSNR-HVS-M tuned. (mozjpeg research)
    PsnrHvsM,
    /// Klein, Silverstein, Carney 1992. "Image Quality and Lambda for DCT-Based Lossy Coder"
    Klein,
    /// Watson, Taylor, Borthwick 1997. "DCTune: Visual Optimization of DCT Quantization"
    Watson,
    /// Ahumada, Watson, Peterson 1993. "A Visual Detection Model for DCT Coefficient Quantization"
    Ahumada,
    /// Peterson, Ahumada, Watson 1993. "An Improved Detection Model for DCT Coefficient Quantization"
    Peterson,
}

impl QuantPreset {
    pub const IMAGE_MAGICK: Self = Self::Robidoux;
}

pub fn preset_luma_table(p: QuantPreset) -> &'static [u16; 64] {
    &LUMA[p as usize]
}
pub fn preset_chroma_table(p: QuantPreset) -> &'static [u16; 64] {
    &CHROMA[p as usize]
}

// Luminance tables (9 variants). Source: mozjpeg jcparam.c via zenjpeg.
const LUMA: [[u16; 64]; 9] = [
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ], // Robidoux
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ], // Annex K
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ], // Flat
    [
        12, 17, 20, 21, 30, 34, 56, 63, 18, 20, 20, 26, 28, 51, 61, 55, 19, 20, 21, 26, 33, 58, 69,
        55, 26, 26, 26, 30, 46, 87, 86, 66, 31, 33, 36, 40, 46, 96, 100, 73, 40, 35, 46, 62, 81,
        100, 111, 91, 46, 66, 76, 86, 102, 121, 120, 101, 68, 90, 90, 96, 113, 102, 105, 103,
    ], // MSSIM
    [
        9, 10, 12, 14, 27, 32, 51, 62, 11, 12, 14, 19, 27, 44, 59, 73, 12, 14, 18, 25, 42, 59, 79,
        78, 17, 18, 25, 42, 61, 92, 87, 92, 23, 28, 42, 75, 79, 112, 112, 99, 40, 42, 59, 84, 88,
        124, 132, 111, 42, 64, 78, 95, 105, 126, 125, 99, 70, 75, 100, 102, 116, 100, 107, 98,
    ], // PSNR-HVS-M
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ], // Klein
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ], // Watson
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ], // Ahumada
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ], // Peterson
];

// Chrominance tables (9 variants). Source: mozjpeg jcparam.c via zenjpeg.
const CHROMA: [[u16; 64]; 9] = [
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ], // Robidoux (=luma)
    [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
        99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ], // Annex K
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ], // Flat
    [
        8, 12, 15, 15, 86, 96, 96, 98, 13, 13, 15, 26, 90, 96, 99, 98, 12, 15, 18, 96, 99, 99, 99,
        99, 17, 16, 90, 96, 99, 99, 99, 99, 96, 96, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ], // MSSIM
    [
        9, 10, 17, 19, 62, 89, 91, 97, 12, 13, 18, 29, 84, 91, 88, 98, 14, 19, 29, 93, 95, 95, 98,
        97, 20, 26, 84, 88, 95, 95, 98, 94, 26, 86, 91, 93, 97, 99, 98, 99, 99, 100, 98, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 97, 97, 99, 99, 99, 99, 97, 99,
    ], // PSNR-HVS-M
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ], // Klein (=luma)
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ], // Watson (=luma)
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ], // Ahumada (=luma)
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ], // Peterson (=luma)
];

/// Scale a base quantization table by quality (libjpeg/mozjpeg formula).
pub fn scale_quant_table(base: &[u16; 64], quality: u8, twelve_bit: bool) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let max_val = if twelve_bit { 4095u32 } else { 255 };
    let mut table = [0u16; 64];
    for i in 0..64 {
        table[i] = ((base[i] as u32 * scale + 50) / 100).clamp(1, max_val) as u16;
    }
    table
}

pub fn luma_quant_table(quality: u8, preset: QuantPreset, twelve_bit: bool) -> [u16; 64] {
    scale_quant_table(preset_luma_table(preset), quality, twelve_bit)
}

pub fn chroma_quant_table(quality: u8, preset: QuantPreset, twelve_bit: bool) -> [u16; 64] {
    scale_quant_table(preset_chroma_table(preset), quality, twelve_bit)
}

/// Quantize DCT coefficients. Divisor is Q*8 because the LL&M forward
/// DCT leaves output scaled by 8 (libjpeg/mozjpeg convention).
#[inline]
pub fn quantize(coeffs: &[i32; 64], qt: &[u16; 64], out: &mut [i16; 64]) {
    for i in 0..64 {
        let q = qt[i] as i32 * 8; // Q*8 absorbs DCT scale factor
        let c = coeffs[i];
        out[i] = if c >= 0 {
            ((c + q / 2) / q) as i16
        } else {
            ((c - q / 2) / q) as i16
        };
    }
}

#[inline]
pub fn dequantize(coeffs: &[i16; 64], qt: &[u16; 64], out: &mut [i32; 64]) {
    for i in 0..64 {
        out[i] = coeffs[i] as i32 * qt[i] as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_covers_all() {
        let mut s = [false; 64];
        for &p in &ZIGZAG {
            assert!(!s[p]);
            s[p] = true;
        }
        assert!(s.iter().all(|&x| x));
    }
    #[test]
    fn unzigzag_inverts() {
        for i in 0..64 {
            assert_eq!(UNZIGZAG[ZIGZAG[i]], i);
        }
    }
    #[test]
    fn default_is_robidoux() {
        assert_eq!(QuantPreset::default(), QuantPreset::Robidoux);
    }
    #[test]
    fn robidoux_high_freq() {
        assert!(preset_luma_table(QuantPreset::Robidoux)[63] > 300);
    }
    #[test]
    fn robidoux_chroma_eq_luma() {
        assert_eq!(
            preset_luma_table(QuantPreset::Robidoux),
            preset_chroma_table(QuantPreset::Robidoux)
        );
    }
    #[test]
    fn flat_uniform() {
        for &v in preset_luma_table(QuantPreset::Flat) {
            assert_eq!(v, 16);
        }
    }
    #[test]
    fn annex_k_dc() {
        assert_eq!(preset_luma_table(QuantPreset::AnnexK)[0], 16);
        assert_eq!(preset_chroma_table(QuantPreset::AnnexK)[0], 17);
    }
    #[test]
    fn q50_preserves() {
        let b = preset_luma_table(QuantPreset::Robidoux);
        let s = luma_quant_table(50, QuantPreset::Robidoux, false);
        for i in 0..64 {
            assert_eq!(s[i], b[i].min(255));
        }
    }
    #[test]
    fn q100_all_ones() {
        let t = luma_quant_table(100, QuantPreset::AnnexK, false);
        for &v in &t {
            assert_eq!(v, 1);
        }
    }
    #[test]
    fn q25_libjpeg() {
        assert_eq!(luma_quant_table(25, QuantPreset::AnnexK, false)[0], 32);
    }
    #[test]
    fn q75_libjpeg() {
        assert_eq!(luma_quant_table(75, QuantPreset::AnnexK, false)[0], 8);
    }
    #[test]
    fn q95_libjpeg() {
        assert_eq!(luma_quant_table(95, QuantPreset::AnnexK, false)[0], 2);
    }
    #[test]
    fn twelve_bit() {
        assert!(
            luma_quant_table(1, QuantPreset::Robidoux, true)
                .iter()
                .any(|&v| v > 255)
        );
    }

    #[test]
    fn all_nine_presets_valid() {
        for p in [
            QuantPreset::Robidoux,
            QuantPreset::AnnexK,
            QuantPreset::Flat,
            QuantPreset::MssimTuned,
            QuantPreset::PsnrHvsM,
            QuantPreset::Klein,
            QuantPreset::Watson,
            QuantPreset::Ahumada,
            QuantPreset::Peterson,
        ] {
            let l = luma_quant_table(75, p, false);
            let c = chroma_quant_table(75, p, false);
            for i in 0..64 {
                assert!(l[i] >= 1 && l[i] <= 255, "{p:?} l[{i}]");
                assert!(c[i] >= 1 && c[i] <= 255, "{p:?} c[{i}]");
            }
        }
    }

    #[test]
    fn quantize_roundtrip() {
        // Quantize divides by Q*8 (LL&M convention), dequantize multiplies by Q.
        // So quantize→dequantize gives c / 8 (approximately).
        // The error from the original is bounded by Q*8/2 (half-step).
        let qt = luma_quant_table(50, QuantPreset::AnnexK, false);
        let mut c = [0i32; 64];
        for i in 0..64 {
            c[i] = (i as i32 * 70 - 2000).clamp(-8192, 8191); // LL&M scale (8x larger)
        }
        let mut q = [0i16; 64];
        quantize(&c, &qt, &mut q);
        let mut d = [0i32; 64];
        dequantize(&q, &qt, &mut d);
        // After quantize(÷Q*8) → dequantize(×Q), result ≈ c/8.
        // Error: |c/8 - d| <= Q/2 (half quantization step after the 8x is removed)
        for i in 0..64 {
            let expected = c[i] / 8; // approximate
            assert!(
                (expected - d[i]).abs() <= qt[i] as i32,
                "pos {i}: expected≈{expected} got {}, Qt={}", d[i], qt[i]
            );
        }
    }

    #[test]
    fn quantize_sign() {
        // Q*8: 10*8=80. 50/80=0 (rounds to 0), need larger values
        let qt = [10u16; 64];
        let mut c = [0i32; 64];
        c[0] = 400; // 400 / (10*8) = 400/80 = 5
        c[1] = -400;
        let mut q = [0i16; 64];
        quantize(&c, &qt, &mut q);
        assert_eq!(q[0], 5);
        assert_eq!(q[1], -5);
    }
}
