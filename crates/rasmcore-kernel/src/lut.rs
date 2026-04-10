//! LUT composition and point operation infrastructure — no image type dependencies.
//!
//! Every point operation is a pure per-channel mapping expressed as a lookup table:
//! - **8-bit:** `u8 → u8` via 256-entry LUT (256 bytes)
//! - **16-bit:** `u16 → u16` via 65536-entry LUT (128 KB)
//!
//! When multiple point ops are chained, their LUTs can be composed into a single
//! fused LUT at plan time, reducing N operations to one memory pass regardless
//! of chain length. This applies at both bit depths.

/// A pixel point operation — a pure `u8 → u8` per-channel mapping.
#[derive(Debug, Clone)]
pub enum PointOp {
    /// Power-law gamma correction. `LUT[i] = round(255 * (i/255)^(1/gamma))`
    Gamma(f32),
    /// Negate all channels. `LUT[i] = 255 - i`
    Invert,
    /// Binary threshold. `LUT[i] = if i >= level { 255 } else { 0 }`
    Threshold(u8),
    /// Reduce to N discrete levels. `LUT[i] = round(round(i*(n-1)/255) * 255/(n-1))`
    Posterize(u8),
    /// Clamp to [min, max] range. `LUT[i] = i.clamp(min, max)`
    Clamp(u8, u8),
    /// Brightness offset (-1.0 to 1.0). `LUT[i] = (i + amount*255).clamp(0, 255)`
    /// Matches ImageMagick `-brightness-contrast Bx0` where B = amount * 100.
    Brightness(f32),
    /// Contrast adjustment (-1.0 to 1.0). `LUT[i] = (factor*(i-128)+128).clamp(0,255)`
    Contrast(f32),
    /// Solarize: invert values at or above threshold. `LUT[i] = if i >= threshold { 255 - i } else { i }`
    /// Uses `>=` to match ImageMagick's Q16-HDRI boundary behavior where 50% threshold
    /// at Q8 maps to value 128, and value 128 IS solarized.
    Solarize(u8),
    /// Levels: remap [black, white] input range to [0, 255] with gamma correction.
    /// `LUT[i] = clamp(((i/255 - black) / (white - black)) ^ (1/gamma) * 255)`
    /// Matches ImageMagick `-level black%,white%,gamma`.
    Levels { black: f32, white: f32, gamma: f32 },
    /// Sigmoidal contrast: S-curve contrast adjustment.
    /// `sigmoid(x) = 1 / (1 + exp(-strength * (x - midpoint)))`
    /// `LUT[i] = (sigmoid(i/255) - sigmoid(0)) / (sigmoid(1) - sigmoid(0)) * 255`
    /// sharpen=true increases contrast, sharpen=false decreases it.
    /// Matches ImageMagick `-sigmoidal-contrast strengthxmidpoint%`.
    SigmoidalContrast {
        strength: f32,
        midpoint: f32,
        sharpen: bool,
    },
}

/// Build a 256-entry LUT for a single point operation.
pub fn build_lut(op: &PointOp) -> [u8; 256] {
    let mut lut = [0u8; 256];
    match op {
        PointOp::Gamma(gamma) => {
            let inv_gamma = 1.0 / gamma;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (255.0 * (i as f32 / 255.0).powf(inv_gamma) + 0.5) as u8;
            }
        }
        PointOp::Invert => {
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (255 - i) as u8;
            }
        }
        PointOp::Threshold(level) => {
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = if (i as u8) >= *level { 255 } else { 0 };
            }
        }
        PointOp::Posterize(levels) => {
            let n = (*levels).max(2) as f32;
            for (i, entry) in lut.iter_mut().enumerate() {
                let quantized = (i as f32 * (n - 1.0) / 255.0 + 0.5) as u8;
                *entry = ((quantized as f32) * 255.0 / (n - 1.0) + 0.5) as u8;
            }
        }
        PointOp::Clamp(min, max) => {
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (i as u8).clamp(*min, *max);
            }
        }
        PointOp::Brightness(amount) => {
            let offset = (*amount * 255.0).round() as i16;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (i as i16 + offset).clamp(0, 255) as u8;
            }
        }
        PointOp::Contrast(amount) => {
            let factor = if *amount >= 0.0 {
                1.0 + amount * 2.0
            } else {
                1.0 / (1.0 - amount * 2.0)
            };
            for (i, entry) in lut.iter_mut().enumerate() {
                let v = factor * (i as f32 - 128.0) + 128.0;
                *entry = v.clamp(0.0, 255.0) as u8;
            }
        }
        PointOp::Solarize(threshold) => {
            for (i, entry) in lut.iter_mut().enumerate() {
                let v = i as u8;
                *entry = if v >= *threshold { 255 - v } else { v };
            }
        }
        PointOp::Levels {
            black,
            white,
            gamma,
        } => {
            let range = (white - black).max(1e-6);
            let inv_gamma = 1.0 / gamma;
            for (i, entry) in lut.iter_mut().enumerate() {
                let normalized = ((i as f32 / 255.0) - black) / range;
                let clamped = normalized.clamp(0.0, 1.0);
                *entry = (clamped.powf(inv_gamma) * 255.0 + 0.5) as u8;
            }
        }
        PointOp::SigmoidalContrast {
            strength,
            midpoint,
            sharpen,
        } => {
            if *strength < 1e-6 {
                // Identity
                for (i, entry) in lut.iter_mut().enumerate() {
                    *entry = i as u8;
                }
            } else {
                // IM formula: sigmoidal contrast uses scaled sigmoid
                // sigmoid(x) = 1 / (1 + exp(strength * (midpoint - x)))
                // Normalize to map [0,1] output range
                let sig = |x: f32| -> f32 { 1.0 / (1.0 + (-*strength * (x - midpoint)).exp()) };
                let sig_0 = sig(0.0);
                let sig_1 = sig(1.0);
                let range = sig_1 - sig_0;

                for (i, entry) in lut.iter_mut().enumerate() {
                    let x = i as f32 / 255.0;
                    let v = if *sharpen {
                        // Increase contrast: apply sigmoid
                        (sig(x) - sig_0) / range
                    } else {
                        // Decrease contrast: apply inverse sigmoid
                        // inv_sig(y) = midpoint - ln((1 - y_scaled) / y_scaled) / strength
                        // where y_scaled = y * range + sig_0
                        let y_scaled = x * range + sig_0;
                        let y_clamped = y_scaled.clamp(1e-7, 1.0 - 1e-7);
                        *midpoint - ((1.0 - y_clamped) / y_clamped).ln() / strength
                    };
                    *entry = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
    }
    lut
}

/// Compose two LUTs: the result applies `first` then `second`.
///
/// `fused[i] = second[first[i]]`
///
/// O(256) — trivial cost at plan time, saves a full pixel pass at runtime.
pub fn compose_luts(first: &[u8; 256], second: &[u8; 256]) -> [u8; 256] {
    let mut fused = [0u8; 256];
    for (i, entry) in fused.iter_mut().enumerate() {
        *entry = second[first[i] as usize];
    }
    fused
}

// ─── 16-bit LUT infrastructure ─────────────────────────────────────────────

const MAX16: f32 = 65535.0;
const HALF16: u16 = 32768;

/// Build a 65536-entry LUT for a single point operation at 16-bit depth.
///
/// Memory: 128 KB per LUT. Acceptable for typical chains of 2-4 fused ops.
pub fn build_lut_u16(op: &PointOp) -> Vec<u16> {
    let mut lut = vec![0u16; 65536];
    match op {
        PointOp::Gamma(gamma) => {
            let inv_gamma = 1.0 / gamma;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (MAX16 * (i as f32 / MAX16).powf(inv_gamma) + 0.5) as u16;
            }
        }
        PointOp::Invert => {
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (65535 - i) as u16;
            }
        }
        PointOp::Threshold(level) => {
            // Scale 8-bit threshold to 16-bit: level * 257
            let level16 = *level as u32 * 257;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = if (i as u32) >= level16 { 65535 } else { 0 };
            }
        }
        PointOp::Posterize(levels) => {
            let n = (*levels).max(2) as f32;
            for (i, entry) in lut.iter_mut().enumerate() {
                let quantized = (i as f32 * (n - 1.0) / MAX16 + 0.5) as u16;
                *entry = ((quantized as f32) * MAX16 / (n - 1.0) + 0.5) as u16;
            }
        }
        PointOp::Clamp(min, max) => {
            // Scale 8-bit clamp bounds to 16-bit
            let min16 = (*min as u16) * 257;
            let max16 = (*max as u16) * 257;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (i as u16).clamp(min16, max16);
            }
        }
        PointOp::Brightness(amount) => {
            let offset = (*amount * MAX16).round() as i32;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = (i as i32 + offset).clamp(0, 65535) as u16;
            }
        }
        PointOp::Contrast(amount) => {
            let factor = if *amount >= 0.0 {
                1.0 + amount * 2.0
            } else {
                1.0 / (1.0 - amount * 2.0)
            };
            for (i, entry) in lut.iter_mut().enumerate() {
                let v = factor * (i as f32 - HALF16 as f32) + HALF16 as f32;
                *entry = v.clamp(0.0, MAX16) as u16;
            }
        }
        PointOp::Solarize(threshold) => {
            // Scale 8-bit threshold to 16-bit: threshold * 257
            let threshold16 = *threshold as u32 * 257;
            for (i, entry) in lut.iter_mut().enumerate() {
                *entry = if (i as u32) >= threshold16 {
                    (65535 - i) as u16
                } else {
                    i as u16
                };
            }
        }
        PointOp::Levels {
            black,
            white,
            gamma,
        } => {
            let range = (white - black).max(1e-6);
            let inv_gamma = 1.0 / gamma;
            for (i, entry) in lut.iter_mut().enumerate() {
                let normalized = ((i as f32 / MAX16) - black) / range;
                let clamped = normalized.clamp(0.0, 1.0);
                *entry = (clamped.powf(inv_gamma) * MAX16 + 0.5) as u16;
            }
        }
        PointOp::SigmoidalContrast {
            strength,
            midpoint,
            sharpen,
        } => {
            if *strength < 1e-6 {
                for (i, entry) in lut.iter_mut().enumerate() {
                    *entry = i as u16;
                }
            } else {
                let sig = |x: f32| -> f32 { 1.0 / (1.0 + (-*strength * (x - midpoint)).exp()) };
                let sig_0 = sig(0.0);
                let sig_1 = sig(1.0);
                let range = sig_1 - sig_0;

                for (i, entry) in lut.iter_mut().enumerate() {
                    let x = i as f32 / MAX16;
                    let v = if *sharpen {
                        (sig(x) - sig_0) / range
                    } else {
                        let y_scaled = x * range + sig_0;
                        let y_clamped = y_scaled.clamp(1e-7, 1.0 - 1e-7);
                        *midpoint - ((1.0 - y_clamped) / y_clamped).ln() / strength
                    };
                    *entry = (v.clamp(0.0, 1.0) * MAX16 + 0.5) as u16;
                }
            }
        }
    }
    lut
}

/// Compose two 16-bit LUTs: `fused[i] = second[first[i]]`.
pub fn compose_luts_u16(first: &[u16], second: &[u16]) -> Vec<u16> {
    assert!(first.len() == 65536 && second.len() == 65536);
    let mut fused = vec![0u16; 65536];
    for (i, entry) in fused.iter_mut().enumerate() {
        *entry = second[first[i] as usize];
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── build_lut correctness ───────────────────────────────────────────

    #[test]
    fn lut_gamma_identity() {
        let lut = build_lut(&PointOp::Gamma(1.0));
        for i in 0..=255u8 {
            assert_eq!(lut[i as usize], i, "gamma 1.0 should be identity at {i}");
        }
    }

    #[test]
    fn lut_gamma_brightens() {
        let lut = build_lut(&PointOp::Gamma(2.2));
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        assert!(lut[128] > 128, "gamma 2.2 should brighten midtones");

        let lut_dark = build_lut(&PointOp::Gamma(0.5));
        assert!(lut_dark[128] < 128, "gamma 0.5 should darken midtones");
    }

    #[test]
    fn lut_invert() {
        let lut = build_lut(&PointOp::Invert);
        for i in 0..=255u8 {
            assert_eq!(lut[i as usize], 255 - i);
        }
    }

    #[test]
    fn lut_threshold() {
        let lut = build_lut(&PointOp::Threshold(128));
        for i in 0..128u8 {
            assert_eq!(lut[i as usize], 0);
        }
        for i in 128..=255u8 {
            assert_eq!(lut[i as usize], 255);
        }
    }

    #[test]
    fn lut_posterize_2_levels() {
        let lut = build_lut(&PointOp::Posterize(2));
        assert_eq!(lut[0], 0);
        assert_eq!(lut[127], 0);
        assert_eq!(lut[128], 255);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn lut_clamp() {
        let lut = build_lut(&PointOp::Clamp(50, 200));
        assert_eq!(lut[0], 50);
        assert_eq!(lut[49], 50);
        assert_eq!(lut[50], 50);
        assert_eq!(lut[100], 100);
        assert_eq!(lut[200], 200);
        assert_eq!(lut[201], 200);
        assert_eq!(lut[255], 200);
    }

    #[test]
    fn lut_brightness_positive() {
        let lut = build_lut(&PointOp::Brightness(0.5));
        assert_eq!(lut[0], 128);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn lut_contrast_zero_is_identity() {
        let lut = build_lut(&PointOp::Contrast(0.0));
        for i in 0..=255u8 {
            assert_eq!(lut[i as usize], i, "contrast 0.0 should be identity at {i}");
        }
    }

    // ── compose_luts ────────────────────────────────────────────────────

    #[test]
    fn compose_identity() {
        let identity: [u8; 256] = std::array::from_fn(|i| i as u8);
        let invert_lut = build_lut(&PointOp::Invert);
        let fused = compose_luts(&identity, &invert_lut);
        assert_eq!(fused, invert_lut);
    }

    #[test]
    fn compose_invert_twice_is_identity() {
        let inv = build_lut(&PointOp::Invert);
        let fused = compose_luts(&inv, &inv);
        for i in 0..=255u8 {
            assert_eq!(
                fused[i as usize], i,
                "double invert should be identity at {i}"
            );
        }
    }

    // ── 16-bit LUT tests ───────────────────────────────────────────────

    #[test]
    fn lut_u16_gamma_identity() {
        let lut = build_lut_u16(&PointOp::Gamma(1.0));
        for i in 0..=65535u16 {
            assert_eq!(
                lut[i as usize], i,
                "16-bit gamma 1.0 should be identity at {i}"
            );
        }
    }

    #[test]
    fn lut_u16_invert() {
        let lut = build_lut_u16(&PointOp::Invert);
        assert_eq!(lut[0], 65535);
        assert_eq!(lut[65535], 0);
        assert_eq!(lut[32768], 32767);
    }

    #[test]
    fn compose_u16_invert_twice_is_identity() {
        let inv = build_lut_u16(&PointOp::Invert);
        let fused = compose_luts_u16(&inv, &inv);
        for i in 0..=65535u16 {
            assert_eq!(
                fused[i as usize], i,
                "16-bit double invert should be identity at {i}"
            );
        }
    }
}
