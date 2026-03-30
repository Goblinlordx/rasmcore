//! Composable LUT-based pixel point operations.
//!
//! Every point operation is a pure per-channel mapping expressed as a lookup table:
//! - **8-bit:** `u8 → u8` via 256-entry LUT (256 bytes)
//! - **16-bit:** `u16 → u16` via 65536-entry LUT (128 KB)
//!
//! When multiple point ops are chained, their LUTs can be composed into a single
//! fused LUT at plan time, reducing N operations to one memory pass regardless
//! of chain length. This applies at both bit depths.
//!
//! Public convenience functions auto-dispatch based on pixel format:
//! 8-bit formats use the compact LUT, 16-bit formats use the full LUT.
//!
//! Non-point-op nodes (blur, resize, etc.) act as fusion barriers — only
//! consecutive runs of point ops are fused.

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

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
    Levels {
        black: f32,
        white: f32,
        gamma: f32,
    },
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

/// Apply a LUT to all color channels of a pixel buffer, preserving alpha.
///
/// For RGB8: applies LUT to all 3 channels.
/// For RGBA8: applies LUT to R, G, B; copies A unchanged. Uses batch unrolling
/// (4 pixels = 16 bytes per iteration) for better throughput.
/// For Gray8: applies LUT to the single channel.
pub fn apply_lut(pixels: &[u8], info: &ImageInfo, lut: &[u8; 256]) -> Result<Vec<u8>, ImageError> {
    match info.format {
        PixelFormat::Rgb8 => {
            // Process 4 pixels (12 bytes) per iteration for better pipelining
            let mut result = vec![0u8; pixels.len()];
            let chunks = pixels.len() / 12;
            let remainder = pixels.len() % 12;
            for i in 0..chunks {
                let base = i * 12;
                result[base] = lut[pixels[base] as usize];
                result[base + 1] = lut[pixels[base + 1] as usize];
                result[base + 2] = lut[pixels[base + 2] as usize];
                result[base + 3] = lut[pixels[base + 3] as usize];
                result[base + 4] = lut[pixels[base + 4] as usize];
                result[base + 5] = lut[pixels[base + 5] as usize];
                result[base + 6] = lut[pixels[base + 6] as usize];
                result[base + 7] = lut[pixels[base + 7] as usize];
                result[base + 8] = lut[pixels[base + 8] as usize];
                result[base + 9] = lut[pixels[base + 9] as usize];
                result[base + 10] = lut[pixels[base + 10] as usize];
                result[base + 11] = lut[pixels[base + 11] as usize];
            }
            let tail = chunks * 12;
            for i in 0..remainder {
                result[tail + i] = lut[pixels[tail + i] as usize];
            }
            Ok(result)
        }
        PixelFormat::Gray8 => {
            let mut result = vec![0u8; pixels.len()];
            for (out, &inp) in result.iter_mut().zip(pixels.iter()) {
                *out = lut[inp as usize];
            }
            Ok(result)
        }
        PixelFormat::Rgba8 => {
            // Batch 4 pixels (16 bytes) per iteration: 4x(R,G,B lookup + A copy)
            let mut result = vec![0u8; pixels.len()];
            let pixel_count = pixels.len() / 4;
            let batches = pixel_count / 4;
            let remainder = pixel_count % 4;

            for i in 0..batches {
                let base = i * 16;
                // Pixel 0
                result[base] = lut[pixels[base] as usize];
                result[base + 1] = lut[pixels[base + 1] as usize];
                result[base + 2] = lut[pixels[base + 2] as usize];
                result[base + 3] = pixels[base + 3]; // alpha
                                                     // Pixel 1
                result[base + 4] = lut[pixels[base + 4] as usize];
                result[base + 5] = lut[pixels[base + 5] as usize];
                result[base + 6] = lut[pixels[base + 6] as usize];
                result[base + 7] = pixels[base + 7];
                // Pixel 2
                result[base + 8] = lut[pixels[base + 8] as usize];
                result[base + 9] = lut[pixels[base + 9] as usize];
                result[base + 10] = lut[pixels[base + 10] as usize];
                result[base + 11] = pixels[base + 11];
                // Pixel 3
                result[base + 12] = lut[pixels[base + 12] as usize];
                result[base + 13] = lut[pixels[base + 13] as usize];
                result[base + 14] = lut[pixels[base + 14] as usize];
                result[base + 15] = pixels[base + 15];
            }
            let tail = batches * 16;
            for i in 0..remainder {
                let base = tail + i * 4;
                result[base] = lut[pixels[base] as usize];
                result[base + 1] = lut[pixels[base + 1] as usize];
                result[base + 2] = lut[pixels[base + 2] as usize];
                result[base + 3] = pixels[base + 3];
            }
            Ok(result)
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "point op on {other:?} not supported"
        ))),
    }
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

/// Apply a 16-bit LUT to all color channels, preserving alpha.
///
/// Pixels are stored as LE byte pairs in `Vec<u8>`.
pub fn apply_lut_u16(pixels: &[u8], info: &ImageInfo, lut: &[u16]) -> Result<Vec<u8>, ImageError> {
    assert!(lut.len() == 65536);

    match info.format {
        PixelFormat::Rgb16 => {
            let mut result = vec![0u8; pixels.len()];
            for (chunk_in, chunk_out) in pixels.chunks_exact(6).zip(result.chunks_exact_mut(6)) {
                for c in 0..3 {
                    let v = u16::from_le_bytes([chunk_in[c * 2], chunk_in[c * 2 + 1]]);
                    let mapped = lut[v as usize];
                    let bytes = mapped.to_le_bytes();
                    chunk_out[c * 2] = bytes[0];
                    chunk_out[c * 2 + 1] = bytes[1];
                }
            }
            Ok(result)
        }
        PixelFormat::Rgba16 => {
            let mut result = vec![0u8; pixels.len()];
            for (chunk_in, chunk_out) in pixels.chunks_exact(8).zip(result.chunks_exact_mut(8)) {
                // Map R, G, B channels
                for c in 0..3 {
                    let v = u16::from_le_bytes([chunk_in[c * 2], chunk_in[c * 2 + 1]]);
                    let mapped = lut[v as usize];
                    let bytes = mapped.to_le_bytes();
                    chunk_out[c * 2] = bytes[0];
                    chunk_out[c * 2 + 1] = bytes[1];
                }
                // Copy alpha unchanged
                chunk_out[6] = chunk_in[6];
                chunk_out[7] = chunk_in[7];
            }
            Ok(result)
        }
        PixelFormat::Gray16 => {
            let mut result = vec![0u8; pixels.len()];
            for (pair_in, pair_out) in pixels.chunks_exact(2).zip(result.chunks_exact_mut(2)) {
                let v = u16::from_le_bytes([pair_in[0], pair_in[1]]);
                let mapped = lut[v as usize];
                let bytes = mapped.to_le_bytes();
                pair_out[0] = bytes[0];
                pair_out[1] = bytes[1];
            }
            Ok(result)
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "16-bit point op on {other:?} not supported"
        ))),
    }
}

/// Check if a pixel format is 16-bit.
fn is_16bit(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
    )
}

// ─── Public convenience functions ───────────────────────────────────────────

/// Apply a point operation, auto-dispatching between 8-bit and 16-bit paths.
fn apply_op(pixels: &[u8], info: &ImageInfo, op: &PointOp) -> Result<Vec<u8>, ImageError> {
    if is_16bit(info.format) {
        let lut = build_lut_u16(op);
        apply_lut_u16(pixels, info, &lut)
    } else {
        let lut = build_lut(op);
        apply_lut(pixels, info, &lut)
    }
}

/// Apply gamma correction.
pub fn gamma(pixels: &[u8], info: &ImageInfo, gamma_value: f32) -> Result<Vec<u8>, ImageError> {
    if gamma_value <= 0.0 {
        return Err(ImageError::InvalidParameters("gamma must be > 0".into()));
    }
    apply_op(pixels, info, &PointOp::Gamma(gamma_value))
}

/// Invert (negate) all channels.
pub fn invert(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    apply_op(pixels, info, &PointOp::Invert)
}

/// Binary threshold at the given level.
pub fn threshold(pixels: &[u8], info: &ImageInfo, level: u8) -> Result<Vec<u8>, ImageError> {
    apply_op(pixels, info, &PointOp::Threshold(level))
}

/// Solarize: invert pixels above threshold, creating a partial-negative effect.
pub fn solarize(pixels: &[u8], info: &ImageInfo, threshold: u8) -> Result<Vec<u8>, ImageError> {
    apply_op(pixels, info, &PointOp::Solarize(threshold))
}

/// Posterize to N discrete levels per channel.
pub fn posterize(pixels: &[u8], info: &ImageInfo, levels: u8) -> Result<Vec<u8>, ImageError> {
    if levels < 2 {
        return Err(ImageError::InvalidParameters(
            "posterize levels must be >= 2".into(),
        ));
    }
    apply_op(pixels, info, &PointOp::Posterize(levels))
}

/// Clamp pixel values to [min, max].
pub fn clamp(pixels: &[u8], info: &ImageInfo, min: u8, max: u8) -> Result<Vec<u8>, ImageError> {
    apply_op(pixels, info, &PointOp::Clamp(min, max))
}

/// Levels adjustment: remap [black, white] input range with gamma correction.
/// black/white are fractions in [0.0, 1.0], gamma is the exponent (1.0 = linear).
pub fn levels(
    pixels: &[u8],
    info: &ImageInfo,
    black: f32,
    white: f32,
    gamma: f32,
) -> Result<Vec<u8>, ImageError> {
    apply_op(
        pixels,
        info,
        &PointOp::Levels {
            black,
            white,
            gamma,
        },
    )
}

/// Sigmoidal contrast: S-curve contrast adjustment.
/// strength controls curve steepness (0 = identity, higher = more contrast).
/// midpoint is the center of the curve in [0.0, 1.0].
/// sharpen=true increases contrast, sharpen=false decreases it.
pub fn sigmoidal_contrast(
    pixels: &[u8],
    info: &ImageInfo,
    strength: f32,
    midpoint: f32,
    sharpen: bool,
) -> Result<Vec<u8>, ImageError> {
    apply_op(
        pixels,
        info,
        &PointOp::SigmoidalContrast {
            strength,
            midpoint,
            sharpen,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn test_info(w: u32, h: u32, fmt: PixelFormat) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: fmt,
            color_space: ColorSpace::Srgb,
        }
    }

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
        // gamma > 1 brightens: out = in^(1/gamma), 1/gamma < 1 → concave up
        let lut = build_lut(&PointOp::Gamma(2.2));
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        assert!(lut[128] > 128, "gamma 2.2 should brighten midtones");

        // gamma < 1 darkens: out = in^(1/gamma), 1/gamma > 1 → concave down
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
        // 2 levels: 0 and 255
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
        assert_eq!(lut[0], 128); // 0 + round(0.5*255) = 128
        assert_eq!(lut[255], 255); // clamped
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

    #[test]
    fn compose_matches_sequential_application() {
        // gamma -> invert -> posterize: fused should equal sequential
        let g = build_lut(&PointOp::Gamma(2.2));
        let i = build_lut(&PointOp::Invert);
        let p = build_lut(&PointOp::Posterize(4));

        let fused = compose_luts(&compose_luts(&g, &i), &p);

        for v in 0..=255u8 {
            let sequential = p[i[g[v as usize] as usize] as usize];
            assert_eq!(fused[v as usize], sequential, "mismatch at {v}");
        }
    }

    // ── apply_lut ───────────────────────────────────────────────────────

    #[test]
    fn apply_lut_rgb8() {
        let pixels = vec![100u8, 150, 200, 50, 100, 150];
        let info = test_info(2, 1, PixelFormat::Rgb8);
        let inv = build_lut(&PointOp::Invert);
        let result = apply_lut(&pixels, &info, &inv).unwrap();
        assert_eq!(result, vec![155, 105, 55, 205, 155, 105]);
    }

    #[test]
    fn apply_lut_rgba8_preserves_alpha() {
        let pixels = vec![100, 150, 200, 128, 50, 100, 150, 255];
        let info = test_info(2, 1, PixelFormat::Rgba8);
        let inv = build_lut(&PointOp::Invert);
        let result = apply_lut(&pixels, &info, &inv).unwrap();
        assert_eq!(result, vec![155, 105, 55, 128, 205, 155, 105, 255]);
    }

    #[test]
    fn apply_lut_gray8() {
        let pixels = vec![0, 128, 255];
        let info = test_info(3, 1, PixelFormat::Gray8);
        let inv = build_lut(&PointOp::Invert);
        let result = apply_lut(&pixels, &info, &inv).unwrap();
        assert_eq!(result, vec![255, 127, 0]);
    }

    // ── public API ──────────────────────────────────────────────────────

    #[test]
    fn gamma_invalid_returns_error() {
        let info = test_info(1, 1, PixelFormat::Rgb8);
        assert!(gamma(&[128, 128, 128], &info, 0.0).is_err());
        assert!(gamma(&[128, 128, 128], &info, -1.0).is_err());
    }

    #[test]
    fn posterize_invalid_returns_error() {
        let info = test_info(1, 1, PixelFormat::Rgb8);
        assert!(posterize(&[128, 128, 128], &info, 1).is_err());
        assert!(posterize(&[128, 128, 128], &info, 0).is_err());
    }

    #[test]
    fn invert_roundtrip() {
        let pixels = vec![0, 64, 128, 192, 255, 42];
        let info = test_info(2, 1, PixelFormat::Rgb8);
        let inv = invert(&pixels, &info).unwrap();
        let back = invert(&inv, &info).unwrap();
        assert_eq!(back, pixels);
    }

    #[test]
    fn threshold_produces_binary() {
        let pixels: Vec<u8> = (0..=255).collect();
        let info = test_info(256, 1, PixelFormat::Gray8);
        let result = threshold(&pixels, &info, 100).unwrap();
        for &v in &result {
            assert!(v == 0 || v == 255);
        }
    }

    #[test]
    fn clamp_constrains_values() {
        let pixels: Vec<u8> = (0..=255).collect();
        let info = test_info(256, 1, PixelFormat::Gray8);
        let result = clamp(&pixels, &info, 30, 220).unwrap();
        for &v in &result {
            assert!(v >= 30 && v <= 220);
        }
    }

    // ── 16-bit LUT tests ───────────────────────────────────────────────

    fn make_gray16_pixels(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_gray16_pixels(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn lut_u16_gamma_identity() {
        let lut = build_lut_u16(&PointOp::Gamma(1.0));
        for i in 0..=65535u16 {
            assert_eq!(lut[i as usize], i, "gamma 1.0 identity failed at {i}");
        }
    }

    #[test]
    fn lut_u16_invert_roundtrip() {
        let lut = build_lut_u16(&PointOp::Invert);
        for i in 0..=65535u16 {
            assert_eq!(lut[i as usize], 65535 - i);
        }
        let fused = compose_luts_u16(&lut, &lut);
        for i in 0..=65535u16 {
            assert_eq!(
                fused[i as usize], i,
                "double invert should be identity at {i}"
            );
        }
    }

    #[test]
    fn lut_u16_brightness() {
        let lut = build_lut_u16(&PointOp::Brightness(0.5));
        assert_eq!(lut[0], 32768); // 0 + round(0.5*65535) = 32768
        assert_eq!(lut[65535], 65535); // clamped
    }

    #[test]
    fn apply_lut_u16_gray16() {
        let pixels = make_gray16_pixels(&[0, 32768, 65535]);
        let info = test_info(3, 1, PixelFormat::Gray16);
        let lut = build_lut_u16(&PointOp::Invert);
        let result = apply_lut_u16(&pixels, &info, &lut).unwrap();
        let values = read_gray16_pixels(&result);
        assert_eq!(values, vec![65535, 32767, 0]);
    }

    #[test]
    fn apply_lut_u16_rgb16() {
        // 1 pixel: R=0, G=32768, B=65535
        let mut pixels = Vec::new();
        for &v in &[0u16, 32768, 65535] {
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = test_info(1, 1, PixelFormat::Rgb16);
        let lut = build_lut_u16(&PointOp::Invert);
        let result = apply_lut_u16(&pixels, &info, &lut).unwrap();
        let values = read_gray16_pixels(&result); // reuse for reading u16 pairs
        assert_eq!(values, vec![65535, 32767, 0]);
    }

    #[test]
    fn apply_lut_u16_rgba16_preserves_alpha() {
        // 1 pixel: R=100, G=200, B=300, A=50000
        let mut pixels = Vec::new();
        for &v in &[100u16, 200, 300, 50000] {
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = test_info(1, 1, PixelFormat::Rgba16);
        let lut = build_lut_u16(&PointOp::Invert);
        let result = apply_lut_u16(&pixels, &info, &lut).unwrap();
        let values = read_gray16_pixels(&result);
        assert_eq!(values[0], 65435); // 65535 - 100
        assert_eq!(values[1], 65335); // 65535 - 200
        assert_eq!(values[2], 65235); // 65535 - 300
        assert_eq!(values[3], 50000); // alpha preserved
    }

    #[test]
    fn auto_dispatch_gamma_16bit() {
        let pixels = make_gray16_pixels(&[0, 32768, 65535]);
        let info = test_info(3, 1, PixelFormat::Gray16);
        let result = gamma(&pixels, &info, 1.0).unwrap();
        let values = read_gray16_pixels(&result);
        assert_eq!(
            values,
            vec![0, 32768, 65535],
            "gamma 1.0 at 16-bit should be identity"
        );
    }

    #[test]
    fn auto_dispatch_invert_16bit_roundtrip() {
        let pixels = make_gray16_pixels(&[0, 1000, 32768, 60000, 65535]);
        let info = test_info(5, 1, PixelFormat::Gray16);
        let inv = invert(&pixels, &info).unwrap();
        let back = invert(&inv, &info).unwrap();
        assert_eq!(back, pixels, "16-bit invert roundtrip should be identity");
    }

    #[test]
    fn compose_luts_u16_matches_sequential() {
        let g = build_lut_u16(&PointOp::Gamma(2.2));
        let inv = build_lut_u16(&PointOp::Invert);
        let fused = compose_luts_u16(&g, &inv);
        // Spot-check several values
        for &v in &[0u16, 100, 1000, 10000, 32768, 50000, 65535] {
            let sequential = inv[g[v as usize] as usize];
            assert_eq!(fused[v as usize], sequential, "compose mismatch at {v}");
        }
    }

    // ── Levels tests ──────────────────────────────────────────────────────

    #[test]
    fn levels_identity() {
        // (0%, 100%, gamma=1.0) should be identity
        let lut = build_lut(&PointOp::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        });
        for i in 0..256 {
            assert_eq!(lut[i], i as u8, "levels identity failed at {i}");
        }
    }

    #[test]
    fn levels_black_white_clamping() {
        // black=10%, white=90% — values below 25 should map to 0, above 230 to 255
        let lut = build_lut(&PointOp::Levels {
            black: 0.1,
            white: 0.9,
            gamma: 1.0,
        });
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        // Midpoint (128/255 ≈ 0.502) should map to approximately (0.502-0.1)/(0.9-0.1) ≈ 0.503 → 128
        assert!((lut[128] as i16 - 128).abs() <= 1);
    }

    #[test]
    fn levels_with_gamma() {
        // gamma=2.0 should darken midtones
        let lut = build_lut(&PointOp::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 2.0,
        });
        // With gamma=2.0, inv_gamma=0.5, so midpoint 128 → (0.502)^0.5 ≈ 0.709 → 181
        assert!(lut[128] > 160, "gamma 2.0 should brighten: got {}", lut[128]);
    }

    #[test]
    fn levels_compose_with_invert() {
        let levels_lut = build_lut(&PointOp::Levels {
            black: 0.1,
            white: 0.9,
            gamma: 1.0,
        });
        let invert_lut = build_lut(&PointOp::Invert);
        let fused = compose_luts(&levels_lut, &invert_lut);
        // Fused should be levels then invert
        for i in 0..256 {
            assert_eq!(fused[i], invert_lut[levels_lut[i] as usize]);
        }
    }

    // ── Sigmoidal Contrast tests ──────────────────────────────────────────

    #[test]
    fn sigmoidal_identity_at_zero_strength() {
        let lut = build_lut(&PointOp::SigmoidalContrast {
            strength: 0.0,
            midpoint: 0.5,
            sharpen: true,
        });
        for i in 0..256 {
            assert_eq!(lut[i], i as u8, "sigmoidal identity failed at {i}");
        }
    }

    #[test]
    fn sigmoidal_sharpen_increases_contrast() {
        let lut = build_lut(&PointOp::SigmoidalContrast {
            strength: 5.0,
            midpoint: 0.5,
            sharpen: true,
        });
        // Endpoints should be near 0 and 255
        assert!(lut[0] <= 2, "sigmoidal black point: {}", lut[0]);
        assert!(lut[255] >= 253, "sigmoidal white point: {}", lut[255]);
        // Midpoint (128) should stay near 128
        assert!((lut[128] as i16 - 128).abs() <= 2, "midpoint: {}", lut[128]);
        // Shadows should be darker, highlights brighter (S-curve)
        assert!(lut[64] < 64, "shadows should be darker: {} vs 64", lut[64]);
        assert!(lut[192] > 192, "highlights should be brighter: {} vs 192", lut[192]);
    }

    #[test]
    fn sigmoidal_soften_decreases_contrast() {
        let lut = build_lut(&PointOp::SigmoidalContrast {
            strength: 5.0,
            midpoint: 0.5,
            sharpen: false,
        });
        // Inverse: shadows brighter, highlights darker
        assert!(lut[64] > 64, "softened shadows should be brighter: {} vs 64", lut[64]);
        assert!(lut[192] < 192, "softened highlights should be darker: {} vs 192", lut[192]);
    }

    #[test]
    fn sigmoidal_compose_with_levels() {
        let sig_lut = build_lut(&PointOp::SigmoidalContrast {
            strength: 3.0,
            midpoint: 0.5,
            sharpen: true,
        });
        let levels_lut = build_lut(&PointOp::Levels {
            black: 0.1,
            white: 0.9,
            gamma: 1.0,
        });
        let fused = compose_luts(&levels_lut, &sig_lut);
        for i in 0..256 {
            assert_eq!(fused[i], sig_lut[levels_lut[i] as usize]);
        }
    }
}
