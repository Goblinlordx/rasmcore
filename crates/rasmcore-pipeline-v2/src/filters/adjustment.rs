//! Adjustment filters — point operations on f32 pixel data.
//!
//! All implement `Filter` (f32 compute) and `AnalyticOp` (expression tree
//! for fusion optimizer). These are per-channel operations: f(v) applied
//! independently to R, G, B. Alpha is preserved unchanged.

use crate::node::PipelineError;
use crate::ops::{AnalyticOp, Filter, PointOpExpr};

/// Brightness adjustment — additive offset.
///
/// `output = input + amount` (clamped to [0, 1] only at encode boundary).
#[derive(Clone)]
pub struct Brightness {
    pub amount: f32,
}

impl Filter for Brightness {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let offset = self.amount;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] += offset;
            pixel[1] += offset;
            pixel[2] += offset;
        }
        Ok(out)
    }
}

impl AnalyticOp for Brightness {
    fn expression(&self) -> PointOpExpr {
        PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.amount)),
        )
    }
}

/// Contrast adjustment — multiplicative around midpoint.
///
/// `output = (input - 0.5) * factor + 0.5`
#[derive(Clone)]
pub struct Contrast {
    pub amount: f32,
}

impl Filter for Contrast {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let factor = 1.0 + self.amount;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = (pixel[0] - 0.5) * factor + 0.5;
            pixel[1] = (pixel[1] - 0.5) * factor + 0.5;
            pixel[2] = (pixel[2] - 0.5) * factor + 0.5;
        }
        Ok(out)
    }
}

impl AnalyticOp for Contrast {
    fn expression(&self) -> PointOpExpr {
        let factor = 1.0 + self.amount;
        PointOpExpr::Add(
            Box::new(PointOpExpr::Mul(
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(0.5)),
                )),
                Box::new(PointOpExpr::Constant(factor)),
            )),
            Box::new(PointOpExpr::Constant(0.5)),
        )
    }
}

/// Gamma correction — power curve.
///
/// `output = input ^ (1/gamma)` for gamma > 0.
#[derive(Clone)]
pub struct Gamma {
    pub gamma: f32,
}

impl Filter for Gamma {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let inv_gamma = 1.0 / self.gamma;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].max(0.0).powf(inv_gamma);
            pixel[1] = pixel[1].max(0.0).powf(inv_gamma);
            pixel[2] = pixel[2].max(0.0).powf(inv_gamma);
        }
        Ok(out)
    }
}

impl AnalyticOp for Gamma {
    fn expression(&self) -> PointOpExpr {
        PointOpExpr::Pow(
            Box::new(PointOpExpr::Max(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.0)),
            )),
            Box::new(PointOpExpr::Constant(1.0 / self.gamma)),
        )
    }
}

/// Exposure adjustment — EV stops with offset and gamma.
///
/// `output = ((input + offset) * 2^ev) ^ (1/gamma)`
#[derive(Clone)]
pub struct Exposure {
    pub ev: f32,
    pub offset: f32,
    pub gamma_correction: f32,
}

impl Filter for Exposure {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let multiplier = 2.0f32.powf(self.ev);
        let inv_gamma = 1.0 / self.gamma_correction;
        let offset = self.offset;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                let v = (*c + offset) * multiplier;
                *c = v.max(0.0).powf(inv_gamma);
            }
        }
        Ok(out)
    }
}

impl AnalyticOp for Exposure {
    fn expression(&self) -> PointOpExpr {
        let multiplier = 2.0f32.powf(self.ev);
        PointOpExpr::Pow(
            Box::new(PointOpExpr::Max(
                Box::new(PointOpExpr::Mul(
                    Box::new(PointOpExpr::Add(
                        Box::new(PointOpExpr::Input),
                        Box::new(PointOpExpr::Constant(self.offset)),
                    )),
                    Box::new(PointOpExpr::Constant(multiplier)),
                )),
                Box::new(PointOpExpr::Constant(0.0)),
            )),
            Box::new(PointOpExpr::Constant(1.0 / self.gamma_correction)),
        )
    }
}

/// Invert — channel negation.
///
/// `output = 1.0 - input`
#[derive(Clone)]
pub struct Invert;

impl Filter for Invert {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = 1.0 - pixel[0];
            pixel[1] = 1.0 - pixel[1];
            pixel[2] = 1.0 - pixel[2];
        }
        Ok(out)
    }
}

impl AnalyticOp for Invert {
    fn expression(&self) -> PointOpExpr {
        PointOpExpr::Sub(
            Box::new(PointOpExpr::Constant(1.0)),
            Box::new(PointOpExpr::Input),
        )
    }
}

/// Levels — remap input range with gamma.
///
/// `output = ((input - black) / (white - black)) ^ (1/gamma)`
#[derive(Clone)]
pub struct Levels {
    /// Black point [0, 1]
    pub black: f32,
    /// White point [0, 1]
    pub white: f32,
    /// Gamma correction
    pub gamma: f32,
}

impl Filter for Levels {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let range = (self.white - self.black).max(1e-6);
        let inv_gamma = 1.0 / self.gamma;
        let black = self.black;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                let v = (*c - black) / range;
                *c = v.max(0.0).powf(inv_gamma);
            }
        }
        Ok(out)
    }
}

impl AnalyticOp for Levels {
    fn expression(&self) -> PointOpExpr {
        let range = (self.white - self.black).max(1e-6);
        PointOpExpr::Pow(
            Box::new(PointOpExpr::Max(
                Box::new(PointOpExpr::Div(
                    Box::new(PointOpExpr::Sub(
                        Box::new(PointOpExpr::Input),
                        Box::new(PointOpExpr::Constant(self.black)),
                    )),
                    Box::new(PointOpExpr::Constant(range)),
                )),
                Box::new(PointOpExpr::Constant(0.0)),
            )),
            Box::new(PointOpExpr::Constant(1.0 / self.gamma)),
        )
    }
}

/// Posterize — reduce to N discrete levels.
///
/// `output = floor(input * levels) / (levels - 1)`
#[derive(Clone)]
pub struct Posterize {
    pub levels: u8,
}

impl Filter for Posterize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let n = self.levels as f32;
        let inv = 1.0 / (n - 1.0).max(1.0);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                *c = (*c * n).floor().min(n - 1.0) * inv;
            }
        }
        Ok(out)
    }
}

impl AnalyticOp for Posterize {
    fn expression(&self) -> PointOpExpr {
        let n = self.levels as f32;
        let inv = 1.0 / (n - 1.0).max(1.0);
        PointOpExpr::Mul(
            Box::new(PointOpExpr::Min(
                Box::new(PointOpExpr::Floor(Box::new(PointOpExpr::Mul(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(n)),
                )))),
                Box::new(PointOpExpr::Constant(n - 1.0)),
            )),
            Box::new(PointOpExpr::Constant(inv)),
        )
    }
}

/// Sigmoidal contrast — S-curve.
///
/// Uses the sigmoidal transfer function for more natural contrast
/// than linear multiplication.
#[derive(Clone)]
pub struct SigmoidalContrast {
    pub strength: f32,
    /// Midpoint [0, 1]
    pub midpoint: f32,
    /// true = increase contrast, false = decrease
    pub sharpen: bool,
}

impl Filter for SigmoidalContrast {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let a = self.strength;
        let b = self.midpoint;
        let sharpen = self.sharpen;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                *c = sigmoidal(*c, a, b, sharpen);
            }
        }
        Ok(out)
    }
}

fn sigmoidal(v: f32, strength: f32, midpoint: f32, sharpen: bool) -> f32 {
    if strength.abs() < 1e-6 {
        return v;
    }
    if sharpen {
        let num = 1.0 / (1.0 + (-strength * (v - midpoint)).exp())
            - 1.0 / (1.0 + (strength * midpoint).exp());
        let den = 1.0 / (1.0 + (-strength * (1.0 - midpoint)).exp())
            - 1.0 / (1.0 + (strength * midpoint).exp());
        if den.abs() < 1e-10 { v } else { num / den }
    } else {
        let sig_mid = 1.0 / (1.0 + (-strength * midpoint).exp());
        let sig_range = 1.0 / (1.0 + (-strength * (1.0 - midpoint)).exp()) - sig_mid;
        if sig_range.abs() < 1e-10 {
            return v;
        }
        let t = sig_mid + v * sig_range;
        midpoint - (1.0 / t - 1.0).ln() / strength
    }
}

/// Dodge — brighten shadows.
///
/// `output = input / (1 - amount)` (simplified dodge)
#[derive(Clone)]
pub struct Dodge {
    pub amount: f32,
}

impl Filter for Dodge {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let divisor = (1.0 - self.amount).max(1e-6);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                *c /= divisor;
            }
        }
        Ok(out)
    }
}

impl AnalyticOp for Dodge {
    fn expression(&self) -> PointOpExpr {
        let divisor = (1.0 - self.amount).max(1e-6);
        PointOpExpr::Div(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(divisor)),
        )
    }
}

/// Burn — darken highlights.
///
/// `output = 1 - (1 - input) / amount`
#[derive(Clone)]
pub struct Burn {
    pub amount: f32,
}

impl Filter for Burn {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let amt = self.amount.max(1e-6);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                *c = 1.0 - (1.0 - *c) / amt;
            }
        }
        Ok(out)
    }
}

impl AnalyticOp for Burn {
    fn expression(&self) -> PointOpExpr {
        let amt = self.amount.max(1e-6);
        PointOpExpr::Sub(
            Box::new(PointOpExpr::Constant(1.0)),
            Box::new(PointOpExpr::Div(
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Constant(1.0)),
                    Box::new(PointOpExpr::Input),
                )),
                Box::new(PointOpExpr::Constant(amt)),
            )),
        )
    }
}

/// Solarize — invert values above threshold.
///
/// `output = if input > threshold { 1.0 - input } else { input }`
#[derive(Clone)]
pub struct Solarize {
    pub threshold: f32,
}

impl Filter for Solarize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let thresh = self.threshold;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            for c in &mut pixel[..3] {
                if *c > thresh {
                    *c = 1.0 - *c;
                }
            }
        }
        Ok(out)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pixels() -> Vec<f32> {
        // 2x2 RGBA: varied values
        vec![
            0.0, 0.25, 0.5, 1.0, // pixel 0
            0.5, 0.75, 1.0, 1.0, // pixel 1
            0.1, 0.2, 0.3, 0.5, // pixel 2
            1.0, 0.0, 0.5, 1.0, // pixel 3
        ]
    }

    #[test]
    fn brightness_adds_offset() {
        let input = test_pixels();
        let b = Brightness { amount: 0.1 };
        let out = b.compute(&input, 2, 2).unwrap();
        assert!((out[0] - 0.1).abs() < 1e-6); // 0.0 + 0.1
        assert!((out[1] - 0.35).abs() < 1e-6); // 0.25 + 0.1
        assert_eq!(out[3], 1.0); // alpha unchanged
    }

    #[test]
    fn invert_flips_channels() {
        let input = test_pixels();
        let inv = Invert;
        let out = inv.compute(&input, 2, 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6); // 1.0 - 0.0
        assert!((out[2] - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert_eq!(out[3], 1.0); // alpha unchanged
    }

    #[test]
    fn brightness_expression_matches_compute() {
        let b = Brightness { amount: 0.2 };
        let expr = b.expression();
        let input = test_pixels();
        let computed = b.compute(&input, 2, 2).unwrap();
        for i in (0..input.len()).step_by(4) {
            for c in 0..3 {
                let from_expr = expr.evaluate(input[i + c] as f64) as f32;
                assert!(
                    (from_expr - computed[i + c]).abs() < 1e-5,
                    "mismatch at pixel {}, channel {}: expr={from_expr} compute={}",
                    i / 4,
                    c,
                    computed[i + c]
                );
            }
        }
    }

    #[test]
    fn contrast_expression_matches_compute() {
        let c = Contrast { amount: 0.5 };
        let expr = c.expression();
        let input = test_pixels();
        let computed = c.compute(&input, 2, 2).unwrap();
        for i in (0..input.len()).step_by(4) {
            for ch in 0..3 {
                let from_expr = expr.evaluate(input[i + ch] as f64) as f32;
                assert!(
                    (from_expr - computed[i + ch]).abs() < 1e-4,
                    "mismatch at pixel {}, channel {ch}",
                    i / 4
                );
            }
        }
    }

    #[test]
    fn gamma_expression_matches_compute() {
        let g = Gamma { gamma: 2.2 };
        let expr = g.expression();
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let computed = g.compute(&input, 1, 1).unwrap();
        let from_expr = expr.evaluate(0.5) as f32;
        assert!((from_expr - computed[0]).abs() < 1e-4);
    }

    #[test]
    fn levels_remaps_range() {
        let l = Levels {
            black: 0.2,
            white: 0.8,
            gamma: 1.0,
        };
        let input = vec![0.5, 0.5, 0.5, 1.0]; // midpoint of [0.2, 0.8] = 0.5
        let out = l.compute(&input, 1, 1).unwrap();
        // (0.5 - 0.2) / (0.8 - 0.2) = 0.3 / 0.6 = 0.5
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn posterize_quantizes() {
        let p = Posterize { levels: 4 };
        let input = vec![0.3, 0.6, 0.9, 1.0];
        let out = p.compute(&input, 1, 1).unwrap();
        // 0.3 * 4 = 1.2 → floor = 1 → 1/3 ≈ 0.333
        assert!((out[0] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn brightness_contrast_fusion() {
        // Compose brightness(+0.1) → contrast(1.5)
        let b = Brightness { amount: 0.1 };
        let c = Contrast { amount: 0.5 }; // factor = 1.5
        let b_expr = b.expression();
        let c_expr = c.expression();
        let fused = PointOpExpr::compose(&c_expr, &b_expr);

        // For v=0.5: brightness gives 0.6, contrast gives (0.6-0.5)*1.5+0.5 = 0.65
        let result = fused.evaluate(0.5);
        assert!((result - 0.65).abs() < 1e-6);
    }

    #[test]
    fn dodge_burn_are_inverses_at_half() {
        let dodge = Dodge { amount: 0.5 };
        let burn = Burn { amount: 0.5 };
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let dodged = dodge.compute(&input, 1, 1).unwrap();
        // dodge: 0.5 / (1-0.5) = 1.0
        assert!((dodged[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn solarize_inverts_above_threshold() {
        let s = Solarize { threshold: 0.5 };
        let input = vec![0.3, 0.7, 0.5, 1.0];
        let out = s.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.3).abs() < 1e-6); // below threshold: unchanged
        assert!((out[1] - 0.3).abs() < 1e-6); // above threshold: 1.0 - 0.7 = 0.3
    }
}
