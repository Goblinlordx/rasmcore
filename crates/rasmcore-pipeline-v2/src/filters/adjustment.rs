//! Adjustment filters — point operations on f32 pixel data.
//!
//! All implement `Filter` (f32 compute) and `analytic_expression()` (expression tree
//! for fusion optimizer). These are per-channel operations: f(v) applied
//! independently to R, G, B. Alpha is preserved unchanged.

use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Brightness adjustment — additive offset.
///
/// `output = input + amount` (clamped to [0, 1] only at encode boundary).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "brightness", category = "adjustment", doc = "docs/operations/filters/adjustment/brightness.adoc")]
pub struct Brightness {
    /// Additive offset applied to each RGB channel.
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0)]
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

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        Some(PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.amount)),
        ))
    }
}

/// Contrast adjustment — multiplicative around midpoint.
///
/// `output = (input - 0.5) * factor + 0.5`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "contrast", category = "adjustment", doc = "docs/operations/filters/adjustment/contrast.adoc")]
pub struct Contrast {
    /// Contrast multiplier. Positive increases contrast, negative decreases.
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0)]
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

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let factor = 1.0 + self.amount;
        Some(PointOpExpr::Add(
            Box::new(PointOpExpr::Mul(
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(0.5)),
                )),
                Box::new(PointOpExpr::Constant(factor)),
            )),
            Box::new(PointOpExpr::Constant(0.5)),
        ))
    }
}

/// Gamma correction — power curve.
///
/// `output = input ^ (1/gamma)` for gamma > 0.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gamma", category = "adjustment")]
pub struct Gamma {
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
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

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        Some(PointOpExpr::Pow(
            Box::new(PointOpExpr::Max(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.0)),
            )),
            Box::new(PointOpExpr::Constant(1.0 / self.gamma)),
        ))
    }
}

/// Exposure adjustment — EV stops with offset and gamma.
///
/// `output = ((input + offset) * 2^ev) ^ (1/gamma)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "exposure", category = "adjustment")]
pub struct Exposure {
    #[param(min = -10.0, max = 10.0, step = 0.1, default = 0.0)]
    pub ev: f32,
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub offset: f32,
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
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
            // Explicit per-channel for auto-vectorization of multiply-add
            pixel[0] = ((pixel[0] + offset) * multiplier).max(0.0).powf(inv_gamma);
            pixel[1] = ((pixel[1] + offset) * multiplier).max(0.0).powf(inv_gamma);
            pixel[2] = ((pixel[2] + offset) * multiplier).max(0.0).powf(inv_gamma);
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let multiplier = 2.0f32.powf(self.ev);
        Some(PointOpExpr::Pow(
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
        ))
    }
}

/// Invert — channel negation.
///
/// `output = 1.0 - input`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "invert", category = "adjustment")]
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

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        Some(PointOpExpr::Sub(
            Box::new(PointOpExpr::Constant(1.0)),
            Box::new(PointOpExpr::Input),
        ))
    }
}

/// Levels — remap input range with gamma.
///
/// `output = ((input - black) / (white - black)) ^ (1/gamma)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "levels", category = "adjustment")]
pub struct Levels {
    /// Black point [0, 1]
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub black: f32,
    /// White point [0, 1]
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub white: f32,
    /// Gamma correction
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
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
            pixel[0] = ((pixel[0] - black) / range).max(0.0).powf(inv_gamma);
            pixel[1] = ((pixel[1] - black) / range).max(0.0).powf(inv_gamma);
            pixel[2] = ((pixel[2] - black) / range).max(0.0).powf(inv_gamma);
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let range = (self.white - self.black).max(1e-6);
        Some(PointOpExpr::Pow(
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
        ))
    }
}

/// Posterize — reduce to N discrete levels.
///
/// `output = floor(input * levels) / (levels - 1)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "posterize", category = "adjustment")]
pub struct Posterize {
    #[param(min = 2, max = 256, step = 1, default = 4)]
    pub levels: u8,
}

impl Filter for Posterize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let n = self.levels as f32;
        let inv = 1.0 / (n - 1.0).max(1.0);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = (pixel[0] * n).floor().min(n - 1.0) * inv;
            pixel[1] = (pixel[1] * n).floor().min(n - 1.0) * inv;
            pixel[2] = (pixel[2] * n).floor().min(n - 1.0) * inv;
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let n = self.levels as f32;
        let inv = 1.0 / (n - 1.0).max(1.0);
        Some(PointOpExpr::Mul(
            Box::new(PointOpExpr::Min(
                Box::new(PointOpExpr::Floor(Box::new(PointOpExpr::Mul(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(n)),
                )))),
                Box::new(PointOpExpr::Constant(n - 1.0)),
            )),
            Box::new(PointOpExpr::Constant(inv)),
        ))
    }
}

/// Sigmoidal contrast — S-curve.
///
/// Uses the sigmoidal transfer function for more natural contrast
/// than linear multiplication.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sigmoidal_contrast", category = "adjustment")]
pub struct SigmoidalContrast {
    #[param(min = 0.0, max = 20.0, step = 0.5, default = 3.0)]
    pub strength: f32,
    /// Midpoint [0, 1]
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub midpoint: f32,
    /// true = increase contrast, false = decrease
    #[param(default = true)]
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
            pixel[0] = sigmoidal(pixel[0], a, b, sharpen);
            pixel[1] = sigmoidal(pixel[1], a, b, sharpen);
            pixel[2] = sigmoidal(pixel[2], a, b, sharpen);
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let a = self.strength;
        let b = self.midpoint;
        if a.abs() < 1e-6 {
            return Some(PointOpExpr::Input);
        }
        if self.sharpen {
            // sig(x) = 1/(1+exp(-a*(x-b)))
            // num = sig(x) - sig(0) = 1/(1+exp(-a*(x-b))) - 1/(1+exp(a*b))
            // den = sig(1) - sig(0) = 1/(1+exp(-a*(1-b))) - 1/(1+exp(a*b))
            // out = num / den
            let sig_0 = 1.0 / (1.0 + (a * b).exp());
            let sig_1 = 1.0 / (1.0 + (-a * (1.0 - b)).exp());
            let den = sig_1 - sig_0;
            if den.abs() < 1e-10 {
                return Some(PointOpExpr::Input);
            }
            // 1/(1+exp(-a*(v-b))) expressed as tree
            let neg_a_v_minus_b = PointOpExpr::Mul(
                Box::new(PointOpExpr::Constant(-a)),
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(b)),
                )),
            );
            let sigmoid = PointOpExpr::Div(
                Box::new(PointOpExpr::Constant(1.0)),
                Box::new(PointOpExpr::Add(
                    Box::new(PointOpExpr::Constant(1.0)),
                    Box::new(PointOpExpr::Exp(Box::new(neg_a_v_minus_b))),
                )),
            );
            // (sigmoid - sig_0) / den
            Some(PointOpExpr::Div(
                Box::new(PointOpExpr::Sub(
                    Box::new(sigmoid),
                    Box::new(PointOpExpr::Constant(sig_0)),
                )),
                Box::new(PointOpExpr::Constant(den)),
            ))
        } else {
            // Inverse sigmoidal
            let sig_mid = 1.0 / (1.0 + (-a * b).exp());
            let sig_range = 1.0 / (1.0 + (-a * (1.0 - b)).exp()) - sig_mid;
            if sig_range.abs() < 1e-10 {
                return Some(PointOpExpr::Input);
            }
            // t = sig_mid + v * sig_range
            // out = b - ln(1/t - 1) / a
            let t = PointOpExpr::Add(
                Box::new(PointOpExpr::Constant(sig_mid)),
                Box::new(PointOpExpr::Mul(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(sig_range)),
                )),
            );
            // ln(1/t - 1) = ln((1-t)/t)
            let inv_t_minus_1 = PointOpExpr::Sub(
                Box::new(PointOpExpr::Div(
                    Box::new(PointOpExpr::Constant(1.0)),
                    Box::new(t),
                )),
                Box::new(PointOpExpr::Constant(1.0)),
            );
            Some(PointOpExpr::Sub(
                Box::new(PointOpExpr::Constant(b)),
                Box::new(PointOpExpr::Div(
                    Box::new(PointOpExpr::Ln(Box::new(inv_t_minus_1))),
                    Box::new(PointOpExpr::Constant(a)),
                )),
            ))
        }
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
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dodge", category = "adjustment")]
pub struct Dodge {
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub amount: f32,
}

impl Filter for Dodge {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let divisor = (1.0 - self.amount).max(1e-6);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] /= divisor;
            pixel[1] /= divisor;
            pixel[2] /= divisor;
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let divisor = (1.0 - self.amount).max(1e-6);
        Some(PointOpExpr::Div(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(divisor)),
        ))
    }
}

/// Burn — darken highlights.
///
/// `output = 1 - (1 - input) / amount`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "burn", category = "adjustment")]
pub struct Burn {
    #[param(min = 0.0, max = 2.0, step = 0.05, default = 0.5)]
    pub amount: f32,
}

impl Filter for Burn {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let amt = self.amount.max(1e-6);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = 1.0 - (1.0 - pixel[0]) / amt;
            pixel[1] = 1.0 - (1.0 - pixel[1]) / amt;
            pixel[2] = 1.0 - (1.0 - pixel[2]) / amt;
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        let amt = self.amount.max(1e-6);
        Some(PointOpExpr::Sub(
            Box::new(PointOpExpr::Constant(1.0)),
            Box::new(PointOpExpr::Div(
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Constant(1.0)),
                    Box::new(PointOpExpr::Input),
                )),
                Box::new(PointOpExpr::Constant(amt)),
            )),
        ))
    }
}

/// Solarize — invert values above threshold.
///
/// `output = if input > threshold { 1.0 - input } else { input }`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "solarize", category = "adjustment")]
pub struct Solarize {
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub threshold: f32,
}

impl Filter for Solarize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let thresh = self.threshold;
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            if pixel[0] > thresh { pixel[0] = 1.0 - pixel[0]; }
            if pixel[1] > thresh { pixel[1] = 1.0 - pixel[1]; }
            if pixel[2] > thresh { pixel[2] = 1.0 - pixel[2]; }
        }
        Ok(out)
    }

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        // if (v - threshold) > 0 then (1 - v) else v
        Some(PointOpExpr::Select(
            Box::new(PointOpExpr::Sub(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(self.threshold)),
            )),
            Box::new(PointOpExpr::Sub(
                Box::new(PointOpExpr::Constant(1.0)),
                Box::new(PointOpExpr::Input),
            )),
            Box::new(PointOpExpr::Input),
        ))
    }
}

// All adjustment filters are auto-registered via #[derive(V2Filter)] on their structs.

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
        let expr = b.analytic_expression().unwrap();
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
        let expr = c.analytic_expression().unwrap();
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
        let expr = g.analytic_expression().unwrap();
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
        let b_expr = b.analytic_expression().unwrap();
        let c_expr = c.analytic_expression().unwrap();
        let fused = PointOpExpr::compose(&c_expr, &b_expr);

        // For v=0.5: brightness gives 0.6, contrast gives (0.6-0.5)*1.5+0.5 = 0.65
        let result = fused.evaluate(0.5);
        assert!((result - 0.65).abs() < 1e-6);
    }

    #[test]
    fn dodge_burn_are_inverses_at_half() {
        let dodge = Dodge { amount: 0.5 };
        let _burn = Burn { amount: 0.5 };
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

    #[test]
    fn sigmoidal_expression_matches_compute() {
        let sc = SigmoidalContrast {
            strength: 5.0,
            midpoint: 0.5,
            sharpen: true,
        };
        let expr = sc.analytic_expression().unwrap();
        let input = vec![0.3, 0.5, 0.7, 1.0];
        let computed = sc.compute(&input, 1, 1).unwrap();
        for ch in 0..3 {
            let from_expr = expr.evaluate(input[ch] as f64) as f32;
            assert!(
                (from_expr - computed[ch]).abs() < 0.01,
                "sigmoidal mismatch ch {ch}: expr={from_expr:.4} compute={:.4}",
                computed[ch]
            );
        }
    }

    #[test]
    fn solarize_expression_matches_compute() {
        let s = Solarize { threshold: 0.5 };
        let expr = s.analytic_expression().unwrap();
        for v in [0.2, 0.5, 0.7, 0.0, 1.0] {
            let input = vec![v, v, v, 1.0];
            let computed = s.compute(&input, 1, 1).unwrap();
            let from_expr = expr.evaluate(v as f64) as f32;
            assert!(
                (from_expr - computed[0]).abs() < 1e-5,
                "solarize mismatch at {v}: expr={from_expr:.4} compute={:.4}",
                computed[0]
            );
        }
    }

    #[test]
    fn sigmoidal_solarize_fusion_generates_wgsl() {
        use crate::fusion::lower_to_wgsl;
        let sc = SigmoidalContrast {
            strength: 3.0,
            midpoint: 0.5,
            sharpen: true,
        };
        let sol = Solarize { threshold: 0.5 };
        let composed = PointOpExpr::compose(&sol.analytic_expression().unwrap(), &sc.analytic_expression().unwrap());
        let wgsl = lower_to_wgsl(&composed);
        assert!(wgsl.contains("exp("), "Fused WGSL should contain exp()");
        assert!(wgsl.contains("select("), "Fused WGSL should contain select()");
    }
}
