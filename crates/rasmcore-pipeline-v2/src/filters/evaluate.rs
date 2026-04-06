//! Evaluate math filters — per-channel arithmetic point operations.
//!
//! All implement `analytic_expression()` for automatic fusion: consecutive
//! evaluate ops compose into a single fused LUT or GPU shader. No manual
//! GPU shader needed — the fusion optimizer handles it.

use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Add
// ═══════════════════════════════════════════════════════════════════════════

/// Add a constant to each RGB channel.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_add", category = "evaluate")]
pub struct EvaluateAdd {
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub value: f32,
}

impl Filter for EvaluateAdd {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let v = self.value;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] += v; pixel[1] += v; pixel[2] += v;
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.value)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Subtract
// ═══════════════════════════════════════════════════════════════════════════

/// Subtract a constant from each RGB channel.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_subtract", category = "evaluate")]
pub struct EvaluateSubtract {
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub value: f32,
}

impl Filter for EvaluateSubtract {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let v = self.value;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] -= v; pixel[1] -= v; pixel[2] -= v;
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Sub(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.value)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Multiply
// ═══════════════════════════════════════════════════════════════════════════

/// Multiply each RGB channel by a constant.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_multiply", category = "evaluate")]
pub struct EvaluateMultiply {
    #[param(min = 0.0, max = 4.0, step = 0.01, default = 1.0)]
    pub value: f32,
}

impl Filter for EvaluateMultiply {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let v = self.value;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] *= v; pixel[1] *= v; pixel[2] *= v;
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.value)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Divide
// ═══════════════════════════════════════════════════════════════════════════

/// Divide each RGB channel by a constant.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_divide", category = "evaluate")]
pub struct EvaluateDivide {
    #[param(min = 0.01, max = 4.0, step = 0.01, default = 1.0)]
    pub value: f32,
}

impl Filter for EvaluateDivide {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let v = self.value;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] /= v; pixel[1] /= v; pixel[2] /= v;
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Div(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.value)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Abs
// ═══════════════════════════════════════════════════════════════════════════

/// Take absolute value of each RGB channel (useful after subtract).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_abs", category = "evaluate")]
pub struct EvaluateAbs;

impl Filter for EvaluateAbs {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].abs();
            pixel[1] = pixel[1].abs();
            pixel[2] = pixel[2].abs();
        }
        Ok(out)
    }

    // No analytic_expression — abs() not in PointOpExpr variants.
    // Could be added as a new variant, but for now it's a standalone per-pixel op.
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Pow
// ═══════════════════════════════════════════════════════════════════════════

/// Raise each RGB channel to a power (gamma-like curve).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_pow", category = "evaluate")]
pub struct EvaluatePow {
    #[param(min = 0.1, max = 5.0, step = 0.01, default = 1.0)]
    pub exponent: f32,
}

impl Filter for EvaluatePow {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let e = self.exponent;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].max(0.0).powf(e);
            pixel[1] = pixel[1].max(0.0).powf(e);
            pixel[2] = pixel[2].max(0.0).powf(e);
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Pow(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.exponent)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Log
// ═══════════════════════════════════════════════════════════════════════════

/// Natural logarithm of each RGB channel (shifted by 1: ln(1+v)).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_log", category = "evaluate")]
pub struct EvaluateLog {
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
    pub scale: f32,
}

impl Filter for EvaluateLog {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let s = self.scale;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = (1.0 + pixel[0].max(0.0)).ln() * s;
            pixel[1] = (1.0 + pixel[1].max(0.0)).ln() * s;
            pixel[2] = (1.0 + pixel[2].max(0.0)).ln() * s;
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        // ln(1 + v) * scale
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Ln(Box::new(PointOpExpr::Add(
                Box::new(PointOpExpr::Constant(1.0)),
                Box::new(PointOpExpr::Input),
            )))),
            Box::new(PointOpExpr::Constant(self.scale)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Max
// ═══════════════════════════════════════════════════════════════════════════

/// Clamp each RGB channel to a minimum (max(v, threshold)).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_max", category = "evaluate")]
pub struct EvaluateMax {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub threshold: f32,
}

impl Filter for EvaluateMax {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let t = self.threshold;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].max(t);
            pixel[1] = pixel[1].max(t);
            pixel[2] = pixel[2].max(t);
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Max(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.threshold)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate Min
// ═══════════════════════════════════════════════════════════════════════════

/// Clamp each RGB channel to a maximum (min(v, threshold)).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_min", category = "evaluate")]
pub struct EvaluateMin {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)]
    pub threshold: f32,
}

impl Filter for EvaluateMin {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let t = self.threshold;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].min(t);
            pixel[1] = pixel[1].min(t);
            pixel[2] = pixel[2].min(t);
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Min(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.threshold)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_evaluate_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["evaluate_add", "evaluate_subtract", "evaluate_multiply",
                       "evaluate_divide", "evaluate_abs", "evaluate_pow",
                       "evaluate_log", "evaluate_max", "evaluate_min"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn add_then_subtract_roundtrips() {
        let input = vec![0.5, 0.3, 0.1, 1.0];
        let add = EvaluateAdd { value: 0.2 };
        let sub = EvaluateSubtract { value: 0.2 };
        let mid = add.compute(&input, 1, 1).unwrap();
        let out = sub.compute(&mid, 1, 1).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }

    #[test]
    fn multiply_by_two() {
        let input = vec![0.25, 0.5, 0.75, 1.0];
        let f = EvaluateMultiply { value: 2.0 };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 1.5).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6); // alpha unchanged
    }

    #[test]
    fn pow_has_analytic_expression() {
        let f = EvaluatePow { exponent: 2.2 };
        let exprs = f.analytic_expression_per_channel();
        assert!(exprs.is_some(), "evaluate_pow should have analytic expression for fusion");
    }

    #[test]
    fn max_min_clamp() {
        let input = vec![0.1, 0.5, 0.9, 1.0];
        let max_f = EvaluateMax { threshold: 0.3 };
        let min_f = EvaluateMin { threshold: 0.7 };
        let mid = max_f.compute(&input, 1, 1).unwrap();
        let out = min_f.compute(&mid, 1, 1).unwrap();
        // Should be clamped to [0.3, 0.7]
        assert!((out[0] - 0.3).abs() < 1e-6); // 0.1 → max(0.3) → 0.3 → min(0.7) → 0.3
        assert!((out[1] - 0.5).abs() < 1e-6); // 0.5 → 0.5 → 0.5
        assert!((out[2] - 0.7).abs() < 1e-6); // 0.9 → 0.9 → min(0.7) → 0.7
    }
}
