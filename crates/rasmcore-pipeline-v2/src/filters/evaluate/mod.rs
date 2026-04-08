//! Evaluate math filters — per-channel arithmetic point operations.
//!
//! All implement `analytic_expression()` for automatic fusion: consecutive
//! evaluate ops compose into a single fused LUT or GPU shader. No manual
//! GPU shader needed — the fusion optimizer handles it.

pub mod add;
pub mod subtract;
pub mod multiply;
pub mod divide;
pub mod abs;
pub mod pow;
pub mod log;
pub mod max;
pub mod min;

pub use add::EvaluateAdd;
pub use subtract::EvaluateSubtract;
pub use multiply::EvaluateMultiply;
pub use divide::EvaluateDivide;
pub use abs::EvaluateAbs;
pub use pow::EvaluatePow;
pub use log::EvaluateLog;
pub use max::EvaluateMax;
pub use min::EvaluateMin;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

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
        assert!((out[0] - 0.3).abs() < 1e-6); // 0.1 -> max(0.3) -> 0.3 -> min(0.7) -> 0.3
        assert!((out[1] - 0.5).abs() < 1e-6); // 0.5 -> 0.5 -> 0.5
        assert!((out[2] - 0.7).abs() < 1e-6); // 0.9 -> 0.9 -> min(0.7) -> 0.7
    }
}
