use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Solarize — invert values above threshold.
///
/// `output = if input > threshold { 1.0 - input } else { input }`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "solarize", category = "adjustment", cost = "O(n)")]
pub struct Solarize {
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub threshold: f32,
}

impl Solarize {
    pub fn to_lmt(&self) -> Lmt {
        lmt::analytical_uniform(PointOpExpr::Select(
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

impl Filter for Solarize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.to_lmt().apply(input))
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        self.to_lmt().to_analytical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solarize_inverts_above_threshold() {
        let s = Solarize { threshold: 0.5 };
        let input = vec![0.3, 0.7, 0.5, 1.0];
        let out = s.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.3).abs() < 1e-6); // below threshold: unchanged
        assert!((out[1] - 0.3).abs() < 1e-6); // above threshold: 1.0 - 0.7 = 0.3
    }

    #[test]
    fn solarize_expression_matches_compute() {
        let s = Solarize { threshold: 0.5 };
        let expr = s.analytic_expression_per_channel().unwrap();
        for v in [0.2, 0.5, 0.7, 0.0, 1.0] {
            let input = vec![v, v, v, 1.0];
            let computed = s.compute(&input, 1, 1).unwrap();
            let from_expr = expr[0].evaluate(v as f64) as f32;
            assert!(
                (from_expr - computed[0]).abs() < 1e-5,
                "solarize mismatch at {v}: expr={from_expr:.4} compute={:.4}",
                computed[0]
            );
        }
    }
}
