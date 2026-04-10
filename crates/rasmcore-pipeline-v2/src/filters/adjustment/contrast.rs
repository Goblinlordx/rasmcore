use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Contrast adjustment — multiplicative around midpoint.
///
/// `output = (input - 0.5) * factor + 0.5`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "contrast",
    category = "adjustment",
    cost = "O(n)",
    doc = "docs/operations/filters/adjustment/contrast.adoc"
)]
pub struct Contrast {
    /// Contrast multiplier. Positive increases contrast, negative decreases.
    #[param(min = -1.0, max = 1.0, step = 0.02, default = 0.0)]
    pub amount: f32,
}

impl Contrast {
    pub fn to_lmt(&self) -> Lmt {
        let factor = 1.0 + self.amount;
        lmt::analytical_uniform(PointOpExpr::Add(
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

impl Filter for Contrast {
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
    use crate::filters::adjustment::tests::test_pixels;

    #[test]
    fn contrast_expression_matches_compute() {
        let c = Contrast { amount: 0.5 };
        let expr = c.analytic_expression_per_channel().unwrap();
        let input = test_pixels();
        let computed = c.compute(&input, 2, 2).unwrap();
        for i in (0..input.len()).step_by(4) {
            for ch in 0..3 {
                let from_expr = expr[ch].evaluate(input[i + ch] as f64) as f32;
                assert!(
                    (from_expr - computed[i + ch]).abs() < 1e-4,
                    "mismatch at pixel {}, channel {ch}",
                    i / 4
                );
            }
        }
    }
}
