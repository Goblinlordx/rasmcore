use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Gamma correction — power curve.
///
/// `output = input ^ (1/gamma)` for gamma > 0.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gamma", category = "adjustment", cost = "O(n)")]
pub struct Gamma {
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
    pub gamma: f32,
}

impl Gamma {
    pub fn to_lmt(&self) -> Lmt {
        lmt::analytical_uniform(PointOpExpr::Pow(
            Box::new(PointOpExpr::Max(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.0)),
            )),
            Box::new(PointOpExpr::Constant(1.0 / self.gamma)),
        ))
    }
}

impl Filter for Gamma {
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
    fn gamma_expression_matches_compute() {
        let g = Gamma { gamma: 2.2 };
        let expr = g.analytic_expression_per_channel().unwrap();
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let computed = g.compute(&input, 1, 1).unwrap();
        let from_expr = expr[0].evaluate(0.5) as f32;
        assert!((from_expr - computed[0]).abs() < 1e-4);
    }
}
