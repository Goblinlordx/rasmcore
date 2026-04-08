use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Posterize — reduce to N discrete levels.
///
/// `output = floor(input * levels) / (levels - 1)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "posterize", category = "adjustment", cost = "O(n)")]
pub struct Posterize {
    #[param(min = 2, max = 255, step = 1, default = 4)]
    pub levels: u8,
}

impl Posterize {
    pub fn to_lmt(&self) -> Lmt {
        let n = self.levels as f32;
        let inv = 1.0 / (n - 1.0).max(1.0);
        lmt::analytical_uniform(PointOpExpr::Mul(
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

impl Filter for Posterize {
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
    fn posterize_quantizes() {
        let p = Posterize { levels: 4 };
        let input = vec![0.3, 0.6, 0.9, 1.0];
        let out = p.compute(&input, 1, 1).unwrap();
        // 0.3 * 4 = 1.2 -> floor = 1 -> 1/3 ~ 0.333
        assert!((out[0] - 1.0 / 3.0).abs() < 0.01);
    }
}
