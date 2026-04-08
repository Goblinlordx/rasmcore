use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Levels — remap input range with gamma.
///
/// `output = ((input - black) / (white - black)) ^ (1/gamma)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "levels", category = "adjustment", cost = "O(n)")]
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

impl Levels {
    pub fn to_lmt(&self) -> Lmt {
        let range = (self.white - self.black).max(1e-6);
        lmt::analytical_uniform(PointOpExpr::Pow(
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

impl Filter for Levels {
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
}
