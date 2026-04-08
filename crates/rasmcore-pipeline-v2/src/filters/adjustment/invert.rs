use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Invert — channel negation.
///
/// `output = 1.0 - input`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "invert", category = "adjustment", cost = "O(n)")]
pub struct Invert;

impl Invert {
    pub fn to_lmt(&self) -> Lmt {
        lmt::analytical_uniform(PointOpExpr::Sub(
            Box::new(PointOpExpr::Constant(1.0)),
            Box::new(PointOpExpr::Input),
        ))
    }
}

impl Filter for Invert {
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
    fn invert_flips_channels() {
        let input = test_pixels();
        let inv = Invert;
        let out = inv.compute(&input, 2, 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6); // 1.0 - 0.0
        assert!((out[2] - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert_eq!(out[3], 1.0); // alpha unchanged
    }
}
