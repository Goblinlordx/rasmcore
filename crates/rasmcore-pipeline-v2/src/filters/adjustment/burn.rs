use crate::lmt::{self, Lmt};
use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Burn — darken highlights.
///
/// `output = 1 - (1 - input) / amount`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "burn", category = "adjustment", cost = "O(n)")]
pub struct Burn {
    #[param(min = 0.0, max = 2.0, step = 0.05, default = 0.5)]
    pub amount: f32,
}

impl Burn {
    pub fn to_lmt(&self) -> Lmt {
        let amt = self.amount.max(1e-6);
        lmt::analytical_uniform(PointOpExpr::Sub(
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

impl Filter for Burn {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.to_lmt().apply(input))
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        self.to_lmt().to_analytical()
    }
}
