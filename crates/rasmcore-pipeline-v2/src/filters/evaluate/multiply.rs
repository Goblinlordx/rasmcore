use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

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
