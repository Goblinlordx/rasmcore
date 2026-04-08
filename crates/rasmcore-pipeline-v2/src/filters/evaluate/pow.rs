use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

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
