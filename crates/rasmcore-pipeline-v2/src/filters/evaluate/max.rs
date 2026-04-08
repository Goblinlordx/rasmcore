use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Clamp each RGB channel to a minimum (max(v, threshold)).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_max", category = "evaluate")]
pub struct EvaluateMax {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)]
    pub threshold: f32,
}

impl Filter for EvaluateMax {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let t = self.threshold;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].max(t);
            pixel[1] = pixel[1].max(t);
            pixel[2] = pixel[2].max(t);
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        let expr = PointOpExpr::Max(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(self.threshold)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}
