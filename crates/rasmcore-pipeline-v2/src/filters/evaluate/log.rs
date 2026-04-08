use crate::node::PipelineError;
use crate::ops::{Filter, PointOpExpr};

/// Natural logarithm of each RGB channel (shifted by 1: ln(1+v)).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_log", category = "evaluate")]
pub struct EvaluateLog {
    #[param(min = 0.1, max = 10.0, step = 0.1, default = 1.0)]
    pub scale: f32,
}

impl Filter for EvaluateLog {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        let s = self.scale;
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = (1.0 + pixel[0].max(0.0)).ln() * s;
            pixel[1] = (1.0 + pixel[1].max(0.0)).ln() * s;
            pixel[2] = (1.0 + pixel[2].max(0.0)).ln() * s;
        }
        Ok(out)
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        // ln(1 + v) * scale
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Ln(Box::new(PointOpExpr::Add(
                Box::new(PointOpExpr::Constant(1.0)),
                Box::new(PointOpExpr::Input),
            )))),
            Box::new(PointOpExpr::Constant(self.scale)),
        );
        Some([expr.clone(), expr.clone(), expr])
    }
}
