use crate::node::PipelineError;
use crate::ops::Filter;

/// Take absolute value of each RGB channel (useful after subtract).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "evaluate_abs", category = "evaluate")]
pub struct EvaluateAbs;

impl Filter for EvaluateAbs {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0].abs();
            pixel[1] = pixel[1].abs();
            pixel[2] = pixel[2].abs();
        }
        Ok(out)
    }

    // No analytic_expression — abs() not in PointOpExpr variants.
    // Could be added as a new variant, but for now it's a standalone per-pixel op.
}
