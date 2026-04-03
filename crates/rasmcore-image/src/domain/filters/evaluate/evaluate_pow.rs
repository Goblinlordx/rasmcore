//! Filter: evaluate_pow

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Raise normalized channel values to a power.

/// Parameters for evaluate_pow — raise normalized channel to power.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_pow", category = "evaluate", group = "evaluate", variant = "pow", reference = "ImageMagick -evaluate Pow", point_op = "true")]
pub struct EvaluatePowParams {
    /// Exponent (0.1 to 10)
    #[param(min = 0.1, max = 10.0, step = 0.01, default = 1.0)]
    pub exponent: f32,
}
impl LutPointOp for EvaluatePowParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalPow(self.exponent))
    }
}

impl CpuFilter for EvaluatePowParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalPow(self.exponent))
}
}

