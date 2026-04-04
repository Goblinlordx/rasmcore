//! Filter: evaluate_divide

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Divide each channel by a factor (clamped to 0-255).
/// Parameters for evaluate_divide — divide each channel by factor.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_divide", category = "evaluate", group = "evaluate", variant = "divide", reference = "ImageMagick -evaluate Divide", point_op = "true")]
pub struct EvaluateDivideParams {
    /// Division factor (0.01 to 10)
    #[param(min = 0.01, max = 10.0, step = 0.01, default = 1.0)]
    pub factor: f32,
}
impl LutPointOp for EvaluateDivideParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalDivide(self.factor))
    }
}

impl CpuFilter for EvaluateDivideParams {
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
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalDivide(self.factor))
}
}

