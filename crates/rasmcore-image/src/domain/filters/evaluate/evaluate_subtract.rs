//! Filter: evaluate_subtract

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Subtract a constant value from each channel (clamped to 0-255).
/// Parameters for evaluate_subtract — subtract constant from each channel.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_subtract", category = "evaluate", group = "evaluate", variant = "subtract", reference = "ImageMagick -evaluate Subtract", point_op = "true")]
pub struct EvaluateSubtractParams {
    /// Value to subtract (0 to 255)
    #[param(min = 0.0, max = 255.0, step = 1.0, default = 0.0)]
    pub value: f32,
}
impl LutPointOp for EvaluateSubtractParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalSubtract(self.value as i16))
    }
}

impl CpuFilter for EvaluateSubtractParams {
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
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalSubtract(self.value as i16))
}
}

