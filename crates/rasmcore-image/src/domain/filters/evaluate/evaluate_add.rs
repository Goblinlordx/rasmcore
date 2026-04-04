//! Filter: evaluate_add

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Add a constant value to each channel (clamped to 0-255).
/// Parameters for evaluate_add — add constant to each channel.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_add", category = "evaluate", group = "evaluate", variant = "add", reference = "ImageMagick -evaluate Add", point_op = "true")]
pub struct EvaluateAddParams {
    /// Value to add (-255 to 255)
    #[param(min = -255.0, max = 255.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub value: f32,
}
impl LutPointOp for EvaluateAddParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalAdd(self.value as i16))
    }
}

impl CpuFilter for EvaluateAddParams {
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
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalAdd(self.value as i16))
}
}

