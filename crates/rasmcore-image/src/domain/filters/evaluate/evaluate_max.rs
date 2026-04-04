//! Filter: evaluate_max

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Ceiling each channel at a maximum value.
/// Parameters for evaluate_max — ceiling each channel at maximum value.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_max", category = "evaluate", group = "evaluate", variant = "max", reference = "ImageMagick -evaluate Max", point_op = "true")]
pub struct EvaluateMaxParams {
    /// Maximum value (0-255)
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub value: u8,
}
impl LutPointOp for EvaluateMaxParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalMax(self.value))
    }
}

impl CpuFilter for EvaluateMaxParams {
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
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalMax(self.value))
}
}

