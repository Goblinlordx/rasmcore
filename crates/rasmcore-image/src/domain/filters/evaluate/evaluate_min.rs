//! Filter: evaluate_min

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Floor each channel at a minimum value.

/// Parameters for evaluate_min — floor each channel at minimum value.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_min", category = "evaluate", group = "evaluate", variant = "min", reference = "ImageMagick -evaluate Min", point_op = "true")]
pub struct EvaluateMinParams {
    /// Minimum value (0-255)
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub value: u8,
}
impl LutPointOp for EvaluateMinParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalMin(self.value))
    }
}

impl CpuFilter for EvaluateMinParams {
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
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalMin(self.value))
}
}

