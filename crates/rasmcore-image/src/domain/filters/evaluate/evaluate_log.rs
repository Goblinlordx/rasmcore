//! Filter: evaluate_log

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Logarithmic transform of channel values.
/// Parameters for evaluate_log — logarithmic transform.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "evaluate_log", category = "evaluate", group = "evaluate", variant = "log", reference = "ImageMagick -evaluate Log", point_op = "true")]
pub struct EvaluateLogParams {
    /// Logarithm base (>1)
    #[param(min = 1.01, max = 100.0, step = 0.1, default = 10.0)]
    pub base: f32,
}
impl LutPointOp for EvaluateLogParams {
    fn build_point_lut(&self) -> [u8; 256] {
        crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalLog(self.base))
    }
}

impl CpuFilter for EvaluateLogParams {
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
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalLog(self.base))
}
}

