//! Filter: evaluate_log

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Logarithmic transform of channel values.

/// Parameters for evaluate_log — logarithmic transform.
#[derive(rasmcore_macros::ConfigParams, Clone)]
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

#[rasmcore_macros::register_filter(
    name = "evaluate_log",
    category = "evaluate",
    group = "evaluate",
    variant = "log",
    reference = "ImageMagick -evaluate Log",
    point_op = "true"
)]
pub fn evaluate_log(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &EvaluateLogParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalLog(config.base))
}
