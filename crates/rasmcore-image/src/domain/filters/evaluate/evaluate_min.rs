//! Filter: evaluate_min

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Floor each channel at a minimum value.

/// Parameters for evaluate_min — floor each channel at minimum value.
#[derive(rasmcore_macros::ConfigParams, Clone)]
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

#[rasmcore_macros::register_filter(
    name = "evaluate_min",
    category = "evaluate",
    group = "evaluate",
    variant = "min",
    reference = "ImageMagick -evaluate Min",
    point_op = "true"
)]
pub fn evaluate_min(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &EvaluateMinParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    crate::domain::point_ops::apply_op(pixels, info, &crate::domain::point_ops::PointOp::EvalMin(config.value))
}
