//! Filter: evaluate_max

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Ceiling each channel at a maximum value.

/// Parameters for evaluate_max — ceiling each channel at maximum value.
#[derive(rasmcore_macros::ConfigParams, Clone)]
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

#[rasmcore_macros::register_filter(
    name = "evaluate_max",
    category = "evaluate",
    group = "evaluate",
    variant = "max",
    reference = "ImageMagick -evaluate Max",
    point_op = "true"
)]
pub fn evaluate_max(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &EvaluateMaxParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            evaluate_max(r, &mut u, i8, config)
        });
    }
    let lut = crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalMax(config.value));
    crate::domain::point_ops::apply_lut(pixels, info, &lut)
}
