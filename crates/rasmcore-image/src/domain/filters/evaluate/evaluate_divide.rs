//! Filter: evaluate_divide

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Divide each channel by a factor (clamped to 0-255).

/// Parameters for evaluate_divide — divide each channel by factor.
#[derive(rasmcore_macros::ConfigParams, Clone)]
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

#[rasmcore_macros::register_filter(
    name = "evaluate_divide",
    category = "evaluate",
    group = "evaluate",
    variant = "divide",
    reference = "ImageMagick -evaluate Divide",
    point_op = "true"
)]
pub fn evaluate_divide(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &EvaluateDivideParams,
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
            evaluate_divide(r, &mut u, i8, config)
        });
    }
    let lut = crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalDivide(config.factor));
    crate::domain::point_ops::apply_lut(pixels, info, &lut)
}
