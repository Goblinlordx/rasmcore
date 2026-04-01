//! Filter: evaluate_subtract

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Subtract a constant value from each channel (clamped to 0-255).

/// Parameters for evaluate_subtract — subtract constant from each channel.
#[derive(rasmcore_macros::ConfigParams, Clone)]
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

#[rasmcore_macros::register_filter(
    name = "evaluate_subtract",
    category = "evaluate",
    group = "evaluate",
    variant = "subtract",
    reference = "ImageMagick -evaluate Subtract",
    point_op = "true"
)]
pub fn evaluate_subtract(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &EvaluateSubtractParams,
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
            evaluate_subtract(r, &mut u, i8, config)
        });
    }
    let lut = crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalSubtract(
        config.value as i16,
    ));
    crate::domain::point_ops::apply_lut(pixels, info, &lut)
}
