//! Filter: evaluate_subtract

use crate::domain::filters::common::*;

/// Subtract a constant value from each channel (clamped to 0-255).
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
