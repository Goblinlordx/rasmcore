//! Filter: evaluate_divide

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Divide each channel by a factor (clamped to 0-255).
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
