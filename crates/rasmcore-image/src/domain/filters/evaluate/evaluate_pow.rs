//! Filter: evaluate_pow

use crate::domain::filters::common::*;

/// Raise normalized channel values to a power.
#[rasmcore_macros::register_filter(
    name = "evaluate_pow",
    category = "evaluate",
    group = "evaluate",
    variant = "pow",
    reference = "ImageMagick -evaluate Pow",
    point_op = "true"
)]
pub fn evaluate_pow(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &EvaluatePowParams,
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
            evaluate_pow(r, &mut u, i8, config)
        });
    }
    let lut = crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalPow(config.exponent));
    crate::domain::point_ops::apply_lut(pixels, info, &lut)
}
