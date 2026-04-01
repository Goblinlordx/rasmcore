//! Filter: evaluate_min

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Floor each channel at a minimum value.
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
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            evaluate_min(r, &mut u, i8, config)
        });
    }
    let lut = crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::EvalMin(config.value));
    crate::domain::point_ops::apply_lut(pixels, info, &lut)
}
