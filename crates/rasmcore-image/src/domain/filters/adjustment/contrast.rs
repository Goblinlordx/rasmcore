//! Filter: contrast (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Adjust contrast (-1.0 to 1.0).
///
/// Uses the composable LUT infrastructure from `point_ops`.
#[rasmcore_macros::register_filter(
    name = "contrast",
    category = "adjustment",
    reference = "multiplicative contrast",
    point_op = "true"
)]
pub fn contrast(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ContrastParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;

    if !(-1.0..=1.0).contains(&amount) {
        return Err(ImageError::InvalidParameters(
            "contrast must be between -1.0 and 1.0".into(),
        ));
    }
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            contrast(r, &mut u, i8, config)
        });
    }
    let lut = crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Contrast(amount));
    crate::domain::point_ops::apply_lut(pixels, info, &lut)
}
