//! Filter: solarize (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "solarize",
    category = "effect",
    reference = "Man Ray solarization effect",
    point_op = "true"
)]
pub fn solarize(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SolarizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let threshold = config.threshold;

    crate::domain::point_ops::solarize(pixels, info, threshold)
}
