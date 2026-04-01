//! Filter: posterize (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Posterize to N discrete levels per channel (user-facing, LUT-collapsible).
#[rasmcore_macros::register_filter(
    name = "posterize",
    category = "adjustment",
    reference = "bit-depth reduction",
    point_op = "true"
)]
pub fn posterize_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PosterizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let levels = config.levels;

    crate::domain::point_ops::posterize(pixels, info, levels)
}
