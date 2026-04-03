//! Filter: invert (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Invert / negate all channels (user-facing, LUT-collapsible).
#[rasmcore_macros::register_filter(
    name = "invert",
    category = "adjustment",
    reference = "channel value inversion",
    point_op = "true"
)]
pub fn invert_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    crate::domain::point_ops::invert(pixels, info)
}
// Note: invert has no config struct (zero-param filter), so GpuFilter
// can't be added here. Invert will get GPU via the derive(Filter) migration
// (invert_v2 already has a params struct).
