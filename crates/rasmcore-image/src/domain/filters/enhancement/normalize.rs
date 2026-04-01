//! Filter: normalize (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Normalize — linear contrast stretch with 2% black/1% white clipping.
#[rasmcore_macros::register_filter(
    name = "normalize",
    category = "enhancement",
    reference = "min-max normalization to full range"
)]
pub fn normalize_registered(
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
    crate::domain::histogram::normalize(pixels, info)
}
