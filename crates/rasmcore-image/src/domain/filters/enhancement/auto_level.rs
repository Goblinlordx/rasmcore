//! Filter: auto_level (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Auto-level — linear stretch from actual min to actual max (no clipping).
#[rasmcore_macros::register_filter(
    name = "auto_level",
    category = "enhancement",
    reference = "automatic black/white point"
)]
pub fn auto_level_registered(
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
    crate::domain::histogram::auto_level(pixels, info)
}
