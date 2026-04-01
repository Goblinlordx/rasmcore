//! Filter: equalize (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Histogram equalization — maximize contrast via CDF remapping.
#[rasmcore_macros::register_filter(
    name = "equalize",
    category = "enhancement",
    reference = "histogram equalization"
)]
pub fn equalize_registered(
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
    crate::domain::histogram::equalize(pixels, info)
}
