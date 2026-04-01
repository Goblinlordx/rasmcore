//! Filter: white_balance_gray_world (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Gray world white balance — equalize channel averages.
#[rasmcore_macros::register_filter(
    name = "white_balance_gray_world",
    category = "color",
    group = "white_balance",
    variant = "gray_world",
    reference = "gray world assumption"
)]
pub fn white_balance_gray_world_registered(
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
    crate::domain::color_spaces::white_balance_gray_world(pixels, info)
}
