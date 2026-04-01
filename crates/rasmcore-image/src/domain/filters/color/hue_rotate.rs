//! Filter: hue_rotate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Rotate hue by `degrees` (0-360). Works on RGB8 and RGBA8 images.
#[rasmcore_macros::register_filter(
    name = "hue_rotate",
    category = "color",
    reference = "HSV hue rotation",
    color_op = "true"
)]
pub fn hue_rotate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &HueRotateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let degrees = config.degrees;

    apply_color_op(pixels, info, &ColorOp::HueRotate(degrees))
}
