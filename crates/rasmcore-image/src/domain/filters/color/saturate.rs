//! Filter: saturate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Adjust saturation by `factor` (0=grayscale, 1=unchanged, 2=double).
#[rasmcore_macros::register_filter(
    name = "saturate",
    category = "color",
    reference = "HSV saturation scaling",
    color_op = "true"
)]
pub fn saturate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SaturateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let factor = config.factor;

    apply_color_op(pixels, info, &ColorOp::Saturate(factor))
}
