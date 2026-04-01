//! Filter: sepia (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply sepia tone with given `intensity` (0=none, 1=full sepia).
#[rasmcore_macros::register_filter(
    name = "sepia",
    category = "color",
    reference = "sepia tone matrix",
    color_op = "true"
)]
pub fn sepia(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SepiaParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let intensity = config.intensity;

    apply_color_op(pixels, info, &ColorOp::Sepia(intensity.clamp(0.0, 1.0)))
}
