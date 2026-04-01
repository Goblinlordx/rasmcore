//! Filter: modulate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Combined brightness/saturation/hue adjustment in HSB color space.
///
/// IM equivalent: -modulate brightness,saturation,hue
/// Uses HSB (same as HSV where B=V=max(R,G,B)), not HSL.
/// Identity at (100, 100, 0).
#[rasmcore_macros::register_filter(
    name = "modulate",
    category = "color",
    reference = "luma-preserving HSL modulation",
    color_op = "true"
)]
pub fn modulate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ModulateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let brightness = config.brightness;
    let saturation = config.saturation;
    let hue = config.hue;

    apply_color_op(
        pixels,
        info,
        &ColorOp::Modulate {
            brightness: brightness / 100.0,
            saturation: saturation / 100.0,
            hue,
        },
    )
}
