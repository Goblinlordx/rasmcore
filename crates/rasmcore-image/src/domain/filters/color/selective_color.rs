//! Filter: selective_color (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "selective_color",
    category = "color",
    reference = "hue-range-targeted color adjustment"
)]
pub fn selective_color_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SelectiveColorParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let target_hue = config.target_hue;
    let hue_range = config.hue_range;
    let hue_shift = config.hue_shift;
    let saturation = config.saturation;
    let lightness = config.lightness;

    let params = crate::domain::content_aware::SelectiveColorParams {
        hue_range: crate::domain::content_aware::HueRange {
            center: target_hue,
            width: hue_range,
        },
        hue_shift,
        saturation,
        lightness,
    };
    crate::domain::content_aware::selective_color(pixels, info, &params)
}
