//! Filter: split_toning (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "split_toning",
    category = "grading",
    reference = "shadow/highlight hue tinting",
    color_op = "true"
)]
pub fn split_toning_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SplitToningParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let highlight_hue = config.highlight_hue;
    let shadow_hue = config.shadow_hue;
    let balance = config.balance;

    let st = crate::domain::color_grading::SplitToning {
        highlight_color: hue_to_rgb_tint(highlight_hue),
        shadow_color: hue_to_rgb_tint(shadow_hue),
        balance,
        strength: 0.5,
    };
    crate::domain::color_grading::split_toning(pixels, info, &st)
}
