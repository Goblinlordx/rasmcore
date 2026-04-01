//! Filter: color_balance (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Photoshop-style color balance — per-tonal-range CMY-RGB adjustment.
///
/// Shifts colors along Cyan-Red, Magenta-Green, and Yellow-Blue axes
/// independently for shadows, midtones, and highlights. Tonal ranges use
/// smooth luminance-based weighting with Rec. 709 luma coefficients.
#[rasmcore_macros::register_filter(
    name = "color_balance",
    category = "adjustment",
    reference = "Photoshop color balance (shadow/midtone/highlight CMY-RGB)",
    color_op = "true"
)]
pub fn color_balance(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ColorBalanceParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            color_balance(r, &mut u, i8, config)
        });
    }
    let cb = crate::domain::color_grading::ColorBalance {
        shadow: [
            config.shadow_cyan_red / 100.0,
            config.shadow_magenta_green / 100.0,
            config.shadow_yellow_blue / 100.0,
        ],
        midtone: [
            config.midtone_cyan_red / 100.0,
            config.midtone_magenta_green / 100.0,
            config.midtone_yellow_blue / 100.0,
        ],
        highlight: [
            config.highlight_cyan_red / 100.0,
            config.highlight_magenta_green / 100.0,
            config.highlight_yellow_blue / 100.0,
        ],
        preserve_luminosity: config.preserve_luminosity,
    };
    crate::domain::color_grading::color_balance(pixels, info, &cb)
}
